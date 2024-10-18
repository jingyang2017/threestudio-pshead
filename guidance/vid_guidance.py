from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import PIL
import torch.nn.functional as F
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput
from diffusers.models import AutoencoderKL, UNet3DConditionModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from guidance.clip_camera_projection import CLIPCameraProjection
import kornia
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import TextToVideoSDPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = TextToVideoSDPipeline.from_pretrained(
        ...     "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "Spiderman is surfing"
        >>> video_frames = pipe(prompt).frames
        >>> video_path = export_to_video(video_frames)
        >>> video_path
        ```
"""


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images



class ViVid123Pipeline(TextToVideoSDPipeline):
    r"""
    Pipeline for text-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet3DConditionModel`]):
            A [`UNet3DConditionModel`] to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        novel_view_unet: UNet2DConditionModel,
        image_encoder: CLIPVisionModelWithProjection,
        cc_projection: CLIPCameraProjection,
    ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

        self.register_modules(
            novel_view_unet=novel_view_unet,
            image_encoder=image_encoder,
            cc_projection=cc_projection,
        )
        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor,
        #     do_convert_rgb=True,
        #     do_normalize=True,
        # )

class ViVid123(nn.Module):
    def __init__(self,device,fp16=False,t_range=[0.02,0.98]):
        super().__init__()
        self.device = device
        ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
        VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
        # VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"
        self.precision_t = torch.float16 if fp16 else torch.float32

        zero123_unet = UNet2DConditionModel.from_pretrained(ZERO123_MODEL_ID, subfolder="unet", cache_dir=XDG_CACHE_HOME,torch_dtype=self.precision_t)
        zero123_cam_proj = CLIPCameraProjection.from_pretrained(ZERO123_MODEL_ID, subfolder="clip_camera_projection", cache_dir=XDG_CACHE_HOME,torch_dtype=self.precision_t)
        zero123_img_enc = CLIPVisionModelWithProjection.from_pretrained(ZERO123_MODEL_ID, subfolder="image_encoder", cache_dir=XDG_CACHE_HOME,torch_dtype=self.precision_t)
        # zero123_vae = AutoencoderKL.from_pretrained(ZERO123_MODEL_ID, subfolder="vae", cache_dir=XDG_CACHE_HOME,torch_dtype=self.precision_t)
        # self.zero123_vae = zero123_vae        
        vivid123_pipe = ViVid123Pipeline.from_pretrained(VIDEO_MODEL_ID,
                                                         cache_dir=XDG_CACHE_HOME,
                                                         novel_view_unet=zero123_unet,
                                                         image_encoder=zero123_img_enc,
                                                         cc_projection=zero123_cam_proj,
                                                         torch_dtype=self.precision_t,)
        
        vivid123_pipe.scheduler = DDIMScheduler.from_config(vivid123_pipe.scheduler.config, cache_dir=XDG_CACHE_HOME,torch_dtype=self.precision_t)
        
        # vivid123_pipe.enable_xformers_memory_efficient_attention()
        pipe = vivid123_pipe.to(self.device)

        self.novel_view_unet = pipe.novel_view_unet.eval()
        self.image_encoder = pipe.image_encoder.eval()

        self.cc_projection = pipe.cc_projection.eval()
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.text_encoder = pipe.text_encoder.eval()
        self.encode_prompt = pipe.encode_prompt

        self.scheduler = pipe.scheduler
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device).to(self.precision_t)  
        self.alphas = self.scheduler.alphas_cumprod 
        self.t_range = t_range

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        del pipe

        self.grad_clip_val: Optional[float] = None

        print(f'[INFO] loaded VIVID123 Diffusion!')

    @torch.no_grad()
    def prepare_img_latents(
        self, image, batch_size, dtype, device, generator=None, do_zero123_classifier_free_guidance=False
    ):
        if isinstance(image, list) and isinstance(image[0], torch.Tensor):
                assert image[0].ndim==4,'Image must have 4 dimentions'
                if image[0].min() < -1 or image[0].max() > 1:
                    raise ValueError("Image should be in [-1, 1] range")
                image = torch.cat(image,dim=0)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.mode(generator[i]) for i in range(batch_size)  # sample
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample()
            

        # init_latents = self.vae.config.scaling_factor * init_latents  # todo in original zero123's inference gradio_new.py, model.encode_first_stage() is not scaled by scaling_factor
        if batch_size > init_latents.shape[0]:
            # init_latents = init_latents.repeat(batch_size // init_latents.shape[0], 1, 1, 1)
            num_images_per_prompt = batch_size // init_latents.shape[0]
            # duplicate image latents for each generation per prompt, using mps friendly method
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1)
            init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        # init_latents = torch.cat([init_latents]*2) if do_zero123_classifier_free_guidance else init_latents   # follow zero123
        init_latents = (
            torch.cat([torch.zeros_like(init_latents), init_latents])
            if do_zero123_classifier_free_guidance
            else init_latents
        )

        init_latents = init_latents.to(device=device, dtype=dtype)
        return init_latents

    @torch.no_grad()
    def CLIP_preprocess(self, x):
        dtype = x.dtype
        # following openai's implementation
        # TODO HF OpenAI CLIP preprocessing issue https://github.com/huggingface/transformers/issues/22505#issuecomment-1650170741
        # follow openai preprocessing to keep exact same, input tensor [-1, 1], otherwise the preprocessing will be different, https://github.com/huggingface/transformers/pull/22608
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = kornia.geometry.resize(
            x.to(self.precision_t), (224, 224), interpolation="bicubic", align_corners=True, antialias=False
        ).to(dtype=dtype)
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(
            x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        )
        return x

    # from stable_diffusion_image_variation
    @torch.no_grad()
    def _encode_image(self, image, device, num_images_per_prompt, do_video_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        assert isinstance(image,list), 'image mush be list'
        assert isinstance(image[0],torch.Tensor), 'Image mush be tensor with BCHW'
        assert image[0].ndim==4,'Image must have 4 dimentions'
        if image[0].min() < -1 or image[0].max() > 1: raise ValueError("Image should be in [-1, 1] range")
        
        image = torch.cat(image,dim=0)
        image = image.to(device=device, dtype=dtype)
        image = self.CLIP_preprocess(image)
        image_embeddings = self.image_encoder(image).image_embeds.to(dtype=dtype) #zero123 image_encoder
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_video_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    @torch.no_grad()
    def _encode_pose(self, pose, device, num_images_per_prompt, do_video_classifier_free_guidance):
        dtype = next(self.cc_projection.parameters()).dtype
        if isinstance(pose, torch.Tensor):
            pose_embeddings = pose.unsqueeze(1).to(device=device, dtype=dtype)
        else:
            if isinstance(pose[0], list):
                pose = torch.Tensor(pose)
            else:
                pose = torch.Tensor([pose])
            x, y, z = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1), pose[:, 2].unsqueeze(1)
            pose_embeddings = (
                torch.cat([torch.deg2rad(x), torch.sin(torch.deg2rad(y)), torch.cos(torch.deg2rad(y)), z], dim=-1)
                .unsqueeze(1)
                .to(device=device, dtype=dtype)
            )  # B, 1, 4
        # duplicate pose embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = pose_embeddings.shape
        pose_embeddings = pose_embeddings.repeat(1, num_images_per_prompt, 1)
        pose_embeddings = pose_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_video_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(pose_embeddings)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])
        return pose_embeddings

    @torch.no_grad()
    def _encode_image_with_pose(self, image, pose, device, num_images_per_prompt, do_video_classifier_free_guidance):
        img_prompt_embeds = self._encode_image(image, device, num_images_per_prompt, False)
        # print('img_prompt_embeds', img_prompt_embeds.size()) torch.Size([25, 1, 768])
        pose_prompt_embeds = self._encode_pose(pose, device, num_images_per_prompt, False)
        # print('pose_prompt_embeds', pose_prompt_embeds.size()) torch.Size([25, 1, 4])
        prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
        prompt_embeds = self.cc_projection(prompt_embeds)
        # prompt_embeds = img_prompt_embeds
        # follow 0123, add negative prompt, after projection
        if do_video_classifier_free_guidance:
            negative_prompt = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def train_step_split(
        self,
        pred_img:Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        pred_depth:Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        prompt: Union[str, List[str]] = None,
        guidance_scale_video: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # vivid123 params below
        image: Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        polar: Optional[torch.FloatTensor] = None,
        azimuth: Optional[torch.FloatTensor] = None,
        radius: Optional[torch.FloatTensor] = None,
        guidance_scale_zero123: float = 5.0,
        as_latent:bool=False,
        step_ratio:Optional[tuple[float]]=None,
    ):

        num_frames = pred_img.size(0)
        num_videos_per_image_prompt = 1

        #0. prepare cam_pose_torch
        cam_elevation = torch.deg2rad(polar.view(len(polar),1))
        cam_azimuth = azimuth.view(len(azimuth),1)
        cam_radius = radius.view(len(radius),1)
        cam_azimuth_sin_cos = torch.cat([torch.sin(torch.deg2rad(cam_azimuth)), torch.cos(torch.deg2rad(cam_azimuth))], dim=1)
        cam_pose_torch = torch.cat([cam_elevation, cam_azimuth_sin_cos, cam_radius], dim=1).to(self.device).to(self.precision_t)

        # 1. Prepare images
        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image = image*2-1
        image = image.to(self.precision_t)


        if not as_latent:
            pred_img = F.interpolate(pred_img, (256, 256), mode='bilinear', align_corners=False)
            pred_img = pred_img*2-1
            pred_img = pred_img.to(self.precision_t)

        # do_video_classifier_free_guidance = guidance_scale_video > 1.0
        # do_zero123_classifier_free_guidance = guidance_scale_zero123 > 1.0
        do_video_classifier_free_guidance = guidance_scale_video > 0
        do_zero123_classifier_free_guidance = guidance_scale_zero123 > 0

        # 2. Encode input prompt for video diffusion
        if guidance_scale_video>0:
            with torch.no_grad():
                text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=num_videos_per_image_prompt,
                    do_classifier_free_guidance=do_video_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                )

                #print(prompt_embeds.size(), negative_prompt_embeds.size()) torch.Size([1, 77, 1024]) torch.Size([1, 77, 1024])
                if do_video_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Encode input image for zero123
        if guidance_scale_zero123>0:
            # print('guidance_scale_zero123',guidance_scale_zero123)
            zero123_cond_images = [image for _ in range(num_frames)]
            zero123_embeds = self._encode_image_with_pose( #TODO
                zero123_cond_images,
                cam_pose_torch,
                self.device,
                num_videos_per_image_prompt,
                do_zero123_classifier_free_guidance,
            )  # (2xF) x 1 x 768

            img_latents = self.prepare_img_latents(
                zero123_cond_images,
                batch_size=num_frames,
                dtype=zero123_embeds.dtype,
                device=self.device,
                generator=generator,
                do_zero123_classifier_free_guidance=True,
            )
        
        # TODO 4. Prepare latent variables for diffusion-unet unclear whether use pipe.vae to encode images
        # num_channels_latents = self.unet.config.in_channels
        if as_latent:
            pred_depth = F.interpolate(pred_depth, (256, 256), mode='bilinear', align_corners=False)
            pred_depth = pred_depth*2-1
            # print(pred_depth.max(),pred_depth.min())
            pred_depth = pred_depth.repeat(1,3,1,1).to(self.precision_t)
            latents = self.vae.encode(pred_depth).latent_dist.sample()*self.vae.config.scaling_factor #(25,4,32,32)
        else:
            posterior = self.vae.encode(pred_img).latent_dist
            latents = posterior.sample()*self.vae.config.scaling_factor

        #5. Denoising
        with torch.no_grad():
            noise = torch.randn_like(latents)
            if guidance_scale_video>0:
                noise_video = noise.permute(1,0,2,3).contiguous().unsqueeze(0)
                latents_video = latents.permute(1,0,2,3).contiguous().unsqueeze(0)
                if step_ratio is not None:
                    # dreamtime-like
                    # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                    t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                    t_video = torch.full((latents_video.shape[0],), t, dtype=torch.long, device=self.device)
                else:    
                    t_video = torch.randint(self.min_step, self.max_step + 1,(latents_video.shape[0],), dtype=torch.long, device=self.device)
                latents_noisy = self.scheduler.add_noise(latents_video, noise_video, t_video)
                x_in = torch.cat([latents_noisy] * 2) if do_video_classifier_free_guidance else latents_noisy                
                # predict the noise residual with video diffusion
                noise_pred_video = self.unet(
                    x_in,
                    t_video,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform classifier-free guidance for video diffusion
                if do_video_classifier_free_guidance:
                    noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                    noise_pred_video = noise_pred_video_uncond + guidance_scale_video * (
                        noise_pred_video_text - noise_pred_video_uncond
                    )
                    
            # zero123 denoising
            if guidance_scale_zero123>0:
                latents_zero123 = latents #.clone()
                noise_zero123 = noise #.clone()
                if step_ratio is not None:
                    t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                    t_zero123 = torch.full((latents_zero123.shape[0],), t, dtype=torch.long, device=self.device)
                else:
                    t_zero123 = torch.randint(self.min_step, self.max_step + 1,(latents_zero123.shape[0],), dtype=torch.long, device=self.device)
                latents_noisy = self.scheduler.add_noise(latents_zero123, noise_zero123, t_zero123)
                latent_model_input_zero123 = torch.cat([latents_noisy] * 2) if do_zero123_classifier_free_guidance else latents_noisy
                augmented_latent_model_input_zero123 = torch.cat([latent_model_input_zero123, img_latents],dim=1,).to(self.novel_view_unet.dtype)
                t_in = torch.cat([t_zero123] * 2)
                noise_pred_zero123 = self.novel_view_unet(
                    augmented_latent_model_input_zero123,
                    t_in,
                    encoder_hidden_states=zero123_embeds,
                    return_dict=True,
                ).sample


                if do_zero123_classifier_free_guidance:
                    noise_pred_zero123_uncond, noise_pred_zero123_text = noise_pred_zero123.chunk(2)
                    noise_pred_zero123 = noise_pred_zero123_uncond + guidance_scale_zero123 * (
                        noise_pred_zero123_text - noise_pred_zero123_uncond
                    )

        loss_vivid = 0
        if guidance_scale_video>0:
            noise_pred = noise_pred_video
            w = (1 - self.alphas[t_video]).reshape(-1, 1, 1, 1)
            grad = w * (noise_pred - noise_video)
            target = (latents_video - grad).detach()
            loss_vivid += 0.5* F.mse_loss(latents_video, target,reduction="sum") / noise_pred.size(0)

        if guidance_scale_zero123>0:
            noise_pred = noise_pred_zero123
            w = (1 - self.alphas[t_zero123]).reshape(-1, 1, 1, 1)
            grad = w * (noise_pred - noise)
            target = (latents_zero123 - grad).detach()
            loss_vivid += 0.5* F.mse_loss(latents_zero123, target,reduction="sum") / noise_pred.size(0)

        # print('loss_vivid', loss_vivid.dtype)

        return loss_vivid






if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    # no use now, can only run in fp32
    parser.add_argument('--fp16', action='store_true',
                        help="use float16 for training")

    parser.add_argument('--polar', type=float, default=0,
                        help='delta polar angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0,
                        help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0,
                        help='delta camera radius multiplier in [-0.5, 0.5]')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(
        2, 0, 1).unsqueeze(0).contiguous().to(device)

    print(f'[INFO] loading model ...')
    zero123 = Zero123(device, opt.fp16, opt=opt)

    print(f'[INFO] running model ...')
    outputs = zero123(image, polar=opt.polar,
                      azimuth=opt.azimuth, radius=opt.radius)
    plt.imshow(outputs[0])
    plt.show()