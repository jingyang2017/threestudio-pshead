import copy
import os
import torch
import glob
import PIL
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from os.path import isfile
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# suppress partial model loading warning
logging.set_verbosity_error()
import dataclasses
import math
from typing import List, Mapping, Optional, Tuple, Union
import threestudio

import cv2
import matplotlib.pyplot as plt
import numpy as np
from threestudio.utils.base import BaseObject
from dataclasses import dataclass, field

_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(
    color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad
)
right_eye_draw = DrawingSpec(
    color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad
)
right_eyebrow_draw = DrawingSpec(
    color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad
)
left_iris_draw = DrawingSpec(
    color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad
)
left_eye_draw = DrawingSpec(
    color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad
)
left_eyebrow_draw = DrawingSpec(
    color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad
)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw


def draw_landmarks(
    image: np.ndarray,
    landmark_list: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = face_connection_spec.keys(),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = face_connection_spec,
):

    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    idx_to_coordinates = {i: landmark_list[i] for i in range(len(landmark_list))}
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            drawing_spec = (
                connection_drawing_spec[connection]
                if isinstance(connection_drawing_spec, Mapping)
                else connection_drawing_spec
            )
            cv2.line(
                image,
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                drawing_spec.color,
                drawing_spec.thickness,
            )
    return image[:, :, ::-1]  # flip BGR


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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

@threestudio.register("mp-guidance")
class LMKDiffusion(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        sd_version: str="2.1"

    cfg: Config

    def configure(self) -> None:
        fp16=True
        vram_O=False
        sd_version=self.cfg.sd_version
        t_range=[0.02, 0.98]
        print(f'[INFO] loading stable diffusion...')
        self.dtype = torch.float16 if fp16 else torch.float32
        # Create model
        model_key = 'stabilityai/stable-diffusion-2-1-base'
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, 
            safety_checker=None, 
            torch_dtype=self.dtype,
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
        
        pipe.to(self.device)

        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.unet = pipe.unet.eval()
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)


        mp_controlnet = ControlNetModel.from_pretrained(
            "CrucibleAI/ControlNetMediaPipeFace",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        
        del pipe

        self.controlnet = mp_controlnet.eval()
        for p in self.controlnet.parameters():
            p.requires_grad_(False)
            
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.dtype)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        # base_options = python.BaseOptions(model_asset_path='/home/jy496/work/dreamgaussian/face_landmarker_v2_with_blendshapes.task')
        # options = vision.FaceLandmarkerOptions(base_options=base_options,
        #                     output_face_blendshapes=True,
        #                     output_facial_transformation_matrixes=True,
        #                     num_faces=1)

        # self.landmark_detector = vision.FaceLandmarker.create_from_options(options)
        print(f'[INFO] loaded ControlNetMediaPipeFace diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    def train_step(self,pred_rgb, guidance_scale=100, control_scale=1, as_latent=False, grad_clip=None, control=None):
        B = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        control = control.to(self.dtype)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [B], dtype=torch.long, device=self.device)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # Landmark based controlnet
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            control = torch.cat([control] * 2)
            
            # print(latent_model_input.size())
            # print(self.text_embeddings.size())
            # print(control.size())
            # print(tt.size())

            text_embeddings = self.text_embeddings.repeat(B,1,1)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control,
                conditioning_scale=control_scale,
                return_dict=False,
            )

            # predict the noise residual with landmark control
            noise_pred = self.unet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        # w(t), sigma_t^2
        # w = (1 - self.alphas[t])
        # perform guidance (high scale from paper!)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w.view(-1, 1, 1, 1) * (noise_pred - noise)
        # import numpy as np
        # np.save("grad.npy", grad.detach().cpu().numpy())

        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        # print('mp', loss.dtype)
        return loss 

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                #torch.save(t, "produce_latents_t.pt")
                #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def sdedit(self, data_dir, height=512, width=512, num_inference_steps=50,test_data_dir = None, guidance_scale=7.5,control_scale=1):
        base_options = python.BaseOptions(model_asset_path='/home/jy496/work/dreamgaussian/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
        landmark_detector = vision.FaceLandmarker.create_from_options(options)
        render_resolution = 512
        noise_level = 200
        res_dir = data_dir
        origin_data_dir = os.path.join(res_dir, 'data')
        if not os.path.exists(origin_data_dir):
            print('no data dir')
            return
        update_data_dir = os.path.join(res_dir, 'update_data')
        os.makedirs(update_data_dir, exist_ok=True)
        if len(glob.glob(origin_data_dir + '/*.png')) == len(glob.glob(update_data_dir + '/*.png')):
            print('already done')
            return
        
        print('gen data for ', res_dir)
        name = os.path.basename(res_dir)

        # Prompts -> text embeds
        prompts = 'A portrait of a men, with a beard and mustache, wearing a red shirt, photorealistic, 8K, HDR.'
        negative_prompts = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution.'
        self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]
        text_embeddings = self.text_embeddings.repeat(1,1,1)
        
        for image_path in glob.glob(origin_data_dir + '/*.png'):
            im_name = image_path.split('/')[-1].split('.')[0]
            image = PIL.Image.open(image_path).convert('RGB')
            cv_mat = np.array(image)
            origin_img = torch.from_numpy(cv_mat).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # --> 0,1
            origin_img = origin_img / 255.0
            origin_img = origin_img.to(self.dtype)
            latents = self.encode_imgs(origin_img)
            print(latents.size())
            #generate control
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
            # STEP 4: Detect face landmarks from the input image.
            detection_result = landmark_detector.detect(image)
            x_list = []
            y_list = []
            # z_list = []
            for j in range(len(detection_result.face_landmarks[0])):
                x_list.append(detection_result.face_landmarks[0][j].x)
                y_list.append(detection_result.face_landmarks[0][j].y)
                # z_list.append(detection_result.face_landmarks[0][i].z)
            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            # z_arr = np.array(z_list)
            xy = np.concatenate((x_arr[None,:],y_arr[None,:]),axis=0)
            xy = xy*render_resolution
            xy_ = xy.transpose(1,0).astype(int)
            image = draw_landmarks(np.zeros_like(cv_mat), xy_)
            cv2.imwrite(f'{update_data_dir}/lmk_{im_name}.png',image)

            image = torch.from_numpy(image.copy()).float().to(self.device)
            control = image.permute(2, 0, 1) / 255.0
            control = control.unsqueeze(0)
            control = control.to(self.dtype)

            t = torch.tensor([noise_level], dtype=torch.long,device=self.device)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latents = self.denoise_latents(text_embeddings, noise_level, num_inference_steps=num_inference_steps,
                                               guidance_scale=guidance_scale, latents=latents_noisy,control=control, control_scale=control_scale)

            # Img latents -> imgs
            img = self.decode_latents(latents)  # [1, 3, 512, 512]
            # Img to Numpy
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            img = (img * 255).round().astype('uint8')[0]
            PIL.Image.fromarray(img).save(os.path.join(update_data_dir, os.path.basename(image_path)))

    def detect_lmk(self, rgb): 
        image = np.asarray((rgb.cpu().data)*255.0)
        cv_mat = np.ascontiguousarray(image.astype(np.uint8))
        #generate control
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
        # STEP 4: Detect face landmarks from the input image.
        result = self.landmark_detector.detect(image)
        return result

    def __call__(self, rgb, num_inference_steps=50, guidance_scale=7.5,control_scale=1):
        #rgb: Float[Tensor, "B H W C"]

        render_resolution = 512
        noise_level = 200
        
        # Prompts -> text embeds
        prompts = 'A portrait of a men, with a beard and mustache, wearing a red shirt, photorealistic, 8K, HDR.'
        negative_prompts = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution.'
        self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]
        text_embeddings = self.text_embeddings.repeat(1,1,1)
        img_clean = []
        idx_sel = []
        for idx in range(len(rgb)):
            image = np.asarray((rgb[idx].cpu().data)*255.0)
            # im_name = image_path.split('/')[-1].split('.')[0]
            # image = PIL.Image.open(image_path).convert('RGB')
            # cv_mat = np.ascontiguousarray(image[:,:,::-1].astype(np.uint8))
            cv_mat = np.ascontiguousarray(image.astype(np.uint8))
            origin_img = torch.from_numpy(cv_mat).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # --> 0,1
            origin_img = origin_img / 255.0
            origin_img = origin_img.to(self.dtype)
            latents = self.encode_imgs(origin_img)

            #generate control
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
            # STEP 4: Detect face landmarks from the input image.
            detection_result = self.landmark_detector.detect(image)
            if len(detection_result.face_landmarks)>0:
                idx_sel.append(idx)
            else:
                continue
            x_list = []
            y_list = []
            # z_list = []
            for j in range(len(detection_result.face_landmarks[0])):
                x_list.append(detection_result.face_landmarks[0][j].x)
                y_list.append(detection_result.face_landmarks[0][j].y)
                # z_list.append(detection_result.face_landmarks[0][i].z)
            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            # z_arr = np.array(z_list)
            xy = np.concatenate((x_arr[None,:],y_arr[None,:]),axis=0)
            xy = xy*render_resolution
            xy_ = xy.transpose(1,0).astype(int)
            image = draw_landmarks(np.zeros_like(cv_mat), xy_)
            # cv2.imwrite(f'{update_data_dir}/lmk_{im_name}.png',image)

            image = torch.from_numpy(image.copy()).float().to(self.device)
            control = image.permute(2, 0, 1) / 255.0
            control = control.unsqueeze(0)
            control = control.to(self.dtype)
            t = torch.tensor([noise_level], dtype=torch.long,device=self.device)
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latents = self.denoise_latents(text_embeddings, noise_level, num_inference_steps=num_inference_steps,
                                               guidance_scale=guidance_scale, latents=latents_noisy,control=control, control_scale=control_scale)

            # Img latents -> imgs
            img = self.decode_latents(latents)  # [1, 3, 512, 512]
            # Img to Numpy
            if idx==0:
                img_ = img.detach().cpu().permute(0, 2, 3, 1).numpy()
                img_ = (img_ * 255).round().astype('uint8')[0]
                PIL.Image.fromarray(img_).save(f'/home/jy496/work/threestudio/debug/{idx}.png')

            img_clean.append(img)
        output = torch.concatenate(img_clean,dim=0)
        return output,idx_sel
    
    def denoise_latents(self, text_embeddings, start_t,num_inference_steps=50, guidance_scale=7.5, latents=None,control=None,control_scale=1):
        self.scheduler.set_timesteps(num_inference_steps)
        control = torch.cat([control] * 2)

        for _, t in enumerate(self.scheduler.timesteps):
            # print(t)
            # print(latents.size())
            if t>start_t:
                continue
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
            # print(t)
            t = t.unsqueeze(0).to(self.device)
            tt = torch.cat([t] * 2)
            # print(latent_model_input.device)
            # print(tt.device)
            # print(text_embeddings.device)
            # print(self.controlnet.device)
            # print(control.device)
            # print(control_scale)

            # print(latent_model_input.dtype)
            # print(tt.dtype)
            # print(text_embeddings.dtype)
            # # print(self.controlnet.dtyoe)
            # print(control.dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control,
                conditioning_scale=control_scale,
                return_dict=False,
            )
            # predict the noise residual with landmark control
            noise_pred = self.unet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )['sample']


            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            latents = latents.to(self.dtype)

        return latents

    @torch.no_grad()
    def refine(self, rgbs, guidance_lmks_projs, lmk3d, num_inference_steps=50, guidance_scale=7.5,control_scale=1,lmk_only=False):
        #rgb: Float[Tensor, "B H W C"]
        render_resolution = 512
        noise_level = 200
        # prompts = 'A portrait of a men, with a beard and mustache, wearing a red shirt, photorealistic, 8K, HDR.'
        # negative_prompts = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution.'
        # self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]
        text_embeddings = self.text_embeddings.repeat(1,1,1)
        img_clean = []
        # idx_sel = []
        lmk_list = []
        for idx in range(len(rgbs)):
            image = np.asarray((rgbs[idx].cpu().data)*255.0)
            cv_mat = np.ascontiguousarray(image.astype(np.uint8))
            origin_img = torch.from_numpy(cv_mat).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # --> 0,1
            origin_img = origin_img / 255.0
            origin_img = origin_img.to(self.dtype)
            latents = self.encode_imgs(origin_img)
            # print(latents.size())
            lmk3d_1 = torch.cat((lmk3d,torch.ones_like(lmk3d)[:,0][:,None]),dim=1)
            xy = torch.matmul(lmk3d_1,guidance_lmks_projs[idx])#w->img plane
            xy = (xy[:,:3]/xy[:,2][:,None])[:,:2] #-1-1
            lmk_list.append(xy)
            if lmk_only:
                continue
            xy = (xy+1)/2*render_resolution
            xy_ = np.asarray(xy.cpu().data).astype(int)
            # save 2d projection
            image = draw_landmarks(np.zeros_like(cv_mat), xy_)
            # Save image
            # cv2.imwrite(f'/home/jy496/work/threestudio/debug/lmk_{idx}.png',image)
            image = torch.from_numpy(image.copy()).float().to(self.device)
            control = image.permute(2, 0, 1) / 255.0
            control = control.unsqueeze(0)
            control = control.to(self.dtype)
            t = torch.tensor([noise_level], dtype=torch.long,device=self.device)
            # predict the noise residual with unet, NO grad!
            
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latents = self.denoise_latents(text_embeddings, noise_level, num_inference_steps=num_inference_steps,
                                               guidance_scale=guidance_scale, latents=latents_noisy,control=control, control_scale=control_scale)

            # Img latents -> imgs
            img = self.decode_latents(latents)  # [1, 3, 512, 512]
            # Save image
            # img_ = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            # img_ = (img_ * 255).round().astype('uint8')[0]
            # PIL.Image.fromarray(img_).save(f'/home/jy496/work/threestudio/debug/{idx}.png')

            img_clean.append(img)
        
        if lmk_only:
            return None, lmk_list
        output = torch.concatenate(img_clean,dim=0)
        return output, lmk_list
  
if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    import PIL

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,help='Network pickle filename', required=True)
    parser.add_argument('--test_data_dir', type=str,help='test_data_dir', required=True)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('-gs', type=int, default=7.5)
    parser.add_argument('-cs', type=int, default=1)
    opt = parser.parse_args()
    seed_everything(opt.seed)
    device = torch.device('cuda')
    sd = LMKDiffusion(device)
    imgs = sd.sdedit(opt.data_dir,opt.H, opt.W, opt.steps,opt.test_data_dir,guidance_scale=opt.gs,control_scale=opt.cs)

# if __name__ == '__main__':

#     import argparse
#     import matplotlib.pyplot as plt

#     parser = argparse.ArgumentParser()
#     parser.add_argument('prompt', type=str)
#     parser.add_argument('--negative', default='', type=str)
#     parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
#     parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
#     parser.add_argument('--fp16', action='store_true', help="use float16 for training")
#     parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
#     parser.add_argument('-H', type=int, default=512)
#     parser.add_argument('-W', type=int, default=512)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--steps', type=int, default=50)
#     opt = parser.parse_args()

#     seed_everything(opt.seed)

#     device = torch.device('cuda')

#     sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

#     imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

#     # visualize image
#     plt.imshow(imgs[0])
#     plt.show()
