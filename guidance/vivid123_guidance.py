import os
import cv2
import threestudio
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
import tqdm
import torchvision
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from dataclasses import dataclass, field

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder
        # del model.first_stage_model.encoder


    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("vivid123-guidance")
class Vivid123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = False #True
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None
        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"
        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)
        view_dependent_prompting: bool = True
        low_ram_vae: int = -1

        pretrained_model_name_or_path_zero123: str = "load/zero123/stable-zero123.ckpt"
        pretrained_config_zero123: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = True
        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2
        guidance_scale_zero123: float = 5.0

    cfg: Config

    def configure(self) -> None:

        threestudio.info(f"Loading Video Diffusion ...")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        # Extra modules
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=self.weights_dtype,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=self.weights_dtype,
        )
        self.text_encoder = self.text_encoder.to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        # self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        # self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        # Extra for latents
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        threestudio.info(f"Loaded Video Diffusion!")

        threestudio.info(f"Loading Stable Zero123 ...")
        self.config = OmegaConf.load(self.cfg.pretrained_config_zero123)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path_zero123,
            device=self.device,
            vram_O=True,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)


        self.grad_clip_val: Optional[float] = None

        self.prepare_embeddings(self.cfg.cond_image_path)

        threestudio.info(f"Loaded Stable Zero123!")

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn, self.c_concat = self.get_img_embeds(self.rgb_256)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat
    
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c_crossattn=None,
        c_concat=None,
        **kwargs,
    ) -> dict:
        T = torch.stack(
            [
                torch.deg2rad(
                    (90 - elevation) - (90 - self.cfg.cond_elevation_deg)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.deg2rad(
                    90 - torch.full_like(elevation, self.cfg.cond_elevation_deg)
                ),
            ],
            dim=-1,
        )[:, None, :].to(self.device)
        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (self.c_crossattn if c_crossattn is None else c_crossattn).repeat(
                        len(T), 1, 1
                    ),
                    T,
                ],
                dim=-1,
            )
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat)
                    .repeat(len(T), 1, 1, 1)
                    .to(self.device),
                    (self.c_concat if c_concat is None else c_concat).repeat(
                        len(T), 1, 1, 1
                    ),
                ],
                dim=0,
            )
        ]
        return cond


    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 576"], normalize: bool = True
    ) -> Float[Tensor, "B 4 40 72"]:
        # breakpoint()
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        # breakpoint()
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        # breakpoint()

        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                posterior_mask = torch.cat(
                    [
                        self.vae.encode(
                            imgs[~mask_vae][i : i + 1].to(self.weights_dtype)
                        ).latent_dist.sample()
                        for i in range(imgs.shape[0] - vnum)
                    ],
                    dim=0,
                )
            posterior = torch.cat(
                [
                    self.vae.encode(
                        imgs[mask_vae][i : i + 1].to(self.weights_dtype)
                    ).latent_dist.sample()
                    for i in range(vnum)
                ],
                dim=0,
            )
            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor

        latents = (
            latents[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        cond: dict,
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            # print(latent_model_input.size()) #torch.Size([2, 4, 25, 32, 32]) 
            # print(t.size()) #torch.Size([1]) 
            # print(text_embeddings.size()) #should be torch.Size([2, 77, 1024])

            # for vid diffusion
            noise_pred_vid = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # for zero123 diffusion
            # need reshape
            # print(latent_model_input.size()) #torch.Size([2, 4, 25, 32, 32]) 
            x_in = latent_model_input.permute(0,2,1,3,4).reshape(-1,4,32,32)
            t_in = torch.cat([t]*50)
            # print(t_in.size()) #torch.Size([50])
            noise_pred_zero123 = self.model.apply_model(x_in, t_in, cond)
            # print(noise_pred_zero123.size()) #torch.Size([50, 4, 32, 32])


        # perform guidance (high scale from paper!)
        # print('noise_pred_vid',noise_pred_vid.size())
        noise_pred_text_vid, noise_pred_uncond_vid = noise_pred_vid.chunk(2) 
        # print(noise_pred_uncond.size()) #([1, 4, 25, 32, 32])
        noise_pred_vid = noise_pred_text_vid + self.cfg.guidance_scale * (
                noise_pred_text_vid - noise_pred_uncond_vid
            )
        
        if self.cfg.guidance_scale==0:
            # print('self.cfg.guidance_scale is 0')
            noise_pred_vid = noise

        noise_pred_uncond_zero123, noise_pred_cond_zero123 = noise_pred_zero123.chunk(2)
        noise_pred_zero123 = noise_pred_uncond_zero123 + self.cfg.guidance_scale_zero123 * (
                noise_pred_cond_zero123 - noise_pred_uncond_zero123
            ) #25*4*32*32
        
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # print(noise_pred_zero123.size(),noise_pred_vid.size())
        # We use spade to align two distributions
        # #(1,4,25,32,32)->(25,4,32,32)
        # noise_pred_vid = adaptive_instance_normalization(noise_pred_vid.squeeze(0).permute(1,0,2,3),noise_pred_zero123)
        # noise_pred_vid = noise_pred_vid[None,:].permute(0,2,1,3,4)

        return w, noise_pred_zero123, noise_pred_vid,noise
        # grad = w * (noise_pred_zero123.permute(1,0,2,3)[None,:] - noise_pred_vid)
        # return grad

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        num_frames: int = 16,
        projector: torch.nn.Module = None,
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)        
        batch_size = rgb_BCHW.shape[0] // num_frames #batch_size==1
        latents: Float[Tensor, "B 4 40 72"]
        
        if kwargs["train_dynamic_camera"]:
            elevation_ = elevation[[0]]
            azimuth_ = azimuth[[0]]
            camera_distances_ = camera_distances[[0]]
        
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (40, 72), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_256 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            rgb_BCHW_256 = rgb_BCHW_256.permute(1, 0, 2, 3)[None]
            # print(rgb_BCHW_256.size())
            latents = self.encode_images(rgb_BCHW_256)
            # print('latents',latents.size())

        
        # print('elevation',elevation.size(),self.cfg.view_dependent_prompting)
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation_, azimuth_, camera_distances_, self.cfg.view_dependent_prompting
        )

        cond = self.get_cond(elevation, azimuth, camera_distances)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        ) #for vid batch_size==1 while for zero123 batch_size=25
        
        # we only debug sds here YJ
        w, noise_pred_zero123, noise_pred_vid,noise = self.compute_grad_sds(latents, text_embeddings, cond, t)
        if projector is not None:
            print("projector is not None")
            noise_pred_vid = noise_pred_vid.squeeze(0).permute(1,0,2,3)
            noise_pred = noise_pred_zero123+projector(torch.cat((noise_pred_zero123, noise_pred_vid),dim=1))
            grad = w*(noise_pred[None,:].permute(0,2,1,3,4)-noise)
        else:
            # print("projector is None")
            grad = w * (noise_pred_zero123.permute(1,0,2,3)[None,:] - noise) #+0.5*w * (noise_pred_vid - noise)
            # grad = 0.9*w * (noise_pred_zero123.permute(1,0,2,3)[None,:] - noise)+0.1*w * (noise_pred_vid - noise)
            # grad = 0.0*w * (noise_pred_zero123.permute(1,0,2,3)[None,:] - noise)+1.0*w * (noise_pred_vid - noise)
            
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds_vivid123": loss_sds,
            "grad_norm": grad.norm(),
        }
    

    @torch.no_grad()
    def refine(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        num_frames: int = 16,
        init_step: int = 10,
        guidance_scale_zero123: int = 3,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)        
        batch_size = rgb_BCHW.shape[0] // num_frames #batch_size==1
        latents: Float[Tensor, "B 4 40 72"]
        if kwargs["train_dynamic_camera"]:
            elevation_ = elevation[[0]]
            azimuth_ = azimuth[[0]]
            camera_distances_ = camera_distances[[0]]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (40, 72), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_256 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            rgb_BCHW_256 = rgb_BCHW_256.permute(1, 0, 2, 3)[None]
            latents = self.encode_images(rgb_BCHW_256) #torch.Size([1, 4, 25, 32, 32]) 
            # print(latents.size())
            # video_tensor = self.decode_latents(latents).detach()[0].permute(1,0,2,3)     
            # torchvision.utils.save_image(video_tensor, f"./debug/vivid123.png", normalize=True)


        text_embeddings = prompt_utils.get_text_embeddings(
            elevation_, azimuth_, camera_distances_, self.cfg.view_dependent_prompting
        )
        cond = self.get_cond(elevation, azimuth, camera_distances)
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)


        t = torch.tensor([init_step], dtype=torch.long,device=self.device)
        
        do_video_classifier_free_guidance = self.cfg.guidance_scale > 1.0 
        do_zero123_classifier_free_guidance = self.cfg.guidance_scale_zero123 > 1.0

        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t)
        
        # print(init_step,num_inference_steps)
        for _, t in enumerate(self.scheduler.timesteps[init_step:]):
            # print(_, t)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_video_classifier_free_guidance else latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_ = t.unsqueeze(0).to(self.device)
            tt = torch.cat([t_] * 2)

            noise_pred_video = self.forward_unet(latent_model_input,tt,encoder_hidden_states=text_embeddings)
            # perform classifier-free guidance for video diffusion
            if do_video_classifier_free_guidance:
                noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                noise_pred_video = noise_pred_video_uncond + self.cfg.guidance_scale * (
                    noise_pred_video_text - noise_pred_video_uncond
                ) ##([1, 4, 25, 32, 32])

            # zero123 denoising
            x_in = latent_model_input.permute(0,2,1,3,4).reshape(-1,4,32,32)
            t_in = torch.cat([t_]*50)
            noise_pred_zero123 = self.model.apply_model(x_in, t_in, cond) 

            if do_zero123_classifier_free_guidance:
                noise_pred_zero123_uncond, noise_pred_zero123_text = noise_pred_zero123.chunk(2)
                # noise_pred_zero123 = noise_pred_zero123_uncond + self.cfg.guidance_scale_zero123 * (
                #     noise_pred_zero123_text - noise_pred_zero123_uncond
                # ) ##25*4*32*32

                noise_pred_zero123 = noise_pred_zero123_uncond + guidance_scale_zero123 * (
                    noise_pred_zero123_text - noise_pred_zero123_uncond
                ) ##25*4*32*32

                noise_pred_zero123 = noise_pred_zero123.unsqueeze(0).permute(0,2,1,3,4) #([1, 4, 25, 32, 32])


            # fusing video diffusion with zero123
            noise_pred = 0.5 * noise_pred_video + 0.5 * noise_pred_zero123

            # reshape latents
            bsz, channel, frames, width, height = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # reshape latents back
            latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

        video_tensor = self.decode_latents(latents)      
        # print(video_tensor.size()) 
        return video_tensor.detach()[0].permute(1,0,2,3)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

