import cv2
import os
import sys
import random
import imageio
import requests
import numpy as np
import threestudio
import torch
import time
import torchvision
import torch.nn.functional as F
from dataclasses import dataclass, field
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef
from typing import NamedTuple
from .loss_utils import l1_loss, ssim, l2_loss,IDLoss,NvidiaVGG16,sobel_loss,perc,chamfer_distance
from pytorch3d.ops import knn_points
torch.autograd.set_detect_anomaly(True)
from wis3d import Wis3D

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor

@threestudio.register("gsface-system")
class GSFace(BaseLift3DSystem): 
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        prob_multi_view: Optional[float] = None

        #zero123
        guidance_type_zero123: str = ""
        guidance_zero123: dict = field(default_factory=dict)


        #video diffusion
        guidance_type_video: str = ""
        guidance_video: dict = field(default_factory=dict)

        prompt_processor_type_video: str = ""
        prompt_processor_video: dict = field(default_factory=dict)


        #Debug: vivid123 combine zero123 and video
        guidance_type_vivid123: str = ""
        guidance_vivid123: dict = field(default_factory=dict)

        prompt_processor_type_vivid123: str = ""
        prompt_processor_vivid123: dict = field(default_factory=dict)
        
        # MVdream
        guidance_type_multi_view: str = ""
        guidance_multi_view: dict = field(default_factory=dict)

        prompt_processor_type_multi_view: str = ""
        prompt_processor_multi_view: dict = field(default_factory=dict)
        
        # facial landmark guided controlnent
        guidance_type_lmk: str = ""
        guidance_lmk: dict = field(default_factory=dict)

        # codeformer for face super resolution
        guidance_type_srface: str = ""
        guidance_srface: dict = field(default_factory=dict)

        guidance_type_srbg: str = ""

        #identity preserving loss with arcface
        guidance_type_id: str = ""
        guidance_id: dict = field(default_factory=dict)

        # blip for hair
        # guidance_type_hair: str=""
        # hair_prompt: str=""

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        
        # SD
        if self.cfg.loss.lambda_vsd>0 or self.cfg.loss.lambda_sds>0:
            if "stabilityai" in self.cfg.guidance.pretrained_model_name_or_path:
                print("attention use stabilityai")
            print('use image sd loss', self.cfg.loss.lambda_sds)
            self.guidance_single_view = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor_single = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.prompt_utils_single_view = self.prompt_processor_single()

        # MVDream
        if self.cfg.loss.lambda_mvdream>0:
            print('use mvdream loss')
            self.cfg.prompt_processor_multi_view["prompt"] = self.cfg.prompt_processor["prompt"]
            self.guidance_multi_view = threestudio.find(self.cfg.guidance_type_multi_view)(self.cfg.guidance_multi_view)
            self.prompt_processor_multi_view = threestudio.find(self.cfg.prompt_processor_type_multi_view)(self.cfg.prompt_processor_multi_view)
            self.prompt_utils_multi_view = self.prompt_processor_multi_view()
        
        # Zero123
        if self.cfg.loss.lambda_zero123>0:
            self.guidance_zero123 = threestudio.find(self.cfg.guidance_type_zero123)(self.cfg.guidance_zero123)

        # Video
        if self.cfg.loss.lambda_sds_video>0:
            print('video diffusion prompt', self.cfg.prompt_processor_video["prompt"], self.cfg.prompt_processor_video["negative_prompt"])
            self.guidance_video = threestudio.find(self.cfg.guidance_type_video)(self.cfg.guidance_video)
            self.prompt_processor_video = threestudio.find(self.cfg.prompt_processor_type_video)(self.cfg.prompt_processor_video)
            self.prompt_utils_video = self.prompt_processor_video()
        
        if self.cfg.loss.lambda_sds_vivid123>0:
            assert not self.cfg.loss.lambda_zero123
            assert not self.cfg.loss.lambda_sds_video
            
            print('vivid123 diffusion prompt', self.cfg.prompt_processor_vivid123["prompt"], self.cfg.prompt_processor_vivid123["negative_prompt"])
            self.guidance_vivid123 = threestudio.find(self.cfg.guidance_type_vivid123)(self.cfg.guidance_vivid123)
            self.prompt_processor_vivid123 = threestudio.find(self.cfg.prompt_processor_type_vivid123)(self.cfg.prompt_processor_vivid123)
            self.prompt_utils_video = self.prompt_processor_vivid123()
            import torch.nn as nn
            import math
            #25,4,32,32
            self.projector = None
            # class projector(nn.Module):
            #     def __init__(self, in_feature, out_feature):
            #         super().__init__()
            #         self.in_feature = in_feature
            #         self.out_feature = out_feature
            #         self.Connectors = nn.Sequential(
            #             nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            #             nn.BatchNorm2d(out_feature), nn.SiLU(),
            #             nn.Conv2d(out_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False))
                    
            #         for m in self.modules():
            #             if isinstance(m, nn.Conv2d):
            #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #                 m.weight.data.normal_(0, math.sqrt(2. / n))
            #             elif isinstance(m, nn.BatchNorm2d):
            #                 m.weight.data.fill_(1)
            #                 m.bias.data.zero_()
            #     def forward(self, input):
            #         output = self.Connectors(input)
            #         return output
            # self.projector = projector(8,4)
            # print(self.projector)
        
        # Face
        if self.cfg.loss.lambda_lmk>0:
            self.ref_batch = None
            self.guidance_lmk = threestudio.find(self.cfg.guidance_type_lmk)(self.cfg.guidance_lmk)
            self.guidance_lmk.get_text_embeds(self.cfg.prompt_processor.prompt.replace('sks',''), self.cfg.prompt_processor.negative_prompt)
            # self.guidance_lmk = threestudio.find(self.cfg.guidance_type_lmk)(self.cfg.guidance_lmk)
        

        # Facce SR
        if self.cfg.loss.lambda_sr>0:
            print('use face super resolution')
            self.guidance_srface = threestudio.find(self.cfg.guidance_type_srface)(self.cfg.guidance_srface)
            # self.guidance_srbg  = threestudio.find(self.cfg.guidance_type_srbg)(device='cuda')
        
        #  ID
        if self.cfg.loss.lambda_id>0:
            assert self.cfg.loss.lambda_sr>0,print("we need to use its alignment crop")
            self.src_feat = None
            print('use id loss')
            print(self.cfg.guidance_id)
            self.guidance_arcface = threestudio.find(self.cfg.guidance_type_id)(self.cfg.guidance_id)

        # if self.cfg.loss.lambda_hair>0:
        #     self.guidance_hair = threestudio.find(self.cfg.guidance_type_hair)(device='cuda',hair_prompt=self.cfg.hair_prompt)
        
        self.VGG = NvidiaVGG16()
        self.lmk = self.load_lmk()

        self.wis3d = Wis3D("example/visual", 'get_started', xyz_pattern=('x', '-y', '-z'))
        self.automatic_optimization = False

    
    def src_id(self,src_im):
        src_im_cropped = self.guidance_srface.align_cropped(src_im)
        self.src_feat = self.guidance_arcface(src_im_cropped) 
        
    def plot_lmk(self,im,lmk,save_name):
        '''
        im: np.array (H,W,C)
        lmk: np.array (478,2) int
        '''
        radius = 1
        color = (255, 255, 255) 
        thickness = 1
        for x,y in lmk:
            im = cv2.circle(im.astype(np.uint8), (x,y), radius, color, thickness) 
        cv2.imwrite('./debug/'+save_name,im)
    

    def load_lmk(self):
        im_path = self.cfg.guidance_zero123.cond_image_path
        im_name= im_path.split('/')[-1].split('.')[0]
        im = cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im,(512,512))
        lmk = torch.load(f'/home/jy496/work/threestudio/load/face/lmk_478/{im_name}.pkl') #(478,2)
        self.plot_lmk(im,lmk,'./debug/check_input_lmk.png')
        temp = torch.from_numpy(lmk).float().to(self.device)/im.shape[0]
        xy = temp*2-1 #(-1,1)
        return xy

    def lmk2d_to_lmk3d(self,pred_depth, xy_tensor, fovy, c2w):
        '''
        pred_depth: 1, H, W, 1
        '''
        fovy =  fovy.item()
        pred_depth = pred_depth.permute(0,3,1,2)
        cx = 0.5
        cy = 0.5
        sk = 0.0
        fy = 1.0 / (2 * np.tan(fovy / 2))
        fx = 1.0 / (2 * np.tan(fovy / 2))
        xy_01 = (xy_tensor+1)/2
        z_cam = torch.nn.functional.grid_sample(pred_depth,xy_tensor[None,None,...].to(self.device))
        z_cam = z_cam.squeeze()[:,None]
        x_cam = (xy_01[:,0]+0.5/512).float().to(self.device)[:,None]
        y_cam = (xy_01[:,1]+0.5/512).float().to(self.device)[:,None]
        
        x_lift = (x_cam - cx + cy*sk/fy - sk*y_cam/fy) / fx* z_cam
        y_lift = (y_cam - cy) / fy * z_cam
        cam_rel_points = torch.cat((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)
        
        w2c, proj, cam_p = get_cam_info_gaussian(c2w=c2w, fovx=fovy, fovy=fovy, znear=0.1, zfar=100)
        viewpoint_cam = Camera(
            FoVx=fovy,
            FoVy=fovy,
            image_width=512,
            image_height=512,
            world_view_transform=w2c,
            full_proj_transform=proj,
            camera_center=cam_p,
        )
        b = torch.matmul(cam_rel_points,torch.inverse(w2c))
        lmk3d = (b/b[:,3][:,None])[:,:3] #homogeneous coordinates
        return lmk3d, viewpoint_cam.full_proj_transform

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if self.cfg.loss.lambda_sds_vivid123>0 and self.projector is not None:
            optim.add_param_group({'params': self.projector.parameters(), 'lr': 0.0001})
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # optimize the self.triplane
        '''
        triplane_path = self.get_save_path(filename='triplane')
        os.makedirs(triplane_path, exist_ok=True)
        self.trainer.save_checkpoint(f"{triplane_path}/triplane_0.pt")
        self.geometry.triplane.train()
        lr = 1e-3
        num_steps = 2000
        optim = torch.optim.Adam(self.geometry.triplane.parameters(),lr=lr, eps=1e-15)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, verbose=True, factor=0.5)
        # gt_xyz = torch.zeros_like(self.geometry._xyz)
        # gt_xyz = self.geometry._xyz
        # shs = self.geometry.get_features
        features_dc = self.geometry._features_dc
        features_dc = features_dc.clip(-2.0, 2.0)
        features_rest = self.geometry._features_rest
        shs = torch.cat((features_dc, features_rest), dim=1)
        # print(shs.size())
        opacity = self.geometry.get_opacity
        save_flag = False
        for i in range(num_steps):
            if i==num_steps-1:
                save_flag = True
            out = self.geometry.triplane(self.geometry._xyz,save_tri=save_flag)
            loss = torch.nn.functional.l1_loss(out['opacity'],opacity.detach())+torch.nn.functional.l1_loss(out['shs'],shs.detach())
            loss.backward()
            if i ==0:
                print('iter %d  loss %.6f \r'%(i, loss.item()))
            if i%50==0:
                sys.stdout.write('iter %d  loss %.6f  grad %.6f\r'%(i, loss.item(), self.geometry.triplane.embeddings.grad.sum()))
                sys.stdout.flush()
            optim.step()
            optim.zero_grad(set_to_none=True)
            lr_scheduler.step(loss.item())
        print('iter %d  loss %.6f \r'%(i, loss.item()))
        print('finish optimize triplane paramters')
        self.trainer.save_checkpoint(f"{triplane_path}/triplane_{num_steps}.pt")
        torch.save(self.geometry.state_dict(),f'{triplane_path}/geometry_{num_steps}.pt')
        '''

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.train_dataset_single.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        if self.cfg.loss.lambda_depth_rel>0:
            self.pearson = PearsonCorrCoef().to(self.device)
    
    def save_ref(self,rgb,mask):
        self.ref_rgb = rgb
        self.ref_mask = mask
    
    def save_refbatch(self,batch):
        self.ref_batch = batch

    def training_step(self,batch,batch_idx):
        opt = self.optimizers()        
        loss_terms = {}
        def set_loss(name, value):
            loss_terms[f"loss_{name}"] = value

        out = self(batch)
        guidance_inp = out["comp_rgb"]
        guidance_inp_mask = out["comp_mask"]
        guidance_inp_kd = out["comp_rgb_kd"] #None
        assert guidance_inp_kd is None
        
        # depth_loss = tv_loss(out['comp_depth'].permute(0,3,1,2))+tv_loss(guidance_inp.permute(0,3,1,2))
        # set_loss("tv", depth_loss)

        batch_size = batch["c2w"].shape[0]

        if batch["single_view"]:
            if batch['width']==512 and self.cfg.loss.lambda_lmk>0 and self.ref_batch is None:
                self.save_refbatch(batch)
            
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]

            if batch['width']==512 and self.cfg.loss.lambda_id>0 and self.src_feat is None:
                src_img = gt_rgb * gt_mask.float()+(1-gt_mask.float())
                self.src_id(src_img)

            if self.cfg.loss.lambda_cd>0: self.save_ref(gt_rgb,gt_mask)
            gt_rgb = gt_rgb * gt_mask.float()            
            pred_image = guidance_inp * gt_mask.float() # B H W C
            image = pred_image.permute(0,3,1,2) # B C H W
            target = gt_rgb.permute(0,3,1,2)
            Ll1 = l1_loss(image, target)
            Ll2 = l2_loss(image, target)

            # find the cloest 3d points and project it to image plane
            if pred_image.size(1)==512 and self.cfg.loss.lambda_lmk>0 and False:
                lmk3d, full_matrix = self.lmk2d_to_lmk3d(out['comp_depth'], self.lmk, fovy = batch["fovy"][0],c2w = batch["c2w"][0])
                xyz = self.geometry.get_xyz
                # print('lmk3d', 'xyz', lmk3d.size(), xyz.size()) 
                # force it on surface and corresponds the dense facial landmarks
                _, index_batch_3d, _ = knn_points(lmk3d.unsqueeze(0), xyz.unsqueeze(0), K=1, return_nn=True)
                closest_index = index_batch_3d.squeeze()

                #select from max opacity and close distance
                # dist_5, index_batch_3d_5, _ = knn_points(lmk3d.unsqueeze(0), xyz.unsqueeze(0), K=5, return_nn=True)
                # theta_5 =  self.geometry.get_opacity[index_batch_3d_5.squeeze()].squeeze()
                # max_5 = torch.argmax(theta_5,dim=1)
                # closest_index = torch.gather(index_batch_3d_5.squeeze(), 1 , max_5[:,None]).squeeze()

                xyz_lmk3d = xyz[closest_index]
                lmk3d_loss =  F.mse_loss(xyz_lmk3d,lmk3d)
                opacity_loss = -torch.mean(self.geometry.get_opacity[closest_index])

                # print(xyz.size())
                xyz_1 = torch.cat((xyz_lmk3d,torch.ones_like(xyz_lmk3d.data[:,0][:,None])),dim=1)
                temp = torch.matmul(xyz_1,full_matrix)
                lmk2d_pred = (temp[:,:3]/temp[:,2][:,None])[:,:2] #(-1,1)
                # print(lmk2d_pred[closest_index].size(),self.lmk.size())

                lmk2d_loss = F.mse_loss(lmk2d_pred,self.lmk.to(self.device))
                # print('lmk3d_loss: %.6f,lmk2d_loss: %.6f, opacity_loss: %.6f'%(lmk3d_loss.item(),lmk2d_loss.item(),opacity_loss.item()))
                
                lmk2d_pred = (lmk2d_pred+1)/2*pred_image.size(1) 
                # Ll1 = Ll1+lmk3d_loss+opacity_loss+lmk2d_loss

                im_pred = np.array((pred_image[0]*255.0).cpu().data)
                im_pred = np.ascontiguousarray(im_pred, dtype=np.uint8)[:,:,::-1]
                lmk_pred = torch.cat((lmk2d_pred[:,0].cpu().data[:,None],lmk2d_pred[:,1].cpu().data[:,None]),dim=1)
                lmk_pred = np.array(lmk_pred).astype(int)
                self.plot_lmk(im_pred,lmk_pred,'./debug/check_pred_lmk.png')
                    
            # These losses cause extra artifacts
            # Lssim, ssim_map = ssim(image, target)
            # Lssim = 1.0 - Lssim
            # sobel, sobel_image = sobel_loss(image.squeeze(0), target.squeeze(0))
            # lpips = perc(target, image, vgg=self.VGG, downsampling=True)
            # id_loss = self.id_loss_helper(image, target)


            set_loss("rgb", Ll1*1.0+Ll2*1.0)
            
            # set_loss("rgb", Ll1*1.0+Ll2*1.0)
            # set_loss("rgb", Ll1*1.0+Ll2*1.0+Lssim*0.5+lpips*1.0+id_loss*1.0+sobel*0.2)
            # set_loss("rgb", Ll1*0.2+Ll2*0.1+Lssim*0.5+lpips*1.0+id_loss*1.0+sobel*0.2)
            
            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), guidance_inp_mask))
            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["comp_depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["comp_depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )
        else:              
            do_vid_loss = batch.get('is_video', False)
            if self.cfg.loss.lambda_sds_video==0:
                do_vid_loss = False
            if self.cfg.loss.lambda_zero123==0:
                do_vid_loss = True
            
            if self.cfg.loss.lambda_mvdream>0 and batch_size in [4,8]:
                guidance_out_list = [self.guidance_multi_view(guidance_inp_i, self.prompt_utils_multi_view, **batch, rgb_as_latents=False)
                                    for guidance_inp_i in guidance_inp.split(batch_size)]
                guidance_out = {
                    k: torch.zeros_like(v) if torch.is_tensor(v) else v
                    for k, v in guidance_out_list[0].items()
                }
                for guidance_out_i in guidance_out_list:
                    for k, v in guidance_out.items():
                        if torch.is_tensor(v):
                            guidance_out[k] = v + guidance_out_i[k]
                
                for k, v in guidance_out.items():
                    if torch.is_tensor(v):
                        guidance_out[k] = v / len(guidance_out_list)
                
                set_loss("mvdream", guidance_out["loss_sds"])

            # for zero123
            if self.cfg.loss.lambda_zero123>0 and not do_vid_loss:
                guidance_out = self.guidance_zero123(
                        guidance_inp,
                        **batch,
                        rgb_as_latents=False,
                        guidance_eval=True,
                    )

                set_loss("zero123", guidance_out["loss_sds"])


                if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
                    if "comp_normal" not in out:
                        raise ValueError(
                            "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                        )
                    normal = out["comp_normal"]
                    set_loss(
                        "normal_smooth",
                        (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                        + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
                    )

            # for video diffusion
            if self.cfg.loss.lambda_sds_video>0 and do_vid_loss:
                batch['train_dynamic_camera'] = True
                guidance_out = self.guidance_video(guidance_inp, self.prompt_utils_video, **batch, rgb_as_latents=False,num_frames=batch_size)                
                if self.global_step>1000:
                    set_loss("sds_video", guidance_out["loss_sds_video"]*0.01)
                else:
                    set_loss("sds_video", guidance_out["loss_sds_video"])
                
            # for vivid123 diffusion
            if self.cfg.loss.lambda_sds_vivid123>0:
                batch['train_dynamic_camera'] = True
                guidance_out = self.guidance_vivid123(guidance_inp, self.prompt_utils_video,projector=self.projector, **batch, rgb_as_latents=False,num_frames=batch_size)
                set_loss("sds_vivid123", guidance_out["loss_sds_vivid123"])

            
            # for lmk controlnet
            if guidance_inp.size(1)>256 and self.cfg.loss.lambda_lmk>0 and self.ref_batch is not None:
                # run reference image
                out_ref = self(self.ref_batch)
                lmk3d, _ = self.lmk2d_to_lmk3d(out_ref['comp_depth'], self.lmk, fovy = self.ref_batch["fovy"][0],c2w = self.ref_batch["c2w"][0])
                guidance_inp_frontal = guidance_inp[8:-8]
                guidance_lmks_projs = out["full_proj_transforms"][8:-8]
                # save image
                # torch.save(lmk3d,"./debug/lmk3d.pkl")
                # for idx in range(8,16):
                #     xyz_1 = torch.cat((lmk3d,torch.ones_like(lmk3d.data[:,0][:,None])),dim=1)
                #     temp = torch.matmul(xyz_1,out["full_proj_transforms"][idx])
                #     lmk2d_pred = (temp[:,:3]/temp[:,2][:,None])[:,:2] #(-1,1)
                #     pred_image = guidance_inp[idx]
                #     lmk2d_pred = (lmk2d_pred+1)/2*pred_image.size(1) 
                #     im_pred = np.array((pred_image*255.0).cpu().data)
                #     im_pred = np.ascontiguousarray(im_pred, dtype=np.uint8)[:,:,::-1]
                #     lmk_pred = torch.cat((lmk2d_pred[:,0].cpu().data[:,None],lmk2d_pred[:,1].cpu().data[:,None]),dim=1)
                #     lmk_pred = np.array(lmk_pred).astype(int)
                #     self.plot_lmk(im_pred,lmk_pred,f'./debug/check_pred_lmk_new_{idx}.png')
                if self.global_step>1500:
                    guidance_out,lmk2d_preds = self.guidance_lmk.refine(guidance_inp_frontal,guidance_lmks_projs,lmk3d)
                    guidance_out = guidance_out.to(guidance_inp.dtype).to(guidance_inp.device).detach()
                    guidance_inp_frontal = guidance_inp_frontal.permute(0,3,1,2)
                    loss_lmk = perc(guidance_out.detach(),guidance_inp_frontal,vgg=self.VGG, downsampling=True, reshape=False)+l1_loss(guidance_inp_frontal,guidance_out.detach())
                # torchvision.utils.save_image(guidance_inp_frontal.data, './debug/lmk_check_in.png', normalize=True)
                # torchvision.utils.save_image(guidance_out.data, './debug/lmk_check_out.png', normalize=True)

                # for 2d to 3d projection, consistency
                # if self.global_step>0:
                #     if self.global_step<=1500:
                #         loss_lmk = 0
                #         _,lmk2d_preds = self.guidance_lmk.refine(guidance_inp_frontal,guidance_lmks_projs,lmk3d,lmk_only=True)
                #     loss_lmk3d = 0
                #     for i_lmk, lmk2d in enumerate(lmk2d_preds):
                #         lmk3d_i, _ = self.lmk2d_to_lmk3d(out['comp_depth'][i_lmk+8].unsqueeze(0), lmk2d, fovy = batch["fovy"][i_lmk+8],c2w = batch["c2w"][i_lmk+8])
                #         lmk3d_1 = torch.cat((lmk3d_i,torch.ones_like(lmk3d_i)[:,0][:,None].cuda()),dim=1)
                #         xy = torch.matmul(lmk3d_1,out_ref["full_proj_transforms"][0])#w->img plane
                #         xy = (xy[:,:3]/xy[:,2][:,None])[:,:2] #-1-1
                #         # print(xy.device,self.q.device)
                #         # print(min(xy[:,0]),max(xy[:,0]))
                #         # print(min(self.lmk[:,0]),max(self.lmk[:,0]))
                #         xy = xy.clamp(-1,1)
                #         loss_lmk3d+= F.mse_loss(lmk3d_i,lmk3d)+F.mse_loss(xy,self.lmk.to(self.device))
                #     loss_lmk = loss_lmk+loss_lmk3d/len(lmk2d_preds)
                #     if self.global_step%20==0:
                #         vertices = lmk3d
                #         self.wis3d.add_point_cloud(vertices,name='lmk3d')
                set_loss("lmk", loss_lmk)


            if self.cfg.loss.lambda_sr>0 and guidance_inp.size(1)>256 and self.global_step>=1000 and random.random()>0.5: 
                rgb_temp = guidance_inp[8:-8]
                pred_tensor, sr_tensor = self.guidance_srface(rgb_temp)
                loss_sr = perc(sr_tensor.detach(),pred_tensor,vgg=self.VGG, downsampling=True, reshape=False)+l1_loss(pred_tensor,sr_tensor.detach())
                set_loss("sr", loss_sr)
                # torchvision.utils.save_image(rgb_temp.data.permute(0,3,1,2), './debug/sr_in_ori.png', normalize=True)
                # torchvision.utils.save_image(pred_tensor.data, './debug/sr_in.png', normalize=True)
                # torchvision.utils.save_image(sr_tensor.data, './debug/sr_gt.png', normalize=True)
                if self.cfg.loss.lambda_id>0:
                    pred_feats = self.guidance_arcface(pred_tensor)
                    sim = (self.src_feat.detach()) @ pred_feats.t()
                    loss_identity = 1 - sim.mean()
                    set_loss("id", loss_identity)

            # DEBUG=False
            # if self.cfg.loss.lambda_zero123==0 and self.cfg.loss.lambda_sds_video==0 and self.cfg.loss.lambda_sds>0:
            #     selected_index = random.sample(range(len(guidance_inp)), 8)
            #     temp_batch = {key: value[selected_index] if isinstance(value, Tensor) else value for key, value in batch.items()}
            #     temp_rgb = guidance_inp[selected_index]
            #     guidance_out = self.guidance_single_view(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False,flag=True)
            #     set_loss("sds", guidance_out['loss_sds'])
            #     DEBUG = True

            # For ablation studies
            # if self.cfg.loss.lambda_sds>0 and random.random()>0.3 and guidance_inp.size(1)>256:
            #     selected_index = random.sample(range(len(guidance_inp)), 8)
            #     temp_rgb = guidance_inp[selected_index]
            #     temp_batch = {key: value[selected_index] if isinstance(value, Tensor) else value for key, value in batch.items()}
            #     guidance_out = self.guidance_single_view(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False,flag=True)
            # #     set_loss("sds", guidance_out['loss_sds'])

            # if "stabilityai" in self.cfg.guidance.pretrained_model_name_or_path:
            #     if self.cfg.loss.lambda_sds>0:
            #         selected_index = random.sample(range(len(guidance_inp)), 8)
            #         temp_rgb = guidance_inp[selected_index]
            #         temp_batch = {key: value[selected_index] if isinstance(value, Tensor) else value for key, value in batch.items()}
            #         guidance_out = self.guidance_single_view(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False,flag=True)
            #         set_loss("sds", guidance_out['loss_sds'])
            # else:
            # 1500->3000
            if self.cfg.loss.lambda_sds>0 and random.random()>0.3 and guidance_inp.size(1)>256:
                if self.global_step<1000 or self.cfg.loss.lambda_sds_video>0.01:
                    selected_index = random.sample(range(len(guidance_inp)), 8)
                else:
                    all_index = list(range(9))
                    all_index.extend(list(range(15,25)))
                    selected_index  = random.sample(all_index,8) #[0,1,2,3,21,22,23,24]
                
                temp_batch = {key: value[selected_index] if isinstance(value, Tensor) else value for key, value in batch.items()}
                temp_rgb = guidance_inp[selected_index]

                if self.global_step<1000:
                    guidance_out = self.guidance_single_view(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False,flag=False)
                    set_loss("sds", guidance_out['loss_sds'])
                    # print(guidance_out['loss_sds'])
                else:
                    refine_img = self.guidance_single_view.refine(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False)
                    temp_rgb = temp_rgb.permute(0,3,1,2)
                    loss_sds = perc(refine_img.detach(),temp_rgb,vgg=self.VGG, downsampling=True, reshape=False)+l1_loss(temp_rgb,refine_img.detach())
                    # torchvision.utils.save_image(guidance_inp.data.permute(0,3,1,2), './debug/all_check.png', normalize=True)
                    # torchvision.utils.save_image(temp_rgb.data, './debug/hair_check.png', normalize=True)
                    # torchvision.utils.save_image(refine_img, './debug/hair_check_refine.png', normalize=True)
                    set_loss("sds", loss_sds*300)
            
                # DEBUG refinenetwork
                # if self.cfg.loss.lambda_sds>0 and random.random()>0.3 and guidance_inp.size(1)>256 and not DEBUG:
                #     selected_index = random.sample(range(len(guidance_inp)), 8)
                #     temp_batch = {key: value[selected_index] if isinstance(value, Tensor) else value for key, value in batch.items()}
                #     temp_rgb = guidance_inp[selected_index]
                    
                #     if self.global_step<1000:
                #         guidance_out = self.guidance_single_view(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False,flag=False)
                #         set_loss("sds", guidance_out['loss_sds'])
                #     else:
                #         refine_img = self.guidance_single_view.refine(temp_rgb, self.prompt_utils_single_view, **temp_batch, rgb_as_latents=False)
                #         temp_rgb = temp_rgb.permute(0,3,1,2)
                #         loss_sds = perc(refine_img.detach(),temp_rgb,vgg=self.VGG, downsampling=True, reshape=False)+l1_loss(temp_rgb,refine_img.detach())
                #         set_loss("sds", loss_sds*100)


            if self.cfg.loss.lambda_cd>0 and random.random()>0.5:
                loss_cd = 0
                input_pixels = self.ref_rgb[0][self.ref_mask.squeeze()>0.9]
                # a = time.time()
                for idx in range(25):
                    pred_pixels = guidance_inp[idx][guidance_inp_mask[idx].squeeze()>0.9]
                    loss_cd = loss_cd+chamfer_distance(input_pixels.unsqueeze(0), pred_pixels.unsqueeze(0), single_directional=False)[0]
                # print(time.time()-a)
                set_loss("cd", loss_cd)

        total_loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith('loss_'):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace('loss_', "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                total_loss += loss_weighted

        total_loss.backward()
        iteration = self.global_step

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        viewspace_point_tensor = out["viewspace_points"]

        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )

        opt.step()
        opt.zero_grad(set_to_none=True)
        return {"loss": total_loss}
    


    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_video(out["comp_rgb"],'val',self.true_global_step)
        if out["comp_rgb_kd"] is not None:
            self.save_video(out["comp_rgb_kd"],'val_kd',self.true_global_step)


    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_video(out["comp_rgb"],'test',self.true_global_step)
        if out["comp_rgb_kd"] is not None:
            self.save_video(out["comp_rgb_kd"],'test_kd',self.true_global_step)

    def on_test_epoch_end(self):
        pass
    

