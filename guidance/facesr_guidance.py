import os
import PIL
import cv2
import kornia
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import threestudio
from threestudio.utils.base import BaseObject
from dataclasses import dataclass, field

@threestudio.register("facesr-guidance")
class codeformer_sr(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        face_detector: str = ""
        net_sr: str = ""

    cfg: Config
    
    def configure(self) -> None:
        upscale_factor = 2
        crop_ratio = (1,1)  # (h, w)
        # the cropped face ratio based on the square face
        face_size = (512,512)

        # standard 5 landmarks for FFHQ faces with 512 x 512 
        # facexlib
        face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                        [201.26117, 371.41043], [313.08905, 371.15118]])
        self.face_template = face_template
        face_detector = torch.load(self.cfg.face_detector)
        face_detector.eval()
        for p in face_detector.parameters():
            p.requires_grad_(False)
        self.face_detector = face_detector.to(self.device)

        net_sr = torch.load(self.cfg.net_sr,'cpu')
        for p in net_sr.parameters():
            p.requires_grad_(False)
        net_sr.eval()
        self.net_sr = net_sr.to(self.device)
    
    def align_cropped(self,rgb): #bhwc
        idx = 0
        input_img_ = np.asarray((rgb[idx].cpu().data)*255.0)
        h, w = input_img_.shape[0:2]
        resize = 640
        scale = resize / min(h, w)
        scale = max(1, scale) # always scale up
        h, w = int(h * scale), int(w * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        input_img = cv2.resize(input_img_, (w, h), interpolation=interp)
        with torch.no_grad():
                bboxes = self.face_detector.detect_faces(input_img)
        if len(bboxes)==1:
            bboxes = bboxes / scale
            bbox = bboxes[0]
            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            tensor_in = rgb[idx].unsqueeze(0).permute(0,3,1,2) #bchw
            # print(tensor_in.size())
            affine_tensor = torch.from_numpy(affine_matrix).unsqueeze(0).to(self.device).to(tensor_in.dtype)
            cropped_face = kornia.geometry.transform.affine(tensor_in, affine_tensor, mode='bilinear', padding_mode='zeros', align_corners=True)
            cropped_face = cropped_face.clamp_(0,1)
        
        img_ = cropped_face.detach().cpu().permute(0, 2, 3, 1).numpy()
        img_ = (img_ * 255).round().astype('uint8')[0]
        # PIL.Image.fromarray(img_).save(f'/home/jy496/work/threestudio/debug/src_{idx}.png')
            
        return cropped_face #bchw


    
    def __call__(self, rgb):
        #rgb: Float[Tensor, "B H W C"]  0-1 RGB
        ## detect face
        # print(rgb.size())
        # loss = 0
        pred_list = []
        sr_list = []
        for idx in range(len(rgb)):
            input_img_ = np.asarray((rgb[idx].cpu().data)*255.0)
            h, w = input_img_.shape[0:2]
            resize = 640
            scale = resize / min(h, w)
            scale = max(1, scale) # always scale up
            h, w = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            input_img = cv2.resize(input_img_, (w, h), interpolation=interp)
            with torch.no_grad():
                bboxes = self.face_detector.detect_faces(input_img)
            if len(bboxes)==1:
                bboxes = bboxes / scale
                bbox = bboxes[0]
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
                affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
                tensor_in = rgb[idx].unsqueeze(0).permute(0,3,1,2)
                # print(tensor_in.size())
                affine_tensor = torch.from_numpy(affine_matrix).unsqueeze(0).to(self.device).to(tensor_in.dtype)
                cropped_face = kornia.geometry.transform.affine(tensor_in, affine_tensor, mode='bilinear', padding_mode='zeros', align_corners=True)
                cropped_face = cropped_face.clamp_(0,1)
                
                cropped_face_ = (cropped_face-0.5)/0.5
                # print(cropped_face_t.size())
                w = 0.5
                with torch.no_grad():
                    output =self.net_sr(cropped_face_, w=w, adain=True)[0]  
                output = output.clamp_(-1,1)
                output = (output+1)/2  #BCHW
                pred_list.append(cropped_face)
                sr_list.append(output)

                # img_ = cropped_face.detach().cpu().permute(0, 2, 3, 1).numpy()
                # img_ = (img_ * 255).round().astype('uint8')[0]
                # PIL.Image.fromarray(img_).save(f'/home/jy496/work/threestudio/debug/in_{idx}.png')
                # img_ = output.detach().cpu().permute(0, 2, 3, 1).numpy()
                # img_ = (img_ * 255).round().astype('uint8')[0]                
                # PIL.Image.fromarray(img_).save(f'/home/jy496/work/threestudio/debug/out_{idx}.png')

        pred_tensor = torch.concatenate(pred_list,dim=0)
        sr_tensor = torch.concatenate(sr_list,dim=0)
        return pred_tensor,sr_tensor

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import PIL
    img_path = '/home/jy496/work/threestudio/custom/threestudio-3dface/results/{head.png}@20240625-094017_ad/save/test/imgs/0003.png'
    input_img_ = cv2.imread(img_path)
    input_img_ = input_img_[:,:,::-1]
    input_tensor = torch.from_numpy(input_img_.copy()).to(torch.float32).unsqueeze(0).cuda()
    input_tensor = input_tensor/255.0
    input_tensor.requires_grad_(True)
    sr = codeformer_sr('cuda')
    loss = sr(input_tensor)
    print(loss)
    loss.backward()
    print(input_tensor.grad)




