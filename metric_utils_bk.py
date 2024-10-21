# * evaluate use laion/CLIP-ViT-H-14-laion2B-s32B-b79K
# best open source clip so far: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
# code adapted from NeuralLift-360

import torch
import torch.nn as nn
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image
# import torchvision.transforms as transforms
import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from os.path import join as osp
import argparse
import pandas as pd
import contextual_loss as cl
# import contextual_loss.fuctional as CF
from insightface.app import FaceAnalysis

class CLIP(nn.Module):

    def __init__(self,
                 device,
                 clip_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
                 size=224):  #'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        super().__init__()
        self.size = size
        self.device = f"cuda:{device}"

        clip_name = clip_name

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-base-patch32')

        self.normalize = transforms.Normalize(
            mean=self.feature_extractor.image_mean,
            std=self.feature_extractor.image_std)

        self.resize = transforms.Resize(224)
        self.to_tensor = transforms.ToTensor()

        # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        # assert len(novel_views) == 100
        clip_scores = []
        for novel in novel_views:
            clip_scores.append(self.score_from_path(ref_img_path, [novel]))
        return np.mean(clip_scores)

    # * recommend to use this function for evaluation
    # def score_gt(self, ref_paths, novel_paths):
    #     clip_scores = []
    #     for img1_path, img2_path in zip(ref_paths, novel_paths):
    #         clip_scores.append(self.score_from_path(img1_path, img2_path))

    #     return np.mean(clip_scores)

    def similarity(self, image1_features: torch.Tensor,
                   image2_features: torch.Tensor) -> float:
        with torch.no_grad(), torch.cuda.amp.autocast():
            y = image1_features.T.view(image1_features.T.shape[1],
                                       image1_features.T.shape[0])
            similarity = torch.matmul(y, image2_features.T)
            # print(similarity)
            return similarity[0][0].item()

    def get_img_embeds(self, img):
        if img.shape[0] == 4:
            img = img[:3, :, :]

        img = self.aug(img).to(self.device)
        img = img.unsqueeze(0)  # b,c,h,w

        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()
        # print(img)

        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1,
                                         keepdim=True)  # normalize features
        return image_z

    def score_from_feature(self, img1, img2):
        img1_feature, img2_feature = self.get_img_embeds(
            img1), self.get_img_embeds(img2)
        # for debug
        return self.similarity(img1_feature, img2_feature)

    def read_img_list(self, img_list):
        size = self.size
        images = []
        # white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print(img_path)
            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)  # Convert BGRA to BGR
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

            # plt.imshow(img)
            # plt.show()

            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        # images = images.astype(np.float32)

        return images

    def score_from_path(self, img1_path, img2_path):
        img1, img2 = self.read_img_list(img1_path), self.read_img_list(img2_path)
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        # plt.imshow(img1)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()

        img1, img2 = self.to_tensor(img1), self.to_tensor(img2)
        # print("img1 to tensor ",img1)
        return self.score_from_feature(img1, img2)


def numpy_to_torch(images):
    images = images * 2.0 - 1.0
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    return images.cuda()


class LPIPSMeter:

    def __init__(self,
                 net='alex',
                 device=None,
                 size=224):  # or we can use 'alex', 'vgg' as network
        self.size = size
        self.net = net
        self.results = []
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

    def read_img_list(self, img_list):
        size = self.size
        images = []
        white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        images = images.astype(np.float32) / 255.0

        return images

    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        for path0, path1 in zip(ref_paths, novel_paths):
            # Load images
            # img0 = lpips.im2tensor(lpips.load_image(path0)).cuda() # RGB image from [-1,1]
            # img1 = lpips.im2tensor(lpips.load_image(path1)).cuda()
            img0, img1 = self.read_img_list([path0]), self.read_img_list(
                [path1])
            img0, img1 = numpy_to_torch(img0), numpy_to_torch(img1)
            # print(img0.shape,img1.shape)
            img0 = F.interpolate(img0,
                                    size=(self.size, self.size),
                                    mode='area')
            img1 = F.interpolate(img1,
                                    size=(self.size, self.size),
                                    mode='area')

            # for debug vis
            # plt.imshow(img0.cpu().squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()
            # plt.imshow(img1.cpu().squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()
            # equivalent to cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA

            # print(img0.shape,img1.shape)

            self.results.append(self.fn.forward(img0, img1).cpu().numpy())

        return self.measure()


class PSNRMeter:

    def __init__(self, size=800):
        self.results = []
        self.size = size

    def read_img_list(self, img_list):
        size = self.size
        images = []
        white_background = np.ones((size, size, 3), dtype=np.uint8) * 255
        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        images = images.astype(np.float32) / 255.0
        # print(images.shape)
        return images

    def update(self, preds, truths):
        # print(preds.shape)

        psnr_values = []
        # For each pair of images in the batches
        for img1, img2 in zip(preds, truths):
            # Compute the PSNR and add it to the list
            # print(img1.shape,img2.shape)

            # for debug
            # plt.imshow(img1)
            # plt.show()
            # plt.imshow(img2)
            # plt.show()

            psnr = compare_psnr(
                img1, img2,
                data_range=1.0)  # assuming your images are scaled to [0,1]
            # print(f"temp psnr {psnr}")
            psnr_values.append(psnr)

        # Convert the list of PSNR values to a numpy array
        self.results = psnr_values

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

    # * recommend to use this function for evaluation
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds = self.read_img_list(ref_paths)
        truths = self.read_img_list(novel_paths)
        self.update(preds, truths)
        return self.measure()


class CLMeter:
    def __init__(self, size=224):
        self.results = []
        self.size = size
        # self.criterion = cl.ContextualLoss().cuda()
        self.cx_model = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4').cuda()

    
    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        # assert len(novel_views) == 100
        clip_scores = []
        for novel in novel_views:
            clip_scores.append(self.score_from_path(ref_img_path, [novel]))
        return np.mean(clip_scores)
    

    def read_img_list(self, img_list):
        size = self.size
        images = []
        white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        images = images.astype(np.float32) / 255.0

        return images
    
    def score_from_path(self, img1_path, img2_path):
        img1, img2 = self.read_img_list(img1_path), self.read_img_list(img2_path)
        img1, img2 = numpy_to_torch(img1), numpy_to_torch(img2)

        # print("img1 to tensor ",img1)
        img1 = F.interpolate(img1,
                        size=(self.size, self.size),
                        mode='area')
        img2 = F.interpolate(img2,
                        size=(self.size, self.size),
                        mode='area')
        return self.cx_model(img1, img2).cpu().numpy()


class FRMeter: 
    def __init__(self, size=224):
        self.results = []
        self.size = size
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        # assert len(novel_views) == 100
        clip_scores = []
        for novel in novel_views:
            clip_scores.append(self.score_from_path(ref_img_path[0], novel))
        # clip_scores = [i.item() for i in clip_scores if i>0]
        # print(clip_scores)
        return np.mean(clip_scores)
    

    def score_from_path(self, img1_path, img2_path):
        image = cv2.imread(img2_path)
        faces = self.app.get(image)
        if len(faces)==0:
            image = cv2.resize(image, (128,128))
            top = 50
            bottom = 50
            left = 50
            right = 50
            border_type = cv2.BORDER_CONSTANT
            padding_color = [0, 0, 0]  # Black padding (BGR format)
            # Pad the image
            image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=padding_color)
            faces = self.app.get(image)
            if len(faces)==0:
                return 1
            else:
                return 0.8
        else:
            return 0.8


# results organization 2
def score_from_method_for_dataset(my_scorer,
                                  input_path,
                                  pred_path,
                                  score_type='clip', 
                                  rgb_name='lambertian', 
                                  result_folder='results/images', 
                                  first_str='*0000*'
                                  ):  # psnr, lpips
    scores = {}
    final_res = 0
    
    examples = os.listdir(input_path)
    for i in range(len(examples)):
        # ref path
        ref_path = osp(input_path, examples[i], 'rgba.png') 
        if score_type == 'clip':
            # compare entire folder for clip
            novel_view = glob.glob(osp(pred_path,'*'+examples[i]+'*', result_folder, f'*{rgb_name}*'))


            print(f'[INOF] {score_type} loss for example {examples[i]} between 1 GT and {len(novel_view)} predictions')
        else:
            # compare first image for other metrics
            novel_view = glob.glob(osp(pred_path, '*'+examples[i]+'*/', result_folder, f'{first_str}{rgb_name}*'))
            print(f'[INOF] {score_type} loss for example {examples[i]} between {ref_path} and {novel_view}')
        # breakpoint()
        score_i = my_scorer.score_gt([ref_path], novel_view)
        scores[examples[i]] = score_i
        final_res += score_i
    avg_score = final_res / len(examples)
    scores['average'] = avg_score
    return scores

def score_from_method_for_dataset_yj(my_scorer,
                                  method,
                                  score_type):  # psnr, lpips
    scores = {}
    final_res = 0
    input_path = "/home/jy496/work/threestudio/ICLR_Vis/refs21/faces/"
    examples = glob.glob(input_path+'/*_rgba.png')
    examples.sort()

    if method=='ours':
        img_list = glob.glob("/home/jy496/work/threestudio/outputs/results_0927/results_refine/**/*/images",recursive=True)
        img_list.sort()
        ref_index = 0 
  
    elif method == "erad3d":
        img_list = []
        for ref_path in examples:
            id_name = os.path.basename(ref_path).replace(".png","")
            # print(id_name)
            img_list.append(f'./era3d/{id_name}/save/images/')
        ref_index = 0
    elif method =="dreamgaussian":
        img_list = []
        for ref_path in examples:
            id_name = os.path.basename(ref_path).replace("_rgba.png","")
            img_list.append(f'./dreamgaussian/{id_name}/images/')
        ref_index = 121
    elif method == "panohead":
        examples_new = []
        img_list = []
        for ref_path in examples:
            id_name = os.path.basename(ref_path).replace("_rgba.png","")
            img_list.append(f'./panohead/{id_name}/images/')
            examples_new.append((f'./panohead/{id_name}/{id_name}/target.png'))
        examples = examples_new
        ref_index = None
    
    elif method=="magic123":
        examples_new = []
        img_list = []
        for ref_path in examples:
            id_name = os.path.basename(ref_path).replace("_rgba.png","")
            img_list.append(f'./magic123-nerf-dmtet/face/magic123_{id_name}_nerf_dmtet/results/images/')
            examples_new.append(f'/home/jy496/work/Magic123/data/face/{id_name}/{id_name}_rgba.png')
        examples = examples_new
        ref_index = 0


    for i in range(len(examples)):
        # ref path
        ref_path = examples[i]
        if score_type in ['clip','cl','fr']:
            if method == "magic123":
                novel_view = glob.glob(f"{img_list[i]}/*lambertian.jpg")
                novel_view.sort()
            else:
                novel_view = glob.glob(f"{img_list[i]}/*.png")
                novel_view.sort()
            if score_type=='fr':
                if method == 'ours':
                    temp = novel_view[120]
                elif method=='dreamgaussian':
                    temp = novel_view[0]
                elif method =="erad3d":
                    temp = novel_view[30]
                    print(temp)
                elif method == "panohead":
                    temp = novel_view[120]
                elif method == 'magic123':
                    temp = novel_view[120]
                novel_view = [temp]

            print(f'[INOF] {score_type} loss for example {examples[i]} between 1 GT and {len(novel_view)} predictions')
        
        else:
            raise NotImplementedError
        
        score_i = my_scorer.score_gt([ref_path], novel_view)
        scores[examples[i]] = score_i
        final_res += score_i
    avg_score = final_res / len(examples)
    scores['average'] = avg_score
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to accept three string arguments")
    parser.add_argument("--method",
                        default="panohead",
                        help="Specify the input path")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="Specify the GPU device to be used")
    parser.add_argument("--save_dir", type=str, default='all_metrics/results21_backside/')
    args = parser.parse_args()

    fr_scorer = FRMeter()

    os.makedirs(args.save_dir, exist_ok=True)
    results_dict = {}
    
    results_dict['fr'] = score_from_method_for_dataset_yj(
    fr_scorer, args.method, 'fr')
    
    df = pd.DataFrame(results_dict)
    df.to_csv(f"{args.save_dir}/{args.method}.csv")