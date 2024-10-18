#!/bin/bash

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda-11.8"

seed=0
gpu=0
exp_root_dir='./threestudio/custom/threestudio-3dface'

python launch.py --config custom/threestudio-3dface/configs/face_config.yaml --train --gpu 0 exp_root_dir=/home/jy496/work/threestudio/custom/threestudio-pshead seed=0 data.single_view.image_path=/home/jy496/work/threestudio/custom/threestudio-pshead/face/subject2_rgba.png system.prompt_processor.prompt="photo of a sks man, light blonde hair,detailed strand hair, black v neck shirt, photorealistic, 8K, HDR" system.prompt_processor.negative_prompt="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions, reflections or highlights on the hair" system.guidance.guidance_scale=25 system.prompt_processor_video.prompt="\"\"" system.prompt_processor_video.negative_prompt="\"\"" system.prompt_processor.pretrained_model_name_or_path=dreambooth/2_subject2/sd_model system.guidance.pretrained_model_name_or_path=dreambooth/2_subject2/sd_model system.loss.lambda_sds=0.5 system.loss.lambda_sr=10 system.loss.lambda_lmk=10 system.loss.lambda_zero123=0.5 system.loss.lambda_sds_video=0.01 system.loss.lambda_sds_vivid123=0 system.loss.lambda_id=1
