#!/bin/bash
# Declare the lists
id_list=('1_subject2')
im_list=( 'subject2_rgba.png')
gd_list=('man')

# Loop through the lists and run the commands
for i in "${!id_list[@]}"; do
    id="${id_list[$i]}"
    gd="${gd_list[$i]}"
    im="${id_list[$i]}_rgba.png"
    echo "Processing ${id} with gender ${gd}, im_name ${im}"

    # Run the first Python script
    python utils/ldm_utils/main.py -t --data_root "dreambooth/${id}/png/" --logdir "dreambooth/${id}/ldm/" --reg_data_root "data/dreambooth_data/class_${gd}_images/" --bg_root "data/dreambooth_data/bg_images/" --class_word "${gd}" --no-test --gpus 0

    # Run the second Python script
    python utils/ldm_utils/convert_ldm_to_diffusers.py --checkpoint_path "dreambooth/${id}/ldm/_v1-finetune_unfrozen/checkpoints/last.ckpt" --original_config_file utils/ldm_utils/configs/stable-diffusion/v1-inference.yaml --scheduler_type ddim --image_size 512 --prediction_type epsilon --dump_path "dreambooth/${id}/sd_model"

    rm -rf dreambooth/${id}/ldm
    python utils/get_prompt_blip.py --img-path "dreambooth/${id}/png/${im}" --out-path "dreambooth/${id}/prompt.txt"
done


