#!/bin/bash
# Declare the lists
# id_list=('0_subject1' '1_subject3' '2_jing' '3_kyle' '4_walter')
# im_list=('subject1_rgba.png' 'head.png' 'jing_rgba.png' 'kyle_rgba.png' 'walter_rgba.png')
# gd_list=('woman' 'man' 'woman' 'man' 'man')

id_list=('5_jing')
im_list=( 'jing_rgba.png')
gd_list=('woman' 'man')

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

    # rm -rf dreambooth/${id}/ldm
    # python utils/get_prompt_blip.py --img-path "dreambooth/${id}/png/${im}" --out-path "dreambooth/${id}/prompt.txt"
done


# id_list=('000121' '004862' '010552' '010619' '019587' '019988' '040052')
# gd_list=('woman' 'woman' 'woman' 'woman' 'woman' 'woman' 'woman')


# # Loop through the lists and run the commands
# for i in "${!id_list[@]}"; do
#     id="${id_list[$i]}"
#     gd="${gd_list[$i]}"
#     im="${id_list[$i]}_rgba.png"
#     echo "Processing ${id} with gender ${gd}, im_name ${im}"

#     # # Run the first Python script
#     # python utils/ldm_utils/main.py -t --data_root "dreambooth/${id}/png/" --logdir "dreambooth/${id}/ldm/" --reg_data_root "data/dreambooth_data/class_${gd}_images/" --bg_root "data/dreambooth_data/bg_images/" --class_word "${gd}" --no-test --gpus 0

#     # # Run the second Python script
#     # python utils/ldm_utils/convert_ldm_to_diffusers.py --checkpoint_path "dreambooth/${id}/ldm/_v1-finetune_unfrozen/checkpoints/last.ckpt" --original_config_file utils/ldm_utils/configs/stable-diffusion/v1-inference.yaml --scheduler_type ddim --image_size 512 --prediction_type epsilon --dump_path "dreambooth/${id}/sd_model"

#     # rm -rf dreambooth/${id}/ldm
#     python utils/get_prompt_blip.py --img-path "dreambooth/${id}/png/${im}" --out-path "dreambooth/${id}/prompt.txt"
# done



# id_list=('000362' '004386' '007216' '007672' '011752' '013417' '014314' '019182' '019679' '022952' '035607' '045246' '047670')
# gd_list=('man' 'man' 'man' 'man' 'man' 'man' 'man' 'man' 'man' 'man' 'man' 'man' 'man')


# # Loop through the lists and run the commands
# for i in "${!id_list[@]}"; do
#     id="${id_list[$i]}"
#     gd="${gd_list[$i]}"
#     im="${id_list[$i]}_rgba.png"
#     echo "Processing ${id} with gender ${gd}, im_name ${im}"

#     # # Run the first Python script
#     # python utils/ldm_utils/main.py -t --data_root "dreambooth/${id}/png/" --logdir "dreambooth/${id}/ldm/" --reg_data_root "data/dreambooth_data/class_${gd}_images/" --bg_root "data/dreambooth_data/bg_images/" --class_word "${gd}" --no-test --gpus 0

#     # # Run the second Python script
#     # python utils/ldm_utils/convert_ldm_to_diffusers.py --checkpoint_path "dreambooth/${id}/ldm/_v1-finetune_unfrozen/checkpoints/last.ckpt" --original_config_file utils/ldm_utils/configs/stable-diffusion/v1-inference.yaml --scheduler_type ddim --image_size 512 --prediction_type epsilon --dump_path "dreambooth/${id}/sd_model"

#     # rm -rf dreambooth/${id}/ldm
#     python utils/get_prompt_blip.py --img-path "dreambooth/${id}/png/${im}" --out-path "dreambooth/${id}/prompt.txt"
# done

