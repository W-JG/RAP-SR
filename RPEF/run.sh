
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch  \
--main_process_port 6255 \
--config_file "./config/hf_config/multigpu_config_2GPUs.yaml" \
train_unet_fine_tune.py \
--params_train_set_path "./config/params_train_set.yaml" \
--output_dir "./experiment/" \








