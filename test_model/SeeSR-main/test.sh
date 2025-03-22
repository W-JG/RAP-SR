#SeeSR official
CUDA_VISIBLE_DEVICES=0 \
python test_seesr.py \
--pretrained_model_path stable-diffusion-2-1-base \
--prompt '' \
--seesr_model_path preset/models/seesr \
--unet_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path SR_Dataset/benchmark/RealSR \
--output_dir result/RealSR \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 

#SeeSR + RAP-SR
CUDA_VISIBLE_DEVICES=1 \
python test_seesr.py \
--pretrained_model_path stable-diffusion-2-1-base \
--prompt '' \
--seesr_model_path preset/models/seesr \
--unet_model_path experiment/rap-sr-unet/checkpoint-4000 \
--ram_ft_path preset/models/DAPE.pth \
--image_path SR_Dataset/benchmark/RealSR \
--output_dir result/RealSR_RAP-SR \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 \
--added_prompt "[X]. " \
--negative_prompt "[V]. " \