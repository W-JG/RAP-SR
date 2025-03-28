
# **RAP-SR: RestorAtion Prior Enhancement in Diffusion Models for Realistic Image Super-Resolution [AAAI2025]**

**Jiangang Wang | [Qingnan Fan](https://fqnchina.github.io/) | Jinwei Chen | Hong Gu | Feng Huang | [Wenqi Ren](https://rwenqi.github.io/)**  
> Shenzhen Campus of Sun Yat-sen University  
> vivo Mobile Communication Co. Ltd  

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.07149-b31b1b.svg)](https://arxiv.org/abs/2412.07149)   ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=W-JG/RAP-SR)

⭐ If you find **RAP-SR** useful for your research or projects, please consider **starring** this repository to support our work. Thank you! 😊

🚩Accepted by AAAI2025

---

## 📜 **Abstract**
> Benefiting from their powerful generative capabilities, pretrained diffusion models have garnered significant attention for real-world image super-resolution (Real-SR). Existing diffusion-based SR approaches typically utilize semantic information from degraded images and restoration prompts to activate prior for producing realistic high-resolution images. However, general-purpose pretrained diffusion models, not designed for restoration tasks, often have suboptimal prior, and manually defined prompts may fail to fully exploit the generated potential. To address these limitations, we introduce RAP-SR, a novel restoration prior enhancement approach in pretrained diffusion models for Real-SR. First, we develop the High-Fidelity Aesthetic Image Dataset (HFAID), curated through a Quality-Driven Aesthetic Image Selection Pipeline (QDAISP). Our dataset not only surpasses existing ones in fidelity but also excels in aesthetic quality. Second, we propose the Restoration Priors Enhancement Framework, which includes Restoration Priors Refinement (RPR) and Restoration-Oriented Prompt Optimization (ROPO) modules. RPR refines the restoration prior using the HFAID, while ROPO optimizes the unique restoration identifier, improving the quality of the resulting images. RAP-SR effectively bridges the gap between general-purpose models and the demands of Real-SR by enhancing restoration prior. Leveraging the plug-and-play nature of RAP-SR, our approach can be seamlessly integrated into existing diffusion-based SR methods, boosting their performance. Extensive experiments demonstrate its broad applicability and state-of-the-art results.

---

## 📢 **Updates**
- **2025/03/21**: Code released.

---

## 📸 **Visual Comparison**

<p align="center">
  <img src="figs/result.png" alt="Performance and Visual Comparison" width="600">
</p>

---

## 🖼️ **High-Fidelity Aesthetic Image Dataset (HFAID)**

Due to copyright restrictions, we can only release the subset of images sourced from public datasets. Images collected from the internet are currently unavailable for download.  
📥 **Download**: [Hugging Face Link](https://huggingface.co/datasets/wangjg333/Rap-SR)  

**Directory Structure:**
```
HFAID.zip
├── image/    # Selected images from public datasets
└── caption/  # Captions generated by Florence-2-large-ft
```

---

## 🔧 **Restoration Priors Enhancement Framework (RPEF)**

### 📂 Step 1: Enter the RPEF folder
```bash
cd RPEF
```

### ⚙️ Step 2: Set up the environment
```bash
conda create -n rap-sr python=3.10
conda activate rap-sr
pip install -r requirements.txt
```

### 📥 Step 3: Download pretrained models
- Download **Stable Diffusion 2.1** from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1)

---

### 🧠 Step 4: Fine-tune Stable Diffusion 2.1

Modify the following fields in `params_train_set.yaml`:
```yaml
pretrained_model_name_or_path: 'stable-diffusion-2-1-base'
top_img_list_txt: 'HFAID/image'
degra_params: 'dataloaders/degradation_setting/params_realesrgan_seesr.yml'
checkpointing_steps: 500
max_train_steps: 4000
```

#### 🚀 Launch training:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
--main_process_port 6255 \
--config_file "./config/hf_config/multigpu_config_2GPUs.yaml" \
train_unet_fine_tune.py \
--params_train_set_path "./config/params_train_set.yaml" \
--output_dir "./experiment/"
```

---

## 🧪 **Evaluation: Test with Diffusion-Based SR Models**

### 📁 Step 1: Navigate to test folder
```bash
cd test_model/SeeSR-main
```

### ⚙️ Step 2: Set up the environment for SeeSR
```bash
conda create -n seesr python=3.8
conda activate seesr
pip install -r requirements.txt
```

### 📥 Step 3: Download dependencies
- **SeeSR** and **DAPE models**:  
  [Google Drive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link)  
  [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/22042244r_connect_polyu_hk/EiUmSfWRmQFNiTGJWs7rOx0BpZn2xhoKN6tXFmTSGJ4Jfw?e=RdLbvg)

📊 Benchmark datasets:  
[DRealSR, RealSR, and DIV2K](https://huggingface.co/datasets/Iceclear/StableSR-TestSets/tree/main)

---

### 🧩 Step 4: Test SeeSR with RAP-SR

We provide pretrained UNet weights for RAP-SR: [Download from Hugging Face](https://huggingface.co/datasets/wangjg333/Rap-SR)

#### Example testing command:
```bash
CUDA_VISIBLE_DEVICES=1 \
python test_seesr.py \
--pretrained_model_path stable-diffusion-2-1-base \
--prompt "" \
--seesr_model_path preset/models/seesr \
--unet_model_path experiment/rap-sr-unet \
--ram_ft_path preset/models/DAPE.pth \
--image_path SR_Dataset/benchmark/RealSR \
--output_dir result/RealSR_RAP-SR \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 \
--added_prompt "[X]. " \
--negative_prompt "[V]. "
```

---

## 📄 **Paper Access**

If you encounter issues accessing the paper on arXiv, you can:
- 📥 Download from Google Drive: [RAP-SR PDF](https://drive.google.com/file/d/1C4IDmI1ZtZdR-Uy6-i5zv2sXa9FzcydC/view?usp=sharing)

---

## 📖 **Citation**
If you find our work useful, please cite it as:
```bibtex
@inproceedings{rap-sr,
  title={RAP-SR: RestorAtion Prior Enhancement in Diffusion Models for Realistic Image Super-Resolution},
  author={Jiangang Wang, Qingnan Fan, Jinwei Chen, Hong Gu, Feng Huang, Wenqi Ren},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

```

---

## 📬 **Contact**

For any questions or collaboration opportunities, feel free to reach out:  
📧 Email: [wangjg33@mail2.sysu.edu.cn](mailto:wangjg33@mail2.sysu.edu.cn)  
Or simply open an issue on this repository.

---

