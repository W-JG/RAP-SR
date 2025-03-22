import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import json
from transformers import CLIPTokenizer
from dataloaders.realesrgan import RealESRGAN_degradation
import numpy as np
import yaml
from collections import OrderedDict
import pickle

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def opt_parse(opt_path):
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt


class Fine_tune_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        opt,
        tokenizer=None,
        device="cpu",
    ):

        super().__init__()
        #opt = opt_parse(opt_path)

        self.tokenizer = tokenizer

        top_img_list_txt = opt["top_img_list_txt"]
        gen_folder_path = opt["gen_folder_path"]
        lr_img_folder = opt["lr_img_folder"]
        top_ration = opt["top_ration"]
        lr_ration = opt["lr_ration"]
        degra_params = opt["degra_params"]
        resolution = opt["resolution"]
        self.null_text_ratio = opt["null_text_ratio"]
        self.wo_x_v_prompt = ( opt["wo_x_v_prompt"] if "wo_x_v_prompt" in opt.keys() else False)
        self.dataset_name = ( opt["dataset_name"] if "dataset_name" in opt.keys() else None)

        #with open(top_img_list_txt) as f:
        self.hq_file_list =  [os.path.join(top_img_list_txt, file) for file in os.listdir(top_img_list_txt)]
        #with open(lr_img_folder) as f:
        self.lr_file_list = ''
        
        self.gen_file_list = ''

        self.top_ration = top_ration
        self.lr_ration = lr_ration
        self.gen_ration = 1 - self.top_ration - self.lr_ration

        self.degradation = RealESRGAN_degradation(degra_params, device=device)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution,interpolation=Image.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_crop = transforms.Compose(
                [
                    transforms.Resize(resolution,interpolation=Image.BICUBIC),
                    transforms.CenterCrop(resolution),
                ]
            )

    def __len__(self):
        return len(self.hq_file_list)


    def __getitem__(self, index):
        select_ration = random.random()
        output = {}
        if select_ration < self.top_ration:
            image_path = self.hq_file_list[index]
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
            if self.dataset_name is None:
                caption = self.get_caption_value(image_path)
            if not self.wo_x_v_prompt:
                caption = "[X]. " + caption
            flag = 'HQ'
        elif select_ration < (self.top_ration + self.lr_ration):
            # pass
            #image_path = random.choice(self.lr_file_list)
            image_path = self.hq_file_list[index]
            image = Image.open(image_path).convert("RGB")
            image = self.image_crop(image)
            GT_image_t, LR_image_t = self.degradation.degrade_process(
                np.asarray(image)/255., resize_bak=True
            )
            image_tensor = LR_image_t.squeeze(0) * 2 -1    # .squeeze(0)
            if self.dataset_name is None:
                caption = self.get_caption_value(image_path)
            if not self.wo_x_v_prompt:
                caption = "[V]. " + caption

            flag = 'LQ'

        if random.random() < self.null_text_ratio:
            caption = ""

        tokenized_text = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        output["input_ids"] = tokenized_text
        output["pixel_values"] = image_tensor
        output["captions"] = caption
        return output

    def get_file_list(self,folder_paths):
        file_list = []
        if isinstance(folder_paths, str):
            file_list += [
                os.path.join(folder_paths, item)
                for item in os.listdir(folder_paths)
                if item.endswith(".png") or item.endswith(".jpg")
            ]
        else:
            for folder_path in folder_paths:
                file_list += [
                    os.path.join(folder_path, item)
                    for item in os.listdir(folder_path)
                    if item.endswith(".png") or item.endswith(".jpg")
                ]
        return file_list


    def get_caption_value(self, file_path):
        file_path = file_path.replace('/image','/caption').replace(".png", ".json").replace(".jpg", ".json")
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "r") as file:
            data = json.load(file)
            if "microsoft-Florence-2-large-ft" in data:
                florence_data = data["microsoft-Florence-2-large-ft"]
                if "<MORE_DETAILED_CAPTION>" in florence_data:
                    return florence_data["<MORE_DETAILED_CAPTION>"]
        return ""

    def collect_meta(self, data_root):
        meta = []
        #for data_root in data_roots:
        for name in os.listdir(data_root):
            if name.endswith(".json"):
                with open(os.path.join(data_root, name), "r") as f:
                    for line in f.readlines():
                        try:
                            line_ = json.loads(line)
                            line_["data_root"] = data_root
                            meta.append(line_)
                        except Exception as e:
                            print(f"Error parsing: {line}")
        return meta



 