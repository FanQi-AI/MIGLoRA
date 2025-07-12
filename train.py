# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pdb
import datasets
import numpy as np
import torch.nn as nn
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import datetime
from omegaconf import OmegaConf
from mask_encoder import ControlNetConditioningEmbedding
import importlib
from collections import defaultdict
from accelerate import DistributedDataParallelKwargs
import safetensors
from safetensors.torch import load_file

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_training_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config_dict = json.load(f)

    return Config(**config_dict)

def serialize_lora_config(lora_config):
    return {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "init_lora_weights": lora_config.init_lora_weights,
        "target_modules": lora_config.target_modules,
    }


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, transform=None, device='cpu', max_conditions=5):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if isinstance(self.data, dict):
            self.data = list(self.data.values())

        self.transform = transform
        self.device = device
        self.tokenizer = tokenizer
        self.max_conditions = max_conditions  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        image_width = item['image_width']
        image_height = item['image_height']
        caption = item['caption']
        masks = item['mask']

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        base_encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        base_class_text_ids = base_encoding.input_ids.squeeze(0) 

        obj_class = []
        obj_class_text_ids = []
        cond_images = []
        num_selected = 0

        for mask in masks:
            label = mask.get('label', 'background')
            if label.lower() == 'background':
                continue  # 跳过背景

            value = mask.get('value', 0)
            obj_class.append(value)

            sub_caption = mask.get('sub_caption', '')
            if sub_caption:
                encoding = self.tokenizer(
                    sub_caption,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                input_ids = encoding.input_ids.squeeze(0)  
            else:
                encoding = self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                input_ids = encoding.input_ids.squeeze(0)
            obj_class_text_ids.append(input_ids)

            box_rel = mask['box']['relative']  # [x1, y1, x2, y2]
            x1_rel, y1_rel, x2_rel, y2_rel = box_rel
            x1 = int(x1_rel * image_width)
            y1 = int(y1_rel * image_height)
            x2 = int(x2_rel * image_width)
            y2 = int(y2_rel * image_height)

            cond_image = np.zeros((image_height, image_width), dtype=np.uint8)  
            cond_image[y1:y2, x1:x2] = 1 
            cond_image = Image.fromarray(cond_image)
            cond_image = cond_image.resize((8, 8)) 
            cond_image = transforms.ToTensor()(cond_image) 
            cond_image = cond_image.repeat(3, 1, 1)
            cond_images.append(cond_image)

            num_selected += 1
            if num_selected >= self.max_conditions:
                break 

        if not obj_class_text_ids:
            dummy_input_ids = self.tokenizer(
                "",
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).input_ids.squeeze(0)
            obj_class_text_ids.append(dummy_input_ids)

            dummy_cond_image = torch.zeros((3, 8, 8), dtype=torch.float32)  
            cond_images.append(dummy_cond_image)
            obj_class.append(0)  
            num_selected = 1

        while len(obj_class) < self.max_conditions:
            obj_class.append(0)  
            dummy_input_ids = self.tokenizer(
                "",
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).input_ids.squeeze(0)
            obj_class_text_ids.append(dummy_input_ids)
            dummy_cond_image = torch.zeros((3, 8, 8), dtype=torch.float32) 
            cond_images.append(dummy_cond_image)

        obj_class = obj_class[:self.max_conditions]
        obj_class_text_ids = obj_class_text_ids[:self.max_conditions]
        cond_images = cond_images[:self.max_conditions]

        obj_class = torch.tensor(obj_class, dtype=torch.int64).to(self.device)
        obj_class_text_ids = torch.stack(obj_class_text_ids).to(self.device)  
        cond_images = torch.stack(cond_images).to(self.device)  

        cond_dict = {
            "obj_class": obj_class,  
            "obj_class_text_ids": obj_class_text_ids,  
            "base_class_text_ids": base_class_text_ids.to(self.device), 
            "cond_image": cond_images,  
            "num_selected": torch.tensor(num_selected, dtype=torch.int64).to(self.device)  
        }

        return image, cond_dict


from prefetch_generator import BackgroundGenerator


class DataLoaderUpd(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='lora_re_mask_v33.json',
                        help='Path to the configuration JSON file.')  
    args = parser.parse_args()

    cfg_path = args.cfg

    cfg = load_training_config(cfg_path)
    logging_dir = Path(cfg.log_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    print(accelerator.project_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    mask_encoder = ControlNetConditioningEmbedding()

    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=r"down_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)|"
                       r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)|"
                       r"up_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)"

    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    mask_encoder.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)


    if cfg.mixed_precision == "fp16":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        for param in mask_encoder.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        list(lora_layers) + list(mask_encoder.parameters()),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = JsonDataset("train_dataset/train.json", transform=transform, tokenizer=tokenizer, device='cpu')

    train_dataloader = DataLoaderUpd(
        train_dataset,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True
    )


    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer
    )

    unet, optimizer, train_dataloader, lr_scheduler, mask_encoder = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, mask_encoder
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = cfg.train_batch_size * \
                       accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )



    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        mask_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet, mask_encoder]):


                batch_images, batch_cond = batch  # batch_images b, 3, 512, 512

                latents = vae.encode(batch_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                obj_class = batch_cond['obj_class']

                noise = torch.randn_like(latents)
                if cfg.noise_offset:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                bsize = len(batch_cond['obj_class_text_ids'])
                cond_mask = []
                encoder_hidden_states_batch = []

                for cid in range(bsize):
                    bsid = min(batch_cond["num_selected"][cid], 5)
                    encoder_hidden_states = text_encoder(batch_cond['obj_class_text_ids'][cid])[0][:bsid]

                    controlnet_image = batch_cond["cond_image"][cid].to(dtype=weight_dtype)[:bsid]

                    mask_cond = mask_encoder(controlnet_image) 

                    cond_mask.append(mask_cond)
                    encoder_hidden_states_batch.append(encoder_hidden_states)

                if isinstance(batch_cond["base_class_text_ids"], list):
                    base_class_text_ids_tensor = torch.stack(batch_cond["base_class_text_ids"]).squeeze(1)
                else:
                    base_class_text_ids_tensor = batch_cond["base_class_text_ids"].squeeze(1)

                unet_encoder_hidden_states = text_encoder(base_class_text_ids_tensor)[0]

                model_pred = unet(noisy_latents, timesteps, unet_encoder_hidden_states, cond_mask=cond_mask,
                                  encoder_hidden_states_batch=encoder_hidden_states_batch).sample


                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + list(mask_encoder.parameters())
                    accelerator.clip_grad_norm_(
                        params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        adapter_config = serialize_lora_config(unet_lora_config)
                        adapter_config_path = os.path.join(save_path, "adapter_config.json")
                        with open(adapter_config_path, "w") as f:
                            json.dump(adapter_config, f, indent=4)

                        unwrap_mask_encoder = accelerator.unwrap_model(mask_encoder)  # 解除封装
                        unwrap_mask_encoder_state_dict = unwrap_mask_encoder.state_dict()
                        mask_encoder_save_path = os.path.join(save_path, "mask_encoder.pth")
                        torch.save(unwrap_mask_encoder_state_dict, mask_encoder_save_path)

                        logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=cfg.ckpt_name + '.safetensor'
        )


    accelerator.end_training()


if __name__ == "__main__":
    main()
