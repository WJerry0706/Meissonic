import os
import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

device = 'cuda'

# Load models as before
model_path = "MeissonFlow/Meissonic"
model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae")
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
pipe = pipe.to(device)

# Parameters
steps = 64
CFG = 9
resolution = 1024
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
batch_size = 2

# Read prompts from file
prompt_file = "coco_cleaned.txt"
with open(prompt_file, "r", encoding="utf-8") as f:
    all_prompts = [line.strip() for line in f if line.strip()]

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process prompts in batches
for batch_start in range(0, len(all_prompts), batch_size):
    batch_prompts = all_prompts[batch_start: batch_start + batch_size]
    batch_size_actual = len(batch_prompts)

    images = pipe(
        prompt=batch_prompts,
        negative_prompt=[negative_prompt] * batch_size_actual,
        height=resolution,
        width=resolution,
        guidance_scale=CFG,
        num_inference_steps=steps
    ).images

    for i, img in enumerate(images):
        prompt = batch_prompts[i]
        sanitized_prompt = prompt.replace(" ", "_").replace("/", "_")
        file_name = f"{batch_start + i + 1}_{sanitized_prompt}_{resolution}_{steps}_{CFG}.png"
        file_path = os.path.join(output_dir, file_name)
        img.save(file_path)
        print(f"Saved image {batch_start + i + 1}/{len(all_prompts)}: {file_path}")
