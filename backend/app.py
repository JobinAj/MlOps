from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DDIMScheduler  # Import a different scheduler
import torch
import uuid
import os
import socket
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI
app = FastAPI()

# Load Stable Diffusion model with a more stable scheduler
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a properly configured scheduler
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Initialize pipeline with the specified scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler=scheduler,
    torch_dtype=torch.float32
)
pipe.to(device)

# The rest of your code remains the same...
