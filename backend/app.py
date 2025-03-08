from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os
import socket
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI
app = FastAPI()

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)

# Ensure 'images' directory exists
if not os.path.exists("images"):
    os.makedirs("images")

# Get server IP for correct image URL
host_ip = socket.gethostbyname(socket.gethostname())

# Define input model
class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate_image(prompt: Prompt):
    image = pipe(prompt.text).images[0]  # Generate image
    filename = f"generated_{uuid.uuid4().hex}.png"
    image.save(f"images/{filename}")  # Save image

    return {"image_url": f"http://{host_ip}:8000/images/{filename}"}

# Serve generated images
app.mount("/images", StaticFiles(directory="images"), name="images")
