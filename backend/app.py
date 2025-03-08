from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uuid

# Initialize FastAPI
app = FastAPI()

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use GPU if available

# Define input model
class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate_image(prompt: Prompt):
    image = pipe(prompt.text).images[0]  # Generate image
    filename = f"generated_{uuid.uuid4().hex}.png"
    image.save(f"images/{filename}")  # Save image

    return {"image_url": f"http://localhost:8000/images/{filename}"}

# Serve generated images
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="images"), name="images")
