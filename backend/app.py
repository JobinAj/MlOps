from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import uuid
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler=scheduler,
    torch_dtype=torch.float32
)
pipe.to(device)

# Ensure "images" directory exists
os.makedirs("images", exist_ok=True)

# Define a request model
class PromptRequest(BaseModel):
    text: str

# Define the image generation endpoint
@app.post("/generate")
async def generate_image(request: PromptRequest):
    try:
        # Generate an image from the text prompt
        image = pipe(request.text).images[0]
        
        # Save image with a unique filename
        image_name = f"images/{uuid.uuid4()}.png"
        image.save(image_name)
        
        return {"image_url": f"/{image_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve images as static files
app.mount("/images", StaticFiles(directory="images"), name="images")

# Health Check
@app.get("/")
async def root():
    return {"message": "Backend is running!"}
