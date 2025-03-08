import streamlit as st
import requests

st.title("AI Image Generator (Stable Diffusion)")

# User input
prompt = st.text_input("Enter a text prompt:")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        response = requests.post("http://backend:8000/generate", json={"text": prompt})

        if response.status_code == 200:
            image_url = response.json()["image_url"]
            st.image(image_url, caption="Generated Image", use_column_width=True)
        else:
            st.error("Error generating image")
