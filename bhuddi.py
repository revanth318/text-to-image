import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import streamlit as st
import torch
from diffusers import StableDiffusionPipeline,LCMScheduler
@st.cache_resource
def load_pipeline():
    pipe=StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
    low_cpu_mem_usage=True,
    use_safetensors=True)
    pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    #pipe.fuse_lora()
    #pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.unet.to(memory_format=torch.channels_last)
    
    return pipe
pipeline = load_pipeline()
with st.sidebar:
    num_steps=st.slider("Number of steps",4)
    guidance_scale=st.slider("GUIDANCE SCALE",1)

prompt=st.text_input("Prompt","masterpiece,ultra detailed,cinematic lighting 8k")
negative_prompt="blurry,low quality"
if st.button("Generate"):
    img=pipeline(
        prompt,
        negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=256,
        width=256
        ).images[0]
    st.image(img)

