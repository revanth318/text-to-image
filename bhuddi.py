import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import streamlit as st,torch
from diffusers import StableDiffusionPipeline,LCMScheduler
@st.cache_resource
def load_pipeline():
    device="cuda" if torch.cuda.is_available() else "cpu"
    model_id="runwayml/stable-diffusion-v1-5"
    #bnb_config=BitsAndBytesConfig(load_in8bit=True)
    dtype=torch.float16 if device=="cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   torch_dtype=dtype)
                                                  # device_map="auto",
                                                  # quantization_config=bnb_config,)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checks=None
    pipe.fuse_lora()
    return pipe.to(device)
pipeline = load_pipeline()
with st.sidebar:
    num_steps=st.slider("Number of steps",2,4,8)
    guidance_scale=st.slider("GUIDANCE SCALE",1,2,3)

prompt=st.text_input("Prompt","masterpiece,ultra detailed,cinematic lighting 8k")
negative_prompt="blurry,low quality"
if st.button("Generate"):
    img=pipeline(
        prompt,
        negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        ).images[0]
    st.image(img)

