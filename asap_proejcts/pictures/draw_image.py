from dotenv import load_dotenv
import numpy as np
import requests
import os
from io import BytesIO
import torch
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from openai import OpenAI
from PIL import Image # PIL 패키지에서 Image 클래스 불러오기
from eval_image import evalulate_image
import matplotlib.pyplot as plt
from rembg import remove # rembg 패키지에서 remove 클래스 불러오기

load_dotenv(verbose=True)
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
HF_TOKEN=os.getenv("HF_TOKEN")
def draw_image_by_DALLE(prompt):
    client=OpenAI(api_key=OPENAI_API_KEY)
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1,
    )
    image_url=response.data[0].url
    return BytesIO(requests.get(image_url).content)

def draw_image_by_SD(img,prompt):

    # 이미지 파일 불러오기
    img_input = img.resize((1024,1024))

    # PIL Image를 NumPy 배열로 변환
    img_rmbg = np.array(remove(img_input))
    mask = (img_rmbg[:, :, 3] == 0).astype(np.uint8)
    mask_img = Image.fromarray(mask * 255, mode='L')
    pipe = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                     torch_dtype=torch.float16,
                                                     variant="fp16",
                                                     token=HF_TOKEN).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pos = "masterpiece, best quality, background, decoration"
    neg = "worst, bad, (distorted:1.3), (deformed:1.3), (blurry:1.3), out of frame, duplicate, (text:1.3), (render:1.3)"
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(2)]
    img_output = pipe(
        prompt=prompt+','+pos,
        negative_prompt=neg,
        image=img_input,
        mask_image=mask_img,
        num_inference_steps=35,
        strength=0.99,  # make sure to use `strength` below 1.0
        num_images_per_prompt=2,
        generator=generator
    ).images[1]
    return Image.fromarray(np.array(img_output))

def draw_filtered_image_by_DALLE(prompt):
    DALLE_img_1st,DALLE_score_1st = 0,0
    for _ in range(2):
        img = draw_image_by_DALLE(prompt)
        acc = evalulate_image(prompt,img)
        if DALLE_score_1st<acc:
            DALLE_img_1st, DALLE_acc_1st = img, acc
            if acc>0.99:
                break
    return (Image.open(DALLE_img_1st), DALLE_acc_1st)