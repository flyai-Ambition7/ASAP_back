from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets, mixins
from rest_framework import status
from .models import ItemInfo, ResultImage
from .serializers import ItemInfoSerializer, ResultImageSerializer

# API KEYs
from config.settings import OPENAI_API_KEY, HUGGINGFACE_API_KEY, AZURE_ENDPOINT, AZURE_SUBSCRIPTION_KEY

# text 생성, 배경 이미지 생성 라이브러리 (GPT, Dalle-3)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI # gpt 용
from openai import OpenAI as openai # dalle 용

# text 이미지 생성 라이브러리
import cv2
import numpy as np
from PIL import Image # PIL 패키지에서 Image 클래스 불러오기
from rembg import remove # rembg 패키지에서 remove 클래스 불러오기
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# AZURE OCR 라이브러리
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# 텍스트 정확도 개선 라이브러리
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Stable Diffusion 라이브러리
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from diffusers.utils import load_image

import torch
from django.conf import settings
from django.core.files.base import ContentFile
import random
# from PIL import ImageFont
# from PIL import ImageDraw


# 이미지 형식의 파일을 받아서, 파일 경로 반환

# dall-e 를 이용해서 텍스트 이미지 생성
class TextImageGenerator():
    def __init__(self, text, max_attempts):
        self.text = text # 홍보에 사용될 문구
        self.max_attempts = max_attempts # 정확도 개선을 위해 시도할 최대 횟수

    # OpenAI DALL-E를 사용하여 이미지 생성하는 함수
    def draw_image_by_dalle(self):
        prompt = f"Text '{self.text}' on a white background, minimalism" # text 이미지를 생성하기 위한 프롬프트

        client = openai(api_key=OPENAI_API_KEY)
        # DALL-E 모델을 사용하여 주어진 프롬프트로 이미지를 생성합니다. 이미지 크기는 1792x1024, 품질은 hd, 생성할 이미지 수는 1입니다.q
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="hd",
            n=1,
        )

        # 생성된 이미지의 URL을 추출합니다.
        image_url = response.data[0].url
        print(image_url)

        return BytesIO(requests.get(image_url).content)

    # 텍스트를 정규화하는 함수 (구두점 제거 및 소문자로 변환)
    def normalize_text(self):
        # 텍스트에서 알파벳, 숫자, 공백만을 남기고 모두 소문자로 변환한 후 반환합니다.
        return ''.join(char.lower() for char in self.text if char.isalnum() or char.isspace()).strip()
    
    # 문장 전처리 함수
    @staticmethod
    def embed_text(sentence):
        sentence = sentence.lower() # 소문자 변환
        tokens = word_tokenize(sentence) # 토큰화
        lemmatizer = WordNetLemmatizer() # 표제어 추출
        tokens = [lemmatizer.lemmatize(word) for word in tokens] # 공백 제거
        tokens = [word for word in tokens if word.isalnum()]  # 재구성된 문장 반환
        sentence = ' '.join(tokens)

        return sentence
    
    # 두 문장 사이의 유사도 측정 함수
    def calculate_similarity(self, sentence1, sentence2):
        # 전처리 및 임베딩 계산
        embedding1, embedding2 = self.embed_text(sentence1), self.embed_text(sentence2)
        # 코사인 유사도 계산
        score = cosine_similarity([embedding1], [embedding2])[0][0]
        return score

    # Azure Computer Vision을 사용하여 OCR(광학 문자 인식) 수행하는 함수    
    def get_text_by_OCR(self, image_data):
        computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_SUBSCRIPTION_KEY))
        read_response = computervision_client.read_in_stream(image_data, raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        # ocr 진행
        # while True:
        #     read_result = computervision_client.get_read_result(operation_id)
        #     if read_result.status not in ['notStarted', 'running']:
        #         break
        #     time.sleep(1)

        # text = ""
        # if read_result.status == OperationStatusCodes.succeeded:
        #     for text_result in read_result.analyze_result.read_results:
        #         for text_line in text_result.lines:
        #             text += text_line.text + " "
        # return text
    
    # 생성된 텍스트 이미지의 cosine simillarity를 계산하는 함수
    def evalulate_image(self, img):
        generated_text = self.normalize_text(self.get_text_by_OCR(img))
        intended_text = self.normalize_text(self.text)

        return self.calculate_similarity(generated_text,intended_text)
    
    # 정확도가 가장 높은 텍스트 이미지를 선택하는 함수
    def draw_filtered_image_by_DALLE(self):
        DALLE_img_1st, DALLE_score_1st = 0, 0 # 정확도가 가장 높은 값 초기화
        DALLE_img_1st = self.draw_image_by_dalle()
        
        # OCR
        # for _ in range(self.max_attempts): # max_attempts 만큼 반복
        #     img = self.draw_image_by_dalle() # dalle를 이용해서 이미지 생성
        #     acc = self.evalulate_image(img) # evaluate_image 함수를 통해 코사인 유사도 계산

        #     if DALLE_score_1st < acc:
        #         DALLE_img_1st, DALLE_acc_1st = img, acc
        #         if acc > 0.99:
        #             break
        return (Image.open(DALLE_img_1st)) # 정확도가 가장 높은 텍스트 이미지 반환     

# SD를 이용해서 배경 이미지를 생성
class BgImageGenerator(): #img, product_name, description,theme, result_type
    def __init__(self, img, product_name, description, theme, result_type):
        self.img = img
        self.product_name = product_name
        self.description_name = description
        self.theme = theme
        self.result_type = result_type
    
    def draw_image_by_SD(self):
        # 이미지 파일을 1024x1024 크기로 조정
        img_input = self.img.resize((1024, 1024))
        img_rmbg = np.array(remove(img_input)) # PIL Image를 NumPy 배열로 변환하여 배경 제거 함수를 적용하고, 결과를 다시 배열로 변환
        mask = (img_rmbg[:, :, 3] == 0).astype(np.uint8) # 알파 채널을 사용하여 마스크 생성 (배경이 제거된 영역을 식별)
        mask_img = Image.fromarray(mask * 255, mode='L') # 생성된 마스크를 흑백 이미지로 변환하여 마스크 이미지 생성

        # Stability AI의 Stable Diffusion XL Refiner 모델 로드
        # FP16 정밀도 사용하여 메모리 사용량 감소 및 계산 속도 향상
        pipe = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                        torch_dtype=torch.float16,
                                                        variant="fp16",
                                                        token=HUGGINGFACE_API_KEY).to("cuda")        
        # 스케줄러 설정으로 인퍼런스 과정에서의 단계 조절
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.to("cuda")  # CUDA를 사용하여 GPU에서 모델 실행

        # 긍정적 및 부정적 프롬프트 설정
        prompt = f"create an advertising background to promote {self.product_name} in the {self.result_type} advertising style."
        pos = f"advertisement, {self.theme}, Bokeh"
        neg = f"(person : 1.5), (woman :1.5), face, worst, bad, (distorted:1.3), (deformed:1.3), (blurry:1.3), out of frame, duplicate, (text:1.3), (render:1.3)"

        # 각 이미지에 대해 다른 무작위 시드를 사용하여 결과 다양성 보장
        #generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(2)]

        # 모델을 호출하여 이미지 생성
        img_output = pipe(
            prompt=prompt + ',' + pos,  # 사용자 프롬프트와 긍정적 프롬프트 결합
            negative_prompt=neg,  # 부정적 프롬프트 설정
            image=img_input,  # 원본 이미지 입력
            mask_image=mask_img,  # 마스크 이미지 입력
            num_inference_steps=35,  # 인퍼런스 단계 수 설정
            strength=0.99,  # 변화 강도 설정 (1.0에 가까울수록 원본에서 큰 변화)
            num_images_per_prompt=2,  # 프롬프트 당 생성할 이미지 수
            #generator=generator  # 무작위 시드 제너레이터

        ).images[1]  # 생성된 이미지 중 두 번째 이미지 선택

        # 생성된 이미지를 PIL.Image 객체로 변환하여 반환
        return Image.fromarray(np.array(img_output))

class ImageSynthesizer():
    def __init__(self, text_image, bg_image, phone_num, location, theme):
        self.text_image = text_image
        self.bg_image = bg_image
        self.phone_num = phone_num
        self.location = location
        self.theme = theme

    def add_images(self):
        text_image, back_image = list(map(lambda img:np.array(img),[self.text_image, self.bg_image]))
        # back_image의 크기를 가져옴
        back_height, back_width, _ = back_image.shape
        # text_image를 back_image의 3분의 1 비율로 크기 조정
        text_image_resize = cv2.resize(text_image, (back_width, back_height // 3))
        # 텍스트 이미지 마스크 생성
        gray = cv2.cvtColor(text_image_resize, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 467, 37)

        #변수 단순화
        dst = back_image
        src = text_image_resize
        mask = binary_mask

        # 레이아웃 부분
        #배경, 텍스트 가로, 세로 길이 변수에 저장
        dst_h,dst_w = dst.shape[:2]
        src_h,src_w = src.shape[:2]

        #텍스트를 넣고자 하는 위치값 미리 러프하게 저장 # 위치별로 더 있으나 우선 테스트용 중앙 위쪽
        crop_centertop = dst[0:src_h, int(dst_w/2-src_w/2):int(dst_w/2+src_w/2)]
        ###########################################################################################################
        #배경에 텍스트 이미지 삽입
        cv2.copyTo(src, mask, crop_centertop)

        # Image 객체로 변환해서 저장하자
        image_data = Image.fromarray(dst)
        # filename = f'result_image{self.id}.png'
        # file_path = os.path.join(settings.MEDIA_ROOT, filename)
    
        # # 이미지 데이터를 파일로 저장
        # image_data.save(file_path)

        return image_data

# 사용자에게 입력받은 정보로 작업을 마친 후, response 성공 여부만 반환
class ItemInfoViewSet(viewsets.GenericViewSet, mixins.CreateModelMixin):
    queryset = ItemInfo.objects.all()
    serializer_class = ItemInfoSerializer

    
    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if response.status_code is not status.HTTP_201_CREATED:
            return Response(response)
        print(response)
        instance_id = response.data.get('id')
        instance = self.get_queryset().filter(id=instance_id).first()

        image_path = instance.image
        result_type = instance.result_type
        theme = instance.theme
        product_name = instance.product_name
        description = instance.description
        location = instance.location
        phone_num = instance.contact

        image = Image.open(image_path)

        # LangChain과 통합
        llm = OpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-instruct")
        prompt_template =  """
                            You are a digital marketing and content creation expert, and you need to create one-line advertisement copy to be printed on {result_type}.
                            Your part is to create advertising copy for product: {product_name}.
                            It must contain a summary of product name: {product_name} and description: {description}.
                            Your copy must grab the reader's attention.
                            Your sentence should reflect the characteristics of the product well.
                            You must translate {product_name} into English if was entered in Korean.
                            The copy should be based on the {theme} mood, and final copy must be written in English and no longer than 30 characters.
                            """
                            
        prompt = PromptTemplate(template=prompt_template, 
                                input_variables=['result_type', 'product_name', 'theme', 'description'])
        
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        generated_text = llm_chain.invoke(input={'result_type': result_type, 'theme': theme, 'product_name': product_name, 'description': description})
        
        if 'text' in generated_text:
            generated_text = generated_text['text'].strip() # gpt를 통한 홍보문구 생성
        else:
            generated_text = "Generated text if not found."
        
        max_attempts = 5

        # 정확도가 가장 높은 텍스트 이미지 생성 (Image 형식으로 전달)
        text_image_generator = TextImageGenerator(generated_text, max_attempts)
        text_image = text_image_generator.draw_filtered_image_by_DALLE()
        text_image.save('text_image.jpg')

        # 배경 이미지 생성 (Image 형식으로 전달)
        bg_image_generator = BgImageGenerator(image, product_name, description, theme, result_type)
        bg_image = bg_image_generator.draw_image_by_SD()
        bg_image.save('bg_image.jpg')

        synthesized_image_generator = ImageSynthesizer(text_image, bg_image, phone_num, location, theme)
        synthesized_image = synthesized_image_generator.add_images() # 텍스트, 배경 이미지 합성

        image_io = BytesIO()
        synthesized_image.save(image_io, format='JPEG')  # 이미지 포맷 지정
        image_io.seek(0)  # 스트림의 시작으로 포인터 이동

        # ContentFile을 사용하여 임시 파일 객체 생성
        temp_name = f'result_image{instance_id}.jpg'  # 임시 파일명, 필요에 따라 동적으로 생성 가능
        temp_file = ContentFile(image_io.read(), temp_name)

        # ResultImage 모델 인스턴스 생성 및 이미지 저장
        result_image_instance = ResultImage()
        result_image_instance.result_image_url.save(temp_name, temp_file, save=True)

        return Response({
                        'message': '작업이 성공적으로 완료되었습니다.'
                        }, status=status.HTTP_201_CREATED)


class ResultImageViewset(viewsets.ReadOnlyModelViewSet):
    queryset = ResultImage.objects.all()
    serializer_class = ResultImageSerializer