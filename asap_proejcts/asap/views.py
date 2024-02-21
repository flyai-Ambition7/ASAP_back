from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from .models import ItemInfo, GeneratedData, ResultImage
from .serializers import ItemInfoSerializer, GeneratedDataSerializer, ResultImageSerializer

# API KEYs
from config.settings import OPENAI_API_KEY, HUGGINGFACE_API_KEY, AZURE_ENDPOINT, AZURE_SUBSCRIPTION_KEY

# text 생성, 배경 이미지 생성 라이브러리 (GPT, Dalle-3)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# text 이미지 생성 라이브러리
import cv2
import numpy as np
from PIL import Image # PIL 패키지에서 Image 클래스 불러오기
from rembg import remove # rembg 패키지에서 remove 클래스 불러오기
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import re
import time

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

import getpass
import torch

class TextImageGenerator():
    def __init__(self, prompt, text, max_attempts):
        self.prompt = prompt # dalle의 이미지를 만들어 내기위한 프롬프트
        self.text = text # 홍보에 사용될 문구
        self.max_attempts = max_attempts # 정확도 개선을 위해 시도할 최대 횟수

    # OpenAI DALL-E를 사용하여 이미지 생성하는 함수
    def draw_image_by_dalle(self):
        client = OpenAI(api_key=OPENAI_API_KEY)
        # DALL-E 모델을 사용하여 주어진 프롬프트로 이미지를 생성합니다. 이미지 크기는 1792x1024, 품질은 hd, 생성할 이미지 수는 1입니다.
        response = client.images.generate(
            model="dall-e-3",
            prompt=self.prompt,
            size="1792x1024",
            quality="hd",
            n=1,
        )

        # 생성된 이미지의 URL을 추출합니다.
        image_url = response.data[0].url

        return BytesIO(requests.get(image_url).content)

    # 텍스트를 정규화하는 함수 (구두점 제거 및 소문자로 변환)
    def normalize_text(self):
        # 텍스트에서 알파벳, 숫자, 공백만을 남기고 모두 소문자로 변환한 후 반환합니다.
        return ''.join(char.lower() for char in self.text if char.isalnum() or char.isspace()).strip()
    
    # 문장 전처리 함수
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
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        text = ""

        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for text_line in text_result.lines:
                    text += text_line.text + " "
        return text
    
    # 생성된 텍스트 이미지의 cosine simillarity를 계산하는 함수
    def evalulate_image(self, img):
        generated_text = self.normalize_text(self.get_text_by_OCR(img))
        intended_text = self.normalize_text(self.text)

        return self.calculate_similarity(generated_text,intended_text)
    
    # 정확도가 가장 높은 텍스트 이미지를 선택하는 함수
    def draw_filtered_image_by_DALLE(self, prompt):
        DALLE_img_1st, DALLE_score_1st = 0, 0 # 정확도가 가장 높은 값 초기화
        
        for _ in range(self.max_attempts): # max_attempts 만큼 반복
            img = self.draw_image_by_dalle(prompt) # dalle를 이용해서 이미지 생성
            acc = self.evalulate_image(prompt,img) # evaluate_image 함수를 통해 코사인 유사도 계산

            if DALLE_score_1st < acc:
                DALLE_img_1st, DALLE_acc_1st = img, acc
                if acc > 0.99:
                    break

        return (Image.open(DALLE_img_1st), DALLE_acc_1st)         

class ItemInfoViewSet(viewsets.ModelViewSet):
    queryset = ItemInfo.objects.all()
    serializer_class = ItemInfoSerializer
    
    def create(self, request, *args, **kwargs):
        item_info_serializer = self.get_serializer(data=request.data)
        if item_info_serializer.is_valid():
            item_info = item_info_serializer.save()

        # LangChain과 통합
        result_type = request.data.get('result_type')
        theme = request.data.get('theme')
        product_name = request.data.get('product_name')
        description = request.data.get('description')
        
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
            generated_text = generated_text['text'].strip()
        else:
            generated_text = "Generated text if not found."

        text_image_prompt = f'Text "{generated_text}" on a white background, minimalism'
        max_attempts = 5

        text_image = TextImageGenerator(text_image_prompt, generated_text, max_attempts) # 텍스트 정확도가 가장 높은 텍스트 이미지 생성

        generated_data = GeneratedData.objects.create(summarized_copy=generated_text)
        result_image = ResultImage.objects.create(text_image=text_image)
        #result_image = ResultImage.objects.create(result_image_url='something')
        
        generated_data_serializer = GeneratedDataSerializer(generated_data)
        result_image_serializer = ResultImageSerializer(result_image)

        # 생성된 결과를 반환
        return Response({
                        'item_info': item_info_serializer.data,
                        'generated_data': generated_data_serializer.data,
                        'result_image' : result_image_serializer.data
                        }, status=status.HTTP_201_CREATED)
    
class GeneratedDataViewSet(viewsets.ModelViewSet):
    queryset = GeneratedData.objects.all()
    serializer_class = GeneratedDataSerializer

class ResultImageViewSet(viewsets.ModelViewSet):
    queryset = ResultImage.objects.all()
    serializer_class = ResultImageSerializer