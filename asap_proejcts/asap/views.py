from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from .models import ItemInfo, GeneratedData, ResultImage
from .serializers import ItemInfoSerializer, GeneratedDataSerializer, ResultImageSerializer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from config.settings import OPENAI_API_KEY, HUGGINGFACE_API_KEY
from rest_framework import viewsets
import cv2
import numpy as np
from PIL import Image # PIL 패키지에서 Image 클래스 불러오기
from rembg import remove # rembg 패키지에서 remove 클래스 불러오기
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch

'''class CommonInfoViewSet(viewsets.ModelViewSet):
    queryset = CommonInfo.objects.all()
    serializer_class = CommonInfoSerializer'''

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

        generated_data = GeneratedData.objects.create(summarized_copy=generated_text)
        
        generated_data_serializer = GeneratedDataSerializer(generated_data)

        # 생성된 결과를 반환
        return Response({
                        'item_info': item_info_serializer.data,
                        'generated': generated_data_serializer.data
                        }, status=status.HTTP_201_CREATED)
    
class GeneratedDataViewSet(viewsets.ModelViewSet):
    queryset = GeneratedData.objects.all()
    serializer_class = GeneratedDataSerializer

class ResultImageViewSet(viewsets.ModelViewSet):
    queryset = ResultImage.objects.all()
    serializer_class = ResultImageSerializer