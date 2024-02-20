from rest_framework import viewsets
from rest_framework import status
from config.settings import OPENAI_API_KEY, HUGGINGFACE_API_KEY
from asap.models import ItemInfo, GeneratedData
from .models import ResultImage
from .serializers import ResultImageSerializer
import cv2
import numpy as np
from PIL import Image # PIL 패키지에서 Image 클래스 불러오기
from rembg import remove # rembg 패키지에서 remove 클래스 불러오기
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch

# Create your views here.
class ResultImageViewSet(viewsets.ModelViewSet):
    queryset = ItemInfo.objects.all()
    serializer_class = ResultImageSerializer