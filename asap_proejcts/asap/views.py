from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import generics, status
from .models import CommonInfo, ItemInfo
from .serializers import CommonInfoSerializer, ItemInfoSerializer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI


class CommonInfoViewSet(viewsets.ModelViewSet):
    queryset = CommonInfo.objects.all()
    serializer_class = CommonInfoSerializer

class ItemInfoViewSet(viewsets.ModelViewSet):
    queryset = ItemInfo.objects.all()
    serializer_class = ItemInfoSerializer
