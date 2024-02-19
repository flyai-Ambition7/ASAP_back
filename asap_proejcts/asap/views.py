from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from .models import CommonInfo, ItemInfo, ResultData
from .serializers import CommonInfoSerializer, ItemInfoSerializer, ResultDataSerializer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from config.settings import OPENAI_API_KEY

class CommonInfoViewSet(viewsets.ModelViewSet):
    queryset = CommonInfo.objects.all()
    serializer_class = CommonInfoSerializer

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

        # ResultData 모델에 저장
        # keyword 및 result_image_url 처리 필요 시 여기에 추가
        result_data = ResultData.objects.create(summarized_copy=generated_text)
        
        # 생성된 결과를 반환
        result_data_serializer = ResultDataSerializer(result_data)

        return Response({
                        'item_info': item_info_serializer.data,
                        'result_data': result_data_serializer.data
                        }, status=status.HTTP_201_CREATED)
class ResultDataViewSet(viewsets.ModelViewSet):
    queryset = ResultData.objects.all()
    serializer_class = ResultDataSerializer