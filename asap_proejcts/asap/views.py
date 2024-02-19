from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from .models import CommonInfo, ItemInfo
from .serializers import CommonInfoSerializer, ItemInfoSerializer
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
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            item_info = serializer.save()
            
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
            
            
            item_info.summarized_copy = generated_text
            item_info.save()
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#class ResultDataSerializer(viewsets.ModelVieSet):