from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from .models import ItemInfo, ResultData
from .serializers import ItemInfoSerializer, ResultDataSerializer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from config.settings import OPENAI_API_KEY

'''class CommonInfoViewSet(viewsets.ModelViewSet):
    queryset = CommonInfo.objects.all()
    serializer_class = CommonInfoSerializer'''

class ItemInfoViewSet(viewsets.ModelViewSet): # GPT지안이 넣은 상태, 여기서 추가로 폰트, 배경을 넣으면 된다.
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

#############-----------\
import time
from draw_image import draw_filtered_image_by_DALLE, draw_image_by_SD
from edit_image import add_images
# from eval_image import evalulate_image
#생성시간 나중에 테이블 추가
        # 데이터베이스에서 필요한 정보들을 읽어오는 함수 호출 -> 디비 데이터변수명 수정
        # 입력받은 이미지, 텍스트 프롬프트(폰트), 이미지 프롬프트(배경)
        # image, summarized_copy, text_prompt, background_prompt, result_image_url
class ResultDataViewSet(viewsets.ModelViewSet):
    queryset = ResultData.objects.all()
    serializer_class = ResultDataSerializer

    def draw(self, request, *args, **kwargs):
        start = time.time()
        result_data_instance = ResultData.objects.first()
        # 가져온 데이터를 변수에 할당
        image = result_data_instance.image
        summarized_copy = result_data_instance.summarized_copy
        text_prompt = result_data_instance.text_prompt
        background_prompt = result_data_instance.background_prompt

        # DALLE 모델을 이용하여 이미지를 그리는 함수 호출
        DALLE_img, DALLE_acc = draw_filtered_image_by_DALLE(text_prompt)
        # SD 모델을 이용하여 이미지를 그리는 함수 호출
        SD_img = draw_image_by_SD(image, background_prompt)
        # 두 이미지를 결합하는 함수 호출
        img_output = add_images(DALLE_img, SD_img)

        serializer = ResultDataSerializer(data={
            'result_image_url': img_output
        })

        if serializer.is_valid():
            serializer.save()
            end = time.time()
            return Response({
                "DALLE_accuracy": f"{DALLE_acc:.2f}",
                "time_consumption": f"{end - start:.2f}"
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# def draw():
#     start = time.time()
#     # 데이터베이스에서 필요한 정보들을 읽어오는 함수 호출
#     img_input, img_file_name, text_prompt, image_prompt = read_infos_from_db(img_chunk_tbl, img_meta_tbl, text_tbl, False)
#     user_id = img_file_name.split('_')[0]
#     # DALLE 모델을 이용하여 이미지를 그리는 함수 호출
#     DALLE_img, DALLE_acc = draw_filtered_image_by_DALLE(text_prompt)
#     # SD 모델을 이용하여 이미지를 그리는 함수 호출
#     SD_img = draw_image_by_SD(img_input, image_prompt)
#     # 두 이미지를 결합하는 함수 호출
#     img_output = add_images(DALLE_img, SD_img)
#     # 결과 이미지를 데이터베이스에 업데이트하는 함수 호출
#     update_image_to_db(img_output, user_id, upload_time=int(start), isinput=False)
#     end = time.time()
#     return {
#         "DALLE_accuracy": f"{DALLE_acc:.2f}",
#         "time_consumption": f"{end - start:.2f}"
#     }
#         image = request.data.get('image')
#         summarized_copy = request.data.get('summarized_copy')
#         text_prompt = request.data.get('text_prompt')
#         background_prompt = request.data.get('background_prompt')