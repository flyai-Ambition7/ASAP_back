from djongo import models

# Create your models here

# 공통 입력사항
class CommonInfo(models.Model):
    image = models.ImageField(blank=False, upload_to='input_image') # 이미지
    store_name = models.CharField(max_length=50, blank=False) # 가게명
    purpose = models.CharField(max_length=50, blank=False) # 사용 목적
    result_type = models.CharField(max_length=10, blank=False) # 결과물 형태
    theme = models.CharField(max_length=50, blank=False) #결과물 테마

# 상품을 홍보하는 경우   
class ItemInfo(CommonInfo, models.Model):
    product_name = models.CharField(max_length=50, blank=False) # 상품명
    price =  models.CharField(max_length=20, blank=True) # 상품 가격
    description = models.TextField() # 상품 설명
    business_hours = models.CharField(max_length=30, blank=True) # 가게 영업시간
    location = models.CharField(max_length=100, blank=True) # 가게 위치
    contact = models.CharField(max_length=20, blank=True) # 가게 전화번호

# summarized_copy, text_prompt, background_prompt, result_image_url
class ResultData(CommonInfo, models.Model):
    summarized_copy = models.CharField(max_length=200, blank=True) # gpt를 이용해 생성된 광고문구
    text_prompt = models.CharField(max_length=200, blank=True) # 텍스트 이미지를 생성하기 위한 프롬프트
    background_prompt = models.CharField(max_length=100, blank=True) # 배경 이미지를 생성하기 위한 프롬프트 
    result_image_url = models.ImageField(blank=True, upload_to='result_image') # 최종 출력 이미지
    # keyword = models.CharField(max_length=100, blank=True) # gpt를 이용해 생성된 키워드
    
# class ShopInfo(models.Model): # 가게를 홍보하는 경우
# class EventInfo(models.Model) # 이벤트를 홍보하는 경우