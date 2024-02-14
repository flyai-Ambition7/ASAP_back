from djongo import models

# Create your models here

class CommonInfo(models.Model): # 공통 입력사항
    storename = models.CharField(max_length=50, blank=True) # 가게명
    purpose = models.CharField(max_length=10, blank=True) # 사용 목적
    type = models.CharField(max_length=10, blank=True) # 결과물 형태
    theme = models.CharField(max_length=10, blank=True) #결과물 테마
    
# 상품을 홍보하는 경우    
class ItemInfo(models.Model):
    productname = models.CharField(max_length=50, blank=True) # 상품명
    price =  models.CharField(max_length=10, blank=True) # 상품 가격
    description = models.CharField(max_length=100, blank=True) # 상품 설명
    storehours = models.CharField(max_length=30, blank=False, null=True) # 가게 영업시간
    location = models.CharField(max_length=100, blank= False, null=True) # 가게 위치
    contact = models.CharField(max_length=20, blank= False, null=True) # 가게 전화번호

# class ShopInfo(models.Model): # 가게를 홍보하는 경우
# class EventInfo(models.Model) # 이벤트를 홍보하는 경우