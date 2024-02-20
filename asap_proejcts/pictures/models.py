from djongo import models

# Create your models here.

class ResultImage(models.Model):
    text_image = models.ImageField(blank=True, upload_to='text_image') # 생성된 텍스트 이미지
    background_image = models.ImageField(blank=True, upload_to='background_image') # 생성된 배경 이미지
    result_image_url = models.ImageField(blank=True, upload_to='result_image')  # 최종 출력 이미지
   