from djongo import models
from django.contrib.auth.models import AbstractUser


# Create your models here.
class User(AbstractUser):
    name = models.CharField(max_length=20) # 사용자 이름
    email = models.EmailField(unique=True) # 사용자 이메일
    phone = models.CharField(max_length=12) # 사용자 전화번호
    
    def __str__(self):
        return self.username