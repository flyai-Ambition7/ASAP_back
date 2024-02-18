from rest_framework import serializers
from .models import CommonInfo, ItemInfo

class CommonInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommonInfo
        fields = '__all__'
        
class ItemInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ItemInfo
        fields = '__all__'