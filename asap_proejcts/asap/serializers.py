from rest_framework import serializers
from .models import CommonInfo, ItemInfo, ResultData

class CommonInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommonInfo
        fields = '__all__'
        
class ItemInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ItemInfo
        fields = '__all__'

class ResultDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultData
        fields = '__all__'