from rest_framework import serializers
from .models import ResultImage

class ResultImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultImage
        fields = '__all__'