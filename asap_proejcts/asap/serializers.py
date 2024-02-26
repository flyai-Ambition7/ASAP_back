from rest_framework import serializers
from .models import ItemInfo, GeneratedData, ResultImage
from django.core.files.base import ContentFile
import base64
import six
import uuid
import base64
import imghdr

"""class CommonInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommonInfo
        fields = '__all__'"""

class Base64ImageField(serializers.ImageField):   #디코딩 입력
    def to_internal_value(self, data):

        if isinstance(data, str):
            # Check if the base64 string is in the "data:" format
            if 'data:' in data and ';base64,' in data:
                # Break out the header from the base64 content
                header, data = data.split(';base64,')           
            try:
                decoded_file = base64.b64decode(data)
            except TypeError:
                self.fail('invalid_image')
            
            file_name = str(uuid.uuid4())[:12] 
            file_extension = self.get_file_extension(file_name, decoded_file)
            complete_file_name = f"{file_name}.{file_extension}"
            data = ContentFile(decoded_file, name=complete_file_name)
        return super(Base64ImageField, self).to_internal_value(data)

    def get_file_extension(self, file_name, decoded_file):
        extension = imghdr.what(file_name, decoded_file)
        extension = "jpg" if extension == "jpeg" else extension
        
        return extension
    
    

class ItemInfoSerializer(serializers.ModelSerializer):
    image = Base64ImageField(
        max_length=None, use_url=True, required=False
    )
    class Meta:
        model = ItemInfo
        fields = '__all__'

class GeneratedDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedData
        fields = '__all__'

class ResultImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultImage
        fields = ['result_image_url']

#--------------------
# # 인코딩 혹시나 모르기에
# import base64
# class Base64ImageEncoder(serializers.BaseSerializer): # 
#     def to_representation(self, obj):
#         if obj and hasattr(obj, 'read'):
#             encoded_data = base64.b64encode(obj.read()).decode("utf-8")
#             return encoded_data
#         return None