from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'password', 'name', 'email', 'phone']
    
    def create(self, validatated_data):
        password = validatated_data.pop('password', None)
        instance = self.Meta.model(validatated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance
        