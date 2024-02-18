from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email', 'phone']

    def create(self, validated_data):
        # Convert 'id' to int if it exists in validated_data and is a valid integer
        id_value = validated_data.get('id')
        if id_value is not None:
            try:
                validated_data['id'] = int(id_value)
            except ValueError:
                # Handle the case where 'id' cannot be converted to an integer
                raise serializers.ValidationError({'id': 'Invalid value for id. Must be an integer.'})

        password = validated_data.pop('password', None)
        instance = self.Meta.model(**validated_data)

        if password is not None:
            instance.set_password(password)

        instance.save()
        return instance
