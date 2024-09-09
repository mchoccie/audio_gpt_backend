from rest_framework import serializers
from .models import FileSave
from .models import CustomProfile
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileSave
        fields = ('audio_name', 'audio_file')
class CustomProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomProfile
        fields = ('username', 'password', 'email', 'apikey', 'langchainkey')


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')
        # print(email, password)
        if email and password:
            user = authenticate(username=email, password=password)
            if user is None:
                raise serializers.ValidationError("Invalid email or password.")
        else:
            raise serializers.ValidationError("Must include 'email' and 'password'.")
        
        data['user'] = user
        return data
