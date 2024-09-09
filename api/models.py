from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError

class FileSave(models.Model):
    audio_name = models.CharField(max_length=50)
    audio_file = models.FileField(upload_to="audio_files/")

class CustomProfile(AbstractUser):
    apikey = models.TextField()
    langchainkey = models.TextField()
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150, blank=True, null=True, unique=False)

    def __str__(self):
        return self.username