from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model

class EmailBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            # Retrieve the user by email instead of username
            user = UserModel.objects.get(email=username)
            print("-------------------")
            print(user)
            print("-------------------")
        except UserModel.DoesNotExist:
            return None

        # Check if the password is correct
        if user.check_password(password):
            return user
        return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None