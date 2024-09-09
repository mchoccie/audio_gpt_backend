from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from cryptography.fernet import Fernet
import uuid
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from .serializers import  FileSerializer
from .serializers import CustomProfileSerializer, LoginSerializer
from django.contrib.auth import login
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.chat_models import ChatOpenAI
import openai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import assemblyai as aai
from dotenv import load_dotenv
import os
from django.contrib.auth import logout
from decouple import config
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import assemblyai as aai
from langchain_postgres.vectorstores import PGVector
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document
from openai import OpenAI
from io import BytesIO


fernet_key = os.getenv('FERNET_KEY')
fernet = Fernet(fernet_key)

# langchain_api_key = os.getenv('LANGCHAIN_API_KEY', 'default_langchain_key')

model_name = "gpt-3.5-turbo"
# os.environ['LANGCHAIN_API_KEY'] = langchain_api_key

load_dotenv()
class AudioFileView(APIView):
    authentication_classes = [TokenAuthentication]
    serializer_class = FileSerializer
    
    def post(self, request, format=None):
        
        def get_postgresql_connection_string():
            db_settings = settings.DATABASES['default']
            return f"postgresql+psycopg2://{db_settings['USER']}:{db_settings['PASSWORD']}@{db_settings['HOST']}:{db_settings['PORT']}/{db_settings['NAME']}"
        data = request.data

        user = request.user
        openai_api_key = user.apikey
        encrypted_api_key_bytes = openai_api_key.encode('utf-8')
        decrypted_data = fernet.decrypt(encrypted_api_key_bytes)
        decrypted_text = decrypted_data.decode('utf-8')
        audiofile = request.FILES['audio_file']
        serializer = self.serializer_class(data=request.data)
        
        data.update(request.FILES)
        print(serializer.is_valid())

        if serializer.is_valid():
            audio_name = data.get('audio_name')
            audio_file = data.get('audio_file')
 
            client = OpenAI(api_key=decrypted_text)
            audio_bytes = audio_file.read()
            audio_stream = BytesIO(audio_bytes)
            audio_stream.name = "audio_name.mp3"
            # Passing the file directly to the OpenAI API
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(audio_stream)  # Directly pass the InMemoryUploadedFile
            )

            class StringDocumentLoader(BaseLoader):
                def __init__(self, text: str):
                    self.text = text

                def load(self):
                    return [Document(page_content=self.text)]

            loader = StringDocumentLoader(transcript.text)
            documents = loader.load()

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
        
            embeddings = OpenAIEmbeddings(openai_api_key=decrypted_text)

            connection = get_postgresql_connection_string()
            COLLECTION_NAME = user.email
            db = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=connection, use_jsonb=True)
            db.add_documents(splits)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            

class MessageView(APIView):
    authentication_classes = [TokenAuthentication]
    
    def post(self, request, format=None):
        def get_postgresql_connection_string():
            db_settings = settings.DATABASES['default']
            return f"postgresql+psycopg2://{db_settings['USER']}:{db_settings['PASSWORD']}@{db_settings['HOST']}:{db_settings['PORT']}/{db_settings['NAME']}"
        data=request.data
        
        user = request.user
        openai_api_key = user.apikey
        encrypted_api_key_bytes = openai_api_key.encode('utf-8')
        decrypted_data = fernet.decrypt(encrypted_api_key_bytes)
        decrypted_text = decrypted_data.decode('utf-8')


        llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=decrypted_text)
        connection = get_postgresql_connection_string()
        COLLECTION_NAME = user.email
        embeddings = OpenAIEmbeddings(openai_api_key=decrypted_text)
        # prompt = hub.pull("rlm/rag-prompt")
        db = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=connection, use_jsonb=True)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        retriever = db.as_retriever(
            search_kwargs={"k": 10}
        )
       
        
        query = data.get('input')
        systemPrompt = data.get('sysPrompt')
        template = systemPrompt + """

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        try:
            resp = rag_chain.invoke(query)
        except Exception as e:
            return Response({"error": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        
        return Response(data={"response": resp}, status=status.HTTP_200_OK)


class Signup(APIView):

    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')
        apikey = request.data.get('apikey')
        langchainkey = request.data.get('langchainkey')

        # Create a new user instance
        user = get_user_model()()  # This creates an instance of the CustomProfile model

        # Set user details
        user.username = username
        user.email = email
        user.set_password(password)


        encAPIKey = fernet.encrypt(apikey.encode()).decode()
        user.apikey = encAPIKey
        
        encLangChainKey = fernet.encrypt(langchainkey.encode()).decode()
        user.langchainkey = encLangChainKey

        data = {
            'username': user.username,
            'email': user.email,
            'password': user.password,  # Use Django's set_password method for hashing
            'apikey': user.apikey,
            'langchainkey': user.langchainkey
        }

        serializer = CustomProfileSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class Login(APIView):
    def post(self, request):
        print(request.data)
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            # print(request.user.is_authenticated)
            login(request, user)
            # print(request.user.is_authenticated)
            
            # Generate or get an existing token for the user
            token, created = Token.objects.get_or_create(user=user)

            return Response({
                'token': token.key,
                'username': user.username,
                'email': user.email
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class Logout(APIView):
    authentication_classes = [TokenAuthentication]
    def post(self, request):
        request.user.auth_token.delete()
        return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)


class ChangeKey(APIView):
    authentication_classes = [TokenAuthentication]
    def post(self, request):
        user = request.user
        new_api_key = request.data.get('apikey')
        encAPIKey = fernet.encrypt(new_api_key.encode()).decode()
        user.apikey = encAPIKey
        user.save()
        return Response({"message": "Changed API Key successfully"}, status=status.HTTP_200_OK)

class ChangeLangchain(APIView):
    authentication_classes = [TokenAuthentication]
    def post(self, request):
        user = request.user
        new_langchain_api_key = request.data.get('langchainkey')
        encAPIKey = fernet.encrypt(new_langchain_api_key.encode()).decode()
        user.langchainkey = encAPIKey
        user.save()
        return Response({"message": "Changed Lanchain Key successfully"}, status=status.HTTP_200_OK)


class ChangePass(APIView):
    authentication_classes = [TokenAuthentication]
    def post(self, request):
        try :
            user = request.user
            new_password = request.data.get('password')
            user.set_password(new_password)
            user.save() 
            return Response({"message": "Changed password successfully"}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)