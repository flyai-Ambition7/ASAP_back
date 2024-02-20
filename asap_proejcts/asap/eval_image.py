import time
import re
from dotenv import load_dotenv
import os
from draw_image import *
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 문장 임베딩 모델 로드
load_dotenv(verbose=True)
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
AZURE_SUBSCRIPTION_KEY=os.getenv("AZURE_SUBSCRIPTION_KEY")
AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")

nltk.download('punkt')
nltk.download('wordnet')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to perform OCR using Azure Computer Vision
def get_text_by_OCR(image_data):
    computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_SUBSCRIPTION_KEY))
    read_response = computervision_client.read_in_stream(image_data, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    # Wait for the OCR operation to complete
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    text = ""
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for text_line in text_result.lines:
                text += text_line.text + " "
    return text

# Text preprocessing
# find text in ' ' symbols from prompt
def get_intended_text(prompt):
    match = re.search(r"'(.*?)'", prompt)
    intended_text = match.group(1)
    return intended_text

def normalize(text):
    return ''.join(char.lower() for char in text if char.isalnum() or char.isspace()).strip()

# 문장 전처리 함수
def embed_text(sentence):
    # 소문자 변환
    sentence = sentence.lower()
    # 토큰화
    tokens = word_tokenize(sentence)
    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # 공백 제거
    tokens = [word for word in tokens if word.isalnum()]
    # 재구성된 문장 반환
    sentence = ' '.join(tokens)
    return model.encode(sentence)

# 두 문장 사이의 유사도 측정 함수
def calculate_similarity(sentence1, sentence2):
    # 전처리 및 임베딩 계산
    embedding1, embedding2 = embed_text(sentence1), embed_text(sentence2)
    # 코사인 유사도 계산
    score = cosine_similarity([embedding1], [embedding2])[0][0]
    return score 

def evalulate_image(prompt,img):
    generated_text=normalize(get_text_by_OCR(img))
    intended_text=normalize(get_intended_text(prompt))
    return calculate_similarity(generated_text,intended_text)