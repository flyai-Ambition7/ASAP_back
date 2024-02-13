from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserSerializer
from .models import User
from django.utils import timezone
# Create your views here.
class SignupView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

class LoginView(APIView):
    def post(self, request):
        username = request.data['username']
        password = request.data['password']
        user = User.objects.filter(username=username).first()
                
        # user가 없는 경우
        if user is None:
            raise AuthenticationFailed('User Not Found!')
        
        # 비밀번호가 일치하지 않는 경우
        if not user.check_password(password):
            raise AuthenticationFailed('Incorrect password!')

        # 토큰 생성
        refresh = RefreshToken.for_user(user)

        # 토큰 payload
        payload = {
            'id': user.id,
            'exp': timezone.now() + timezone.timedelta(minutes=60),
            'iat': timezone.now()
        }
        

        # 토큰 반환
        return Response({
            'access_token': str(refresh.access_token),
            'refresh_token': str(refresh)
        })

class LogoutView(APIView):
    def post(self, request):
        response = Response()
        
        # 클라이언트에서 JWT 토큰 제거 위해 쿠기 삭제
        response.delete_cookie('jwt')
        response.data = {
            'message' : 'success' # 응답 메시지
        }
        return response