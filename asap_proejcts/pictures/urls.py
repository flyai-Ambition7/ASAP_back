from django.urls import path
from rest_framework import routers
from .views import ResultImageViewSet

router = routers.DefaultRouter()
router.register(r'result-image', ResultImageViewSet)

urlpatterns = router.urls
