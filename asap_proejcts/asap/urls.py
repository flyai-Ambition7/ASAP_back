from django.urls import path
from rest_framework import routers
from .views import ItemInfoViewSet, GeneratedDataViewSet, ResultImageViewSet

router = routers.DefaultRouter()
router.register(r'item-info', ItemInfoViewSet)
router.register(r'generated-data', GeneratedDataViewSet)
router.register(r'result-image', ResultImageViewSet)

urlpatterns = router.urls
