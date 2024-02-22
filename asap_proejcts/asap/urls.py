from django.urls import path
from rest_framework import routers
from .views import ItemInfoViewSet, GeneratedDataViewSet, ResultImageViewSet
from django.conf import settings
from django.conf.urls.static import static


router = routers.DefaultRouter()
router.register(r'item-info', ItemInfoViewSet)
router.register(r'generated-data', GeneratedDataViewSet)
router.register(r'result-image', ResultImageViewSet)

urlpatterns = router.urls + static('/', document_root=settings.MEDIA_ROOT)
