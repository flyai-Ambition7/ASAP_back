from django.urls import path
from rest_framework import routers
from .views import ItemInfoViewSet,  ResultImageViewset
from django.conf import settings
from django.conf.urls.static import static


router = routers.DefaultRouter()
router.register(r'item-info', ItemInfoViewSet)
router.register(r'result-data', ResultImageViewset)

urlpatterns = router.urls
