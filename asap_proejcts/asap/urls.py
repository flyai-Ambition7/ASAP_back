from django.urls import path
from rest_framework import routers
from .views import CommonInfoViewSet, ItemInfoViewSet

router = routers.DefaultRouter()
router.register(r'common-info', CommonInfoViewSet)
router.register(r'iteminfo', ItemInfoViewSet)

urlpatterns = router.urls
