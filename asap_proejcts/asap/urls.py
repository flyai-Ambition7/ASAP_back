from django.urls import path
from rest_framework import routers
from .views import CommonInfoViewSet, ItemInfoViewSet

router = routers.DefaultRouter()
router.register(r'common-info', CommonInfoViewSet)
router.register(r'item-info', ItemInfoViewSet)


urlpatterns = router.urls
