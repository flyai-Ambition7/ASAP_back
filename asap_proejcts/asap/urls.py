from django.urls import path
from rest_framework import routers
from .views import ItemInfoViewSet, ResultDataViewSet

router = routers.DefaultRouter()
# router.register(r'common-info', CommonInfoViewSet)
router.register(r'item-info', ItemInfoViewSet)
router.register(r'result-data', ResultDataViewSet)


urlpatterns = router.urls
