from django.urls import path
from prediction.views import train_model_view,predict_view, history_view


urlpatterns = [
    path('', predict_view, name='predict'),
    path('history/', history_view, name='history'),
    path('train/', train_model_view, name='train_model'),
]