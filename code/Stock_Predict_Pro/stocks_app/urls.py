# stocks_app/urls.py

from django.urls import path
from .views import HotStocksView, SearchView, TodaysNewsView, RegisterUserView, LoginUserView, MarketsView

urlpatterns = [
    #首页的url
    path('hot_stocks/', HotStocksView.as_view(), name='hot_stocks'),#finished
    path('search/', SearchView.as_view(), name='search'),
    path('todays_news/', TodaysNewsView.as_view(), name='todays_news'),
    path('register/', RegisterUserView.as_view(), name='register_user'),
    path('login/', LoginUserView.as_view(), name='login_user'),
    path('markets/', MarketsView.as_view(), name='markets'),
    


]
