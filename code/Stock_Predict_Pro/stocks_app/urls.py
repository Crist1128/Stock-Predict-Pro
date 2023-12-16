# stocks_app/urls.py

from django.urls import path
from .views import HotStocksView, SearchView, TodaysNewsView, RegisterUserView, LoginUserView, MarketsView,StockInfoAPIView,StockPriceChartAPIView,PredictDailyCloseAPIView

urlpatterns = [
    #首页的url
    path('hot_stocks/', HotStocksView.as_view(), name='hot_stocks'),#finished 热门股票接口
    path('search/', SearchView.as_view(), name='search'),#finished 搜索栏股票接口
    path('todays_news/', TodaysNewsView.as_view(), name='todays_news'),
    path('register/', RegisterUserView.as_view(), name='register_user'),
    path('login/', LoginUserView.as_view(), name='login_user'),
    path('markets/', MarketsView.as_view(), name='markets'),
    
    #股票页的url
    path('stock/<str:symbol>/', StockInfoAPIView.as_view(), name='stock-info'),#finished 股票信息获取接口
    path('stock/<str:symbol>/price_chart/', StockPriceChartAPIView.as_view(), name='stock-price-chart'),#finished 股票数据获取（1D,5D,1M,6M,1Y）
    path('stock/<str:symbol>/predict_daily_close/', PredictDailyCloseAPIView.as_view(), name='predict_daily_close'),

]
