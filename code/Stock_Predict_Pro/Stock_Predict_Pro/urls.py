"""Stock_Predict_Pro URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users_app import views
from stocks_app.views import HotStockList
from stocks_app.views import StockNews

urlpatterns = [
    path("admin/", admin.site.urls),
    path('index/', views.index_users),
    path('api/hot_stocks/', HotStockList.as_view(), name='hot_stock_list'),
    path('api/todays_news/', StockNews.as_view(), name='stock_new')
]
