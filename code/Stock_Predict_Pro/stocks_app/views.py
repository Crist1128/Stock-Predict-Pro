from django.shortcuts import render
# views.py
from rest_framework import generics
from .models import HotStock  # 继承热门股票API信息
from .models import News  # 继承股票新闻API信息
from .serializers import HotStockSerializer
from .serializers import StockNewslizer


class HotStockList(generics.ListAPIView):
    queryset = HotStock.objects.all()
    serializer_class = HotStockSerializer


class StockNews(generics.ListAPIView):
    queryset = News.objects.all()
    serializer_class = StockNewslizer
