# serializers.py
from rest_framework import serializers
from .models import HotStock
from .models import News


class HotStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = HotStock
        fields = '__all__'


class StockNewslizer:
    class Meta:
        model = News
        fields = '__all__'
