# stocks_app/serializers.py

from rest_framework import serializers
from .models import Stock, Index

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['stock_symbol', 'company_name', 'market', 'type']

class IndexSerializer(serializers.ModelSerializer):
    class Meta:
        model = Index
        fields = ['index_code', 'index_name', 'market']

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['stock_symbol', 'company_name', 'market', 'company_profile', 'type']