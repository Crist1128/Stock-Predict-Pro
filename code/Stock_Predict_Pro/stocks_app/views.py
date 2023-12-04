# stocks_app/views.py
# stocks_app/views.py

import akshare as ak
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Stock, Index
from .serializers import StockSerializer, IndexSerializer
from django.db.models import Q

class HotStocksView(APIView):
    def get(self, request):
        # 使用 akshare 获取热门股票数据
        stock_hot_rank_em_df = ak.stock_hot_rank_em()
        top_50_stocks = stock_hot_rank_em_df.head(50)

        # 将数据格式化为 API 接口定义的 JSON 格式
        hot_stocks_data = []
        for index, row in top_50_stocks.iterrows():
            hot_stock = {
                "stock_symbol": row["代码"],
                "company_name": row["股票名称"],
                "latest_close_price": row["最新价"],
                "change_amount": row["涨跌额"],
                "change_percentage": row["涨跌幅"]
            }
            hot_stocks_data.append(hot_stock)

        return Response(hot_stocks_data)


class SearchView(APIView):
    '''
    - 可以模糊搜索，但是返回结果比较多，前端注意进行分割
    '''
    def get(self, request):
        query = request.GET.get('query', '')

        # 在股票表中搜索
        stock_results = Stock.objects.filter(
            Q(stock_symbol__icontains=query) |
            Q(company_name__icontains=query)
        )
        stock_serializer = StockSerializer(stock_results, many=True)

        # 在指数表中搜索
        index_results = Index.objects.filter(
            Q(index_code__icontains=query) |
            Q(index_name__icontains=query)
        )
        index_serializer = IndexSerializer(index_results, many=True)

        # 将结果格式化为 API 接口定义的 JSON 格式
        search_results = stock_serializer.data + index_serializer.data

        return Response(search_results)

class TodaysNewsView(APIView):
    def get(self, request):
        # 实现获取今日财经新闻的逻辑
        # ...
        pass

class RegisterUserView(APIView):
    def post(self, request):
        # 实现用户注册逻辑
        # ...
        pass

class LoginUserView(APIView):
    def post(self, request):
        # 实现用户登录逻辑
        # ...
        pass

class MarketsView(APIView):
    def get(self, request):
        # 实现获取导航栏信息的逻辑
        # ...
        pass
