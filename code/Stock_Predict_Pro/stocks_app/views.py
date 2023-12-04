# stocks_app/views.py
# stocks_app/views.py

import akshare as ak
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Stock, Index
from .serializers import StockSerializer, IndexSerializer
from django.db.models import Q
from rest_framework import status

'''
首页url及其view函数编写

'''


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


'''
股票页url及view函数编写
'''

from django.utils import timezone

class StockInfoAPIView(APIView):
    def get(self, request, symbol):
        try:
            symbol=symbol[2:]
            # 获取实时行情数据
            stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()

            # 查找特定股票信息
            stock_info = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df['代码'] == symbol]

            # 检查是否有匹配的记录
            if len(stock_info) == 0:
                raise ValueError(f"股票代码 {symbol} 未找到对应记录")

            # 获取请求时间
            request_time = timezone.now()

            # 转换数据格式
            formatted_data = {
                "stock_name": stock_info['名称'].iloc[0],
                "symbol": stock_info['代码'].iloc[0],
                "current_price": {
                    "price": float(stock_info['最新价'].iloc[0]),
                    "timestamp": request_time  # 使用请求时间
                },
                "previous_close": float(stock_info['昨收'].iloc[0]),
                "price_range": {
                    "low": float(stock_info['最低'].iloc[0]),
                    "high": float(stock_info['最高'].iloc[0])
                },
                "year_to_date_return": float(stock_info['年初至今涨跌幅'].iloc[0]),
                "market_cap": f"{float(stock_info['总市值'].iloc[0]) / 1e9:.2f}B",
                "average_volume": f"{float(stock_info['成交量'].iloc[0]) / 1e6:.2f}M",
                # 可继续添加其他字段
            }

            return Response(formatted_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


