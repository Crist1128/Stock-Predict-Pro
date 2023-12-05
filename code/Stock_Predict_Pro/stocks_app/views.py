# stocks_app/views.py

import akshare as ak  # 导入akshare库用于获取股票数据
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Stock, Index
from .serializers import StockSerializer, IndexSerializer
from django.db.models import Q
from rest_framework import status
from django.utils import timezone
from datetime import datetime, timedelta

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

class StockInfoAPIView(APIView):
    def get(self, request, symbol):
        try:
            # 将 symbol 转换为小写字母
            symbol = symbol.lower()

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
                    "timestamp": request_time.isoformat()  # 使用请求时间，并以 ISO 8601 格式返回
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



class StockPriceChartAPIView(APIView):
    def get(self, request, symbol):
        try:
            # 不需要前缀字母
            symbol = symbol[2:]

            # 获取时间范围参数，默认为 "1D"
            time_range = request.query_params.get('time_range', '1D')

            # 获取复权方式参数，默认为 "none"
            adjust = request.query_params.get('adjust', 'none')

            # 获取请求时间
            request_time = datetime.now()

            # 计算开始时间和结束时间
            end_date = request_time.strftime("%Y-%m-%d %H:%M:%S")
            start_date = self.calculate_start_date(request_time, time_range)

            # 获取股票数据
            if time_range in ['1D', '5D']:
                stock_data = self.get_intraday_stock_data(symbol, start_date, end_date, time_range)  # 按分钟分割暂不提供复权
            else:
                stock_data = self.get_daily_stock_data(symbol, start_date, end_date, time_range, adjust)

            # 转换数据格式
            formatted_data = self.format_stock_data(stock_data, time_range)

            return Response(formatted_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def calculate_start_date(self, end_date, time_range):
        # 根据时间范围计算开始时间
        if time_range == '1D':
            return (end_date - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        elif time_range == '5D':
            return (end_date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        elif time_range == '1M':
            return (end_date - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        elif time_range == '6M':
            return (end_date - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")
        elif time_range == '1Y':
            return (end_date - timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Invalid time_range value. Supported values: '1D', '5D', '1M', '6M', '1Y'")

    def get_intraday_stock_data(self, symbol, start_date, end_date, time_range):
        # 根据不同的时间范围设置相应的 period 值
        if time_range == '1D':
            period = '1'
        elif time_range == '5D':
            period = '15'
        else:
            raise ValueError("Invalid time_range value. Supported values: '1D', '5D'")

        # 获取股票数据
        stock_data = ak.stock_zh_a_hist_min_em(symbol=symbol, start_date=start_date, end_date=end_date, period=period, adjust='')

        # 提取时间、收盘价和交易量
        time_close_volume_df = stock_data[['时间', '收盘', '成交量']]

        return time_close_volume_df

    def get_daily_stock_data(self, symbol, start_date, end_date, time_range, adjust):
        # 根据不同的时间范围设置相应的 period 值
        if time_range in ['1M', '6M', '1Y']:
            period = 'daily'
        else:
            raise ValueError("Invalid time_range value. Supported values: '1M', '6M', '1Y'")
        
        start_date_str = start_date.replace('-', '').split()[0]
        end_date_str = end_date.replace('-', '').split()[0]

        # 获取股票数据
        if adjust == 'none':
            adjust = ''
        stock_data = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date_str, end_date=end_date_str, adjust=adjust)
        # 提取时间、收盘价和交易量
        time_close_volume_df = stock_data[['日期', '收盘', '成交量']].rename(columns={'日期': '时间'})

        return time_close_volume_df

    def format_stock_data(self, stock_data, time_range):
        # 获取开始价格和结束价格
        start_price = float(stock_data['收盘'].iloc[0])
        end_price = float(stock_data['收盘'].iloc[-1])

        # 计算涨跌幅
        percentage_change = ((end_price - start_price) / start_price) * 100

        # 转换数据格式
        formatted_data = {
            "time_range": time_range,
            "price_data": [
                {"time": row['时间'], "price": float(row['收盘']), "volume": float(row['成交量'])}
                for _, row in stock_data.iterrows()
            ],
            "start_price": start_price,
            "end_price": end_price,
            "percentage_change": round(percentage_change, 2)
        }

        return formatted_data

