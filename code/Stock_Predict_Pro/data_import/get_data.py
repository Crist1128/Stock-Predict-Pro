# import akshare as ak

# # 获取指定时间范围内股票代码为000001的1分钟级别分时数据
# stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date="2023-12-05 09:32:00", end_date="2023-12-05 15:00:00", period='1', adjust='')

# # 提取时间、收盘价和成交量
# time_close_volume_df = stock_zh_a_hist_min_em_df[['时间', '收盘', '成交量']]

# # 打印结果
# print(time_close_volume_df)


# import akshare as ak

# def get_stock_data(symbol, start_date, end_date, time_range):
#     # 根据不同的时间范围设置相应的period值
#     if time_range == '1D':
#         period = '1'
#     elif time_range == '5D':
#         period = '15'
#     else:
#         period = '1'

#     # 获取股票数据
#     stock_data = ak.stock_zh_a_hist_min_em(symbol=symbol, start_date=start_date, end_date=end_date, period=period, adjust='')

#     # 提取时间、收盘价和交易量
#     time_close_volume_df = stock_data[['时间', '收盘', '成交量']]

#     return time_close_volume_df


# stock_data_1D = get_stock_data(symbol="000001", start_date="2023-12-05 09:32:00", end_date="2023-12-05 15:00:00", time_range='1D')
# stock_data_5D = get_stock_data(symbol="000001", start_date="2023-12-01 09:32:00", end_date="2023-12-05 15:00:00", time_range='5D')

# # 打印结果
# print("1D精确度数据:")
# print(stock_data_1D)

# print("\n5D精确度数据:")
# print(stock_data_5D)

from datetime import datetime, timedelta
import akshare as ak


def get_stock_data_daily(symbol, time_range, adjust=''):
    # 获取当前日期
    current_date = datetime.now().strftime('%Y%m%d')

    # 计算开始日期
    if time_range == '1M':
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    elif time_range == '6M':
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    elif time_range == '1Y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    else:
        raise ValueError("Invalid time_range value. Supported values: '1M', '6M', '1Y'")

    # 获取股票数据
    stock_data = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_date, end_date=current_date,
                                    adjust=adjust)

    # 提取时间、收盘价和交易量
    time_close_volume_df = stock_data[['日期', '收盘', '成交量']]

    return time_close_volume_df


# 示例
symbol = "600519"  # 以贵州茅台为例
stock_data_1M = get_stock_data_daily(symbol=symbol, time_range='1M', adjust="qfq")
stock_data_6M = get_stock_data_daily(symbol=symbol, time_range='6M', adjust="hfq")
stock_data_1Y = get_stock_data_daily(symbol=symbol, time_range='1Y')

# 打印结果
print("1M精确度数据:")
print(stock_data_1M)

print("\n6M精确度数据:")
print(stock_data_6M)

print("\n1Y精确度数据:")
print(stock_data_1Y)
