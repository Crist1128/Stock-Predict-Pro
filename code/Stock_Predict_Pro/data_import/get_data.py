import akshare as ak

# 获取实时行情数据
stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()

# 查找股票代码为000001的信息
stock_000001_info = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df['代码'] == '000001']

print(stock_000001_info)
