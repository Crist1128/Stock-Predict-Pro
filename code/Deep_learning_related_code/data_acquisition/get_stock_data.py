import requests
import time
import pandas as pd

api_key = "clf4lg9r01qoveppk620clf4lg9r01qoveppk62g"  #密钥
symbol = "JD" #股票唯一标识符
start_date = "2023-01-01" #始末时间
end_date = "2023-11-22"

# 将开始日期和结束日期转换为时间戳
start_timestamp = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
end_timestamp = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))

url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
response = requests.get(url)
data = response.json()

# 输出实时股价
print(f"实时股价({symbol}): {data['c']}")

url = f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start_timestamp}&to={end_timestamp}&token={api_key}'
response = requests.get(url)
data = response.json()

# 使用 pandas 处理数据
df = pd.DataFrame(data)


# 将数据保存为 Excel 文件
df.to_excel(f'{symbol}_stock_data.xlsx', index=False)

# 输出文件名
print(f"Excel 文件已生成: {symbol}_stock_data.xlsx")


