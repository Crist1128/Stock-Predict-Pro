import akshare as ak

stock_zh_index_spot_df = ak.stock_zh_index_spot()
stock_zh_index_spot_df.to_excel("ChinaStockIndexSpotData.xlsx", index=False)

print("中国股指实时数据已成功保存到 ChinaStockIndexSpotData.xlsx")
