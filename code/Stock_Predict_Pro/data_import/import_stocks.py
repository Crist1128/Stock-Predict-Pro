'''
导入全部股票代码与名称到数据库中
'''
# import pandas as pd
# import pymysql

# # 连接数据库
# db_connection = pymysql.connect(
#     host='127.0.0.1',
#     user='root',
#     password='20021219',
#     db='stock_predict_pro_database',
#     charset='utf8mb4',
#     cursorclass=pymysql.cursors.DictCursor
# )

# # 读取xlsx文件
# file_path = 'A股所有股票代码与名称.xlsx'
# df = pd.read_excel(file_path)

# # 遍历DataFrame并插入数据到数据库
# with db_connection.cursor() as cursor:
#     for index, row in df.iterrows():
#         stock_symbol = row['代码']
#         company_name = row['名称']
#         market = 'A股'
#         company_profile = ''  # 暂时为空

#         # 构建插入数据的SQL语句
#         sql = f"INSERT INTO stocks_app_stock (stock_symbol, company_name, market, company_profile) VALUES ('{stock_symbol}', '{company_name}', '{market}', '{company_profile}')"

#         # 执行SQL语句
#         cursor.execute(sql)

# # 提交事务并关闭数据库连接
# db_connection.commit()
# db_connection.close()

'''
导入所有A股股票指数名称与代码到数据库中
'''
import pandas as pd
import pymysql

# 连接数据库
db_connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    # 记得修改
    password='123456',
    db='stock_predict_pro_database',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 读取xlsx文件
file_path = 'D:\dev_yjx\Stock-Predict-Pro\code\Stock_Predict_Pro\data_import\ChinaStockIndexSpotData.xlsx'
df = pd.read_excel(file_path)

# 遍历DataFrame并插入数据到数据库
with db_connection.cursor() as cursor:
    for index, row in df.iterrows():
        index_code = row['代码']
        index_name = row['名称']
        market = 'A股'
        index_information = ''  # 暂时为空

        # 构建插入数据的SQL语句
        sql = f"INSERT INTO stocks_app_index (index_code, index_name, market, index_information) VALUES ('{index_code}', '{index_name}', '{market}', '{index_information}')"

        # 执行SQL语句
        cursor.execute(sql)

# 提交事务并关闭数据库连接
db_connection.commit()
db_connection.close()
