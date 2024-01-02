import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import plotly.graph_objects as go
import akshare as ak
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, stock_code="000001", fq_type="hfq", predict_days=14):
        self.training_days = 365
        self.stock_code = stock_code
        self.fq_type = fq_type
        self.predict_days = predict_days

    def fetch_stock_data(self):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.training_days)
        end_date_str = str(end_date).replace('-', '')
        start_date_str = str(start_date).replace('-', '')

        data = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", start_date=start_date_str,
                                  end_date=end_date_str, adjust=self.fq_type)
        return data

    def preprocess_data(self, data):
        price = data[['收盘']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price.loc[:, '收盘'] = scaler.fit_transform(price['收盘'].values.reshape(-1, 1))

        lookback = 60
        x_train, y_train, x_test, y_test = self.split_data(price, lookback)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

        return x_train, x_test, y_train_lstm, scaler, lookback

    def split_data(self, stock, lookback):
        data_raw = stock.to_numpy()
        data = []

        for index in range(len(data_raw) - lookback):
            data.append(data_raw[index: index + lookback])

        data = np.array(data)
        test_set_size = int(np.round(0.2 * data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1, :]
        y_test = data[train_set_size:, -1, :]

        return x_train, y_train, x_test, y_test

    def train_model(self, x_train, y_train_lstm, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, num_epochs=100):
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train_lstm)
            hist[t] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def predict_future(self, model, x_test, scaler, lookback):
        y_predict = np.empty((self.predict_days, 1))
        x_predict = x_test[-1, 1 - lookback:, :].unsqueeze(0)

        for i in range(self.predict_days):
            y = model(x_predict)
            x_predict = torch.cat((x_predict[:, 1:, :], y.unsqueeze(0)), dim=1)
            y_predict[i, 0] = scaler.inverse_transform(y.detach().numpy())

        return y_predict

    def generate_dates(self, last_date):
        future_dates = [last_date + timedelta(days=i) for i in range(1, self.predict_days + 1)]
        return future_dates

    def run_prediction(self):
        data = self.fetch_stock_data()
        x_train, x_test, y_train_lstm, scaler, lookback = self.preprocess_data(data)
        model = self.train_model(x_train, y_train_lstm)
        y_predict = self.predict_future(model, x_test, scaler, lookback)

        last_date = datetime.strptime(str(data.iloc[-1]['日期']), '%Y-%m-%d')
        future_dates = self.generate_dates(last_date)

        result_dict = {}
        for i in range(self.predict_days):
            result_dict[future_dates[i].strftime('%Y-%m-%d')] = round(y_predict[i, 0], 2)

        return result_dict



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# Example Usage
if __name__ == "__main__":
    predictor = StockPredictor(stock_code="000001", fq_type="", predict_days=14)
    result = predictor.run_prediction()
    print(result)

# # 导入所需的库
# import numpy as np  # 导入NumPy库，用于进行科学计算
# import pandas as pd  # 导入Pandas库，用于数据处理和分析
# from sklearn.preprocessing import MinMaxScaler  # 从sklearn库导入MinMaxScaler，用于数据归一化处理
# import torch  # 导入PyTorch库，一个用于深度学习的库
# import torch.nn as nn  # 导入PyTorch的神经网络模块
# import plotly.graph_objects as go  # 导入Plotly库中的graph_objects模块，用于创建交互式图形
# import akshare as ak  # 导入AkShare库，一个开源财经数据接口库
# from datetime import datetime, timedelta  # 从datetime库导入datetime和timedelta，用于处理日期和时间

# # 定义一个股票预测的类
# class StockPredictor:
#     # 初始化函数，设置默认参数
#     def __init__(self, stock_code="000001", fq_type="hfq", predict_days=14):
#         self.training_days = 365  # 设置用于训练的天数
#         self.stock_code = stock_code  # 股票代码
#         self.fq_type = fq_type  # 复权类型
#         self.predict_days = predict_days  # 预测天数

#     # 获取股票数据的函数
#     def fetch_stock_data(self):
#         end_date = datetime.now().date()  # 获取当前日期
#         start_date = end_date - timedelta(days=self.training_days)  # 计算开始日期
#         # 将日期格式化为字符串，用于API调用
#         end_date_str = str(end_date).replace('-', '')
#         start_date_str = str(start_date).replace('-', '')

#         # 使用AkShare库获取股票历史数据
#         data = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", start_date=start_date_str,
#                                   end_date=end_date_str, adjust=self.fq_type)
#         return data

#     # 数据预处理函数
#     def preprocess_data(self, data):
#         price = data[['收盘']]  # 提取收盘价
#         scaler = MinMaxScaler(feature_range=(-1, 1))  # 创建归一化的缩放器
#         # 归一化处理收盘价数据
#         price.loc[:, '收盘'] = scaler.fit_transform(price['收盘'].values.reshape(-1, 1))

#         lookback = 60  # 设置回看天数
#         # 将数据分割为训练集和测试集
#         x_train, y_train, x_test, y_test = self.split_data(price, lookback)
#         # 将数据转换为PyTorch张量
#         x_train = torch.from_numpy(x_train).type(torch.Tensor)
#         x_test = torch.from_numpy(x_test).type(torch.Tensor)
#         y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

#         return x_train, x_test, y_train_lstm, scaler, lookback

#     # 数据分割函数
#     def split_data(self, stock, lookback):
#         data_raw = stock.to_numpy()  # 将数据框转换为NumPy数组
#         data = []

#         # 创建一个数据列表，包含过去lookback天的数据
#         for index in range(len(data_raw) - lookback):
#             data.append(data_raw[index: index + lookback])

#         data = np.array(data)
#         # 定义测试集的大小
#         test_set_size = int(np.round(0.2 * data.shape[0]))
#         train_set_size = data.shape[0] - (test_set_size)

#         # 分割数据为训练集和测试集
#         x_train = data[:train_set_size, :-1, :]
#         y_train = data[:train_set_size, -1, :]

#         x_test = data[train_set_size:, :-1, :]
#         y_test = data[train_set_size:, -1, :]

#         return x_train, y_train, x_test, y_test

#     # 训练模型的函数
#     def train_model(self, x_train, y_train_lstm, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, num_epochs=100):
#         model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)  # 初始化LSTM模型
#         criterion = torch.nn.MSELoss()  # 使用均方误差作为损失函数
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器

#         hist = np.zeros(num_epochs)
#         for t in range(num_epochs):
#             y_train_pred = model(x_train)  # 进行预测

#             loss = criterion(y_train_pred, y_train_lstm)  # 计算损失
#             hist[t] = loss.item()  # 记录损失

#             optimizer.zero_grad()  # 清空之前的梯度
#             loss.backward()  # 反向传播计算新的梯度
#             optimizer.step()  # 更新模型参数

#         return model

#     # 预测未来股价的函数
#     def predict_future(self, model, x_test, scaler, lookback):
#         y_predict = np.empty((self.predict_days, 1))  # 创建一个空数组用于存储预测结果
#         x_predict = x_test[-1, 1 - lookback:, :].unsqueeze(0)  # 获取用于预测的最后一组数据

#         for i in range(self.predict_days):
#             y = model(x_predict)  # 进行预测
#             x_predict = torch.cat((x_predict[:, 1:, :], y.unsqueeze(0)), dim=1)  # 更新预测数据
#             y_predict[i, 0] = scaler.inverse_transform(y.detach().numpy())  # 将预测结果转换回原始范围

#         return y_predict

#     # 生成未来日期的函数
#     def generate_dates(self, last_date):
#         future_dates = [last_date + timedelta(days=i) for i in range(1, self.predict_days + 1)]  # 创建未来日期列表
#         return future_dates

#     # 运行预测的函数
#     def run_prediction(self):
#         data = self.fetch_stock_data()  # 获取股票数据
#         # 预处理数据并训练模型
#         x_train, x_test, y_train_lstm, scaler, lookback = self.preprocess_data(data)
#         model = self.train_model(x_train, y_train_lstm)
#         y_predict = self.predict_future(model, x_test, scaler, lookback)  # 进行预测

#         # 获取最后一个交易日的日期
#         last_date = datetime.strptime(str(data.iloc[-1]['日期']), '%Y-%m-%d')
#         future_dates = self.generate_dates(last_date)  # 生成未来日期

#         result_dict = {}
#         for i in range(self.predict_days):
#             # 将预测结果和日期存储在字典中
#             result_dict[future_dates[i].strftime('%Y-%m-%d')] = round(y_predict[i, 0], 2)

#         return result_dict

# # 定义LSTM模型类
# class LSTM(nn.Module):
#     # 初始化函数
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim  # 隐藏层维度
#         self.num_layers = num_layers  # LSTM层数

#         # 定义LSTM层和全连接层
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     # 前向传播函数
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # 通过LSTM层
#         out = self.fc(out[:, -1, :])  # 通过全连接层
#         return out

# # 如果是主程序执行，则进行以下操作
# if __name__ == "__main__":
#     predictor = StockPredictor(stock_code="000001", fq_type="", predict_days=14)  # 创建预测器实例
#     result = predictor.run_prediction()  # 运行预测
#     print(result)  # 打印结果
