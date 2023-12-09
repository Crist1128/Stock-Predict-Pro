import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import plotly.graph_objects as go
import akshare as ak
from datetime import datetime, timedelta

# 前端接口
stock_type = "000001"  # 接口2：股票代码
# 接口3：股票名称
fq_type = ""  # 接口4：复权类型
predict_days = 14  # 接口5：预测天数

time_line = "daily"
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)
end_date = str(end_date).replace('-', '')
start_date = str(start_date).replace('-', '')

# 接口1：股票类型
data = ak.stock_zh_a_hist(symbol=stock_type, period=time_line, start_date=start_date, end_date=end_date, adjust=fq_type)
# 可换成任意股数据集，只要含“日期”(yyyy-mm-dd日期格式)、“收盘”即可

price = data[['收盘']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price.loc[:, '收盘'] = scaler.fit_transform(price['收盘'].values.reshape(-1, 1))


def split_data(stock, lookback):
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

    return [x_train, y_train, x_test, y_test]


lookback = 60
x_train, y_train, x_test, y_test = split_data(price, lookback)
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


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


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

y_predict = np.empty((predict_days, 1))  # 储存预测结果
x_predict = x_test[-1, 1 - lookback:, :].unsqueeze(0)
for i in range(predict_days):
    y = model(x_predict)
    x_predict = torch.cat((x_predict[:, 1:, :], y.unsqueeze(0)), dim=1)
    y_predict[i, 0] = scaler.inverse_transform(y.detach().numpy())
