import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import akshare as ak

class StockPredictor:
    def __init__(self, stock_code, fq_type, predict_days, start_date, end_date):
        self.stock_code = stock_code
        self.fq_type = fq_type
        self.predict_days = predict_days
        self.start_date = start_date
        self.end_date = end_date

    def fetch_stock_data(self):
        time_line = "daily"
        data = ak.stock_zh_a_hist(symbol=self.stock_code, period=time_line,
                                  start_date=self.start_date, end_date=self.end_date, adjust=self.fq_type)
        return data

    def preprocess_data(self, data):
        price = data[['收盘']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price.loc[:, '收盘'] = scaler.fit_transform(price['收盘'].values.reshape(-1, 1))
        return price, scaler

    def split_data(self, stock, lookback):
        data_raw = stock.to_numpy()
        data = []

        for index in range(len(data_raw) - lookback):
            data.append(data_raw[index: index + lookback])

        data = np.array(data)
        test_set_size = int(np.round(0.2 * data.shape[0]))
        train_set_size = data.shape[0] - test_set_size

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1, :]

        return x_train, y_train, x_test

    def train_model(self, x_train, y_train_lstm, input_dim, hidden_dim, num_layers, output_dim, num_epochs):
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
                out, _ = self.lstm(x, (h0.detach(), c0.detach()))
                out = self.fc(out[:, -1, :])
                return out

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train_lstm).type(torch.Tensor)

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train_lstm)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return model

    def predict(self, model, x_test, scaler):
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_predict = np.empty((self.predict_days, 1))

        for i in range(self.predict_days):
            y = model(x_test)
            x_test = torch.cat((x_test[:, 1:, :], y.unsqueeze(0)), dim=1)
            y_predict[i, 0] = scaler.inverse_transform(y.detach().numpy())

        return y_predict

    def run_prediction(self):
        data = self.fetch_stock_data()
        price, scaler = self.preprocess_data(data)
        x_train, y_train, x_test = self.split_data(price, lookback=60)
        y_train_lstm = y_train[:, -1, :]
        model = self.train_model(x_train, y_train_lstm, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, num_epochs=100)

        x_predict = x_train[-1, 1 - 60:, :].unsqueeze(0)
        y_predict = self.predict(model, x_predict, scaler)

        return scaler.inverse_transform(y_predict.detach().numpy())

# 示例用法
stock_code = "000001"
fq_type = "hfq"
predict_days = 14
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
end_date = datetime.now().strftime("%Y%m%d")

stock_predictor = StockPredictor(stock_code, fq_type, predict_days, start_date, end_date)
predicted_prices = stock_predictor.run_prediction()
print(predicted_prices)
