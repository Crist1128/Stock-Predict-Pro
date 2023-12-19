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
