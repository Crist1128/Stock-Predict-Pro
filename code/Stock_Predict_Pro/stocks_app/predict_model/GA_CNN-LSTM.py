# 导入所需库
import akshare as ak  # 股票数据获取库
import numpy as np  # 科学计算库
import pandas as pd  # 数据处理库
from datetime import datetime  # 日期和时间库
from sklearn.preprocessing import MinMaxScaler  # 数据归一化处理
import torch  # PyTorch库，用于深度学习
import torch.nn as nn  # PyTorch的神经网络模块
from torch.utils.data import DataLoader, TensorDataset  # PyTorch的数据加载器和数据集
import torch.optim as optim  # PyTorch的优化器
import random  # 生成随机数
import matplotlib.pyplot as plt  # 绘图库

# 定义获取股票数据的函数
def fetch_stock_data(symbol, start_date, end_date, adjust=""):
    try:
        # 调用akshare库获取指定日期范围内的股票历史数据
        stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

# 定义获取特定日期股票数据的函数
def get_data_for_specific_date(stock_data, specific_date):
    try:
        specific_date_obj = pd.to_datetime(specific_date)  # 将日期字符串转换为日期对象
        stock_data['日期'] = pd.to_datetime(stock_data['日期']).dt.date  # 转换日期格式
        specific_date_data = stock_data[stock_data['日期'] == specific_date_obj.date()]  # 筛选指定日期的数据
        return specific_date_data
    except Exception as e:
        print(f"Error getting data for specific date: {e}")
        return None

# 数据预处理函数
def preprocess_data(data, look_back=60):
    # 创建两个归一化器，一个用于输入特征，另一个用于收盘价
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_close = MinMaxScaler(feature_range=(-1, 1))
    
    # 归一化处理数据
    scaled_data = scaler.fit_transform(data[['开盘', '最高', '最低', '收盘', '成交量']].values)
    close_prices = scaler_close.fit_transform(data[['收盘']].values)  # 单独缩放收盘价
    
    X, y = [], []
    # 生成训练数据集
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
        y.append(close_prices[i, 0])  # 使用单独缩放的收盘价
    return np.array(X), np.array(y), scaler, scaler_close  # 返回两个scaler

# 定义CNN-LSTM模型类
class CNNLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, kernel_size, num_layers, output_dim=1):
        super(CNNLSTM, self).__init__()
        # 定义CNN层
        self.cnn = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.relu = nn.ReLU()
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 前向传播
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# 训练模型的函数
def train_model(X_train, y_train, params):
    # 创建模型实例并设置参数
    model = CNNLSTM(num_features=5, hidden_dim=params['hidden_dim'], kernel_size=params['kernel_size'], num_layers=params['num_layers'], output_dim=1)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
    
    # 准备数据加载器
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 训练模型
    for epoch in range(100):  # 可根据需要调整epoch数量
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    return model

# 定义遗传算法的个体类
class Individual:
    def __init__(self, param_space):
        self.param_space = param_space
        self.params = {k: random.choice(v) for k, v in param_space.items()}
        self.fitness = None

    # 突变函数
    def mutate(self, mutation_rate):
        for param in self.params:
            if random.random() < mutation_rate:
                self.params[param] = random.choice(self.param_space[param])

# 交叉函数
def crossover(parent1, parent2):
    child_params = {}
    for param in parent1.params:
        child_params[param] = parent1.params[param] if random.random() < 0.5 else parent2.params[param]
    return Individual(parent1.param_space), Individual(parent1.param_space)

# 生成初始种群
def generate_initial_population(pop_size, para_sizm_space):
    return [Individual(param_space) for _ in range(pop_size)]

# 计算适应度函数
def compute_fitness(individual, X_train, y_train, X_val, y_val):
    try:
        model = train_model(X_train, y_train, individual.params)
        val_loss = evaluate_model(model, X_val, y_val)
        return 1 / val_loss  # 适应度是损失的倒数
    except Exception as e:
        print(f"Error during fitness computation: {e}")
        return 0  # 返回一个默认的适应度值，例如0

# 评估模型函数
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_val).float())
        val_loss = nn.MSELoss()(predictions, torch.from_numpy(y_val).float().unsqueeze(1))
    return val_loss.item()

# 父代选择函数
def select_parents(population):
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:2]

# 遗传算法主函数
def genetic_algorithm(X_train, y_train, X_val, y_val, param_space, pop_size=10, num_generations=5, mutation_rate=0.1):
    population = generate_initial_population(pop_size, param_space)

    for generation in range(num_generations):
        for individual in population:
            if individual.fitness is None:
                individual.fitness = compute_fitness(individual, X_train, y_train, X_val, y_val)

        parents = select_parents(population)
        population = []

        for _ in range(pop_size // 2):
            child1, child2 = crossover(parents[0], parents[1])
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            population.extend([child1, child2])

    # 确保所有个体都有适应度值
    if all(individual.fitness is not None for individual in population):
        best_individual = max(population, key=lambda x: x.fitness)
    else:
        print("Some individuals do not have a fitness value.")
        best_individual = population[0]  # 或选择一个默认的最佳个体

    return best_individual.params

# 预测和评估函数
def predict_and_evaluate(model, X_test, y_test, scaler, scaler_close):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float())
        predicted_prices = scaler_close.inverse_transform(predictions.numpy())  # 使用scaler_close进行逆变换
        actual_prices = scaler_close.inverse_transform(y_test.reshape(-1, 1))

        mse = nn.MSELoss()(predictions, torch.from_numpy(y_test).float().unsqueeze(1)).item()

        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Prices')
        plt.plot(predicted_prices, label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        return mse

# 主流程
if __name__ == "__main__":
    print("开始获取数据...")
    stock_symbol = "000001"  # 股票代码
    start_date = "20221201"  # 开始日期
    end_date = "20231230"  # 结束日期
    look_back = 60  # 定义回溯天数

    data = fetch_stock_data(stock_symbol, start_date, end_date)
    if data is not None:
        print("数据获取成功，开始数据预处理...")
        X, y, scaler, scaler_close = preprocess_data(data, look_back)  # 接收两个scaler

        split_index = len(X) - look_back
        if split_index > 0:
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

            print("数据预处理完成，开始遗传算法优化...")
            # 定义参数空间
            param_space = {
                'hidden_dim': [16, 32, 64],
                'kernel_size': [3, 5, 7],
                'num_layers': [1, 2, 3]
            }
            # 运行遗传算法找到最佳参数
            best_params = genetic_algorithm(X_train, y_train, X_test, y_test, param_space)
            print(f"遗传算法优化完成，最佳参数为: {best_params}")

            print("开始模型训练...")
            # 使用最佳参数训练模型
            best_model = train_model(X_train, y_train, best_params)
            print("模型训练完成，开始预测和评估...")
            
            # 进行预测并评估结果
            mse = predict_and_evaluate(best_model, X_test, y_test, scaler, scaler_close)
            print("预测和评估完成。")
            print("Mean Squared Error on Test Set:", mse)
        else:
            print("可用于测试的数据不足，请调整look_back值或数据获取日期范围。")
    else:
        print("无法获取股票数据。")
