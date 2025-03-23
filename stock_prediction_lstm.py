import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
from visualization import (
    plot_stock_prediction,
    plot_training_loss,
    plot_cumulative_earnings,
    plot_accuracy_comparison
)


warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 触发早停
        return False


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def get_stock_data(ticker, data_dir='data'):
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    try:
        data = pd.read_csv(file_path, index_col='日期', parse_dates=True)
        return data
    except Exception as e:
        print(f"加载股票数据失败: {ticker} - {str(e)}")
        raise


def format_feature(data):
    """
    格式化特征数据，准备用于LSTM模型
    
    Args:
        data: 包含所有特征的DataFrame
    
    Returns:
        X: 特征矩阵
        y: 目标变量（价格变化百分比）
    """
    # 确保所有需要的特征都存在
    features = [
        '成交量', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes', 'log_return', 'volume_trend_interaction',
        'volatility_trend_interaction', 'month_cos', 'month_sin', 'day_of_week_sin', 'day_of_week_cos',
        'close_lag_1_diff', 'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5', 'trend_z_5',
        'trend_z_20', 'trend_z_60', 'trend_ratio_5', 'trend_ratio_20', 'trend_ratio_60'
    ]
    
    # 检查是否所有特征都存在
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"缺少以下特征: {missing_features}")
    
    # 提取特征和目标变量
    X = data[features]
    
    # 计算价格变化百分比作为目标变量
    y = (data['收盘'] - data['收盘'].shift(1)) / data['收盘'].shift(1)
    y = y.dropna()
    
    # 确保X和y有相同的索引
    X = X.loc[y.index]
    
    return X, y


def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages, save_dir):
    actual_prices = data['收盘'].loc[test_indices].values
    predicted_prices = np.array(predictions)

    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    sharpe = np.sqrt(252) * (np.mean(predictions) / np.std(predictions))  # 示例夏普比率

    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy, 'r2': r2, 'mape': mape, 'mse': mse,
               'sharpe_ratio': sharpe}
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)

    return metrics


def predict_future_days(model, data, X_scaled, scaler_y, n_steps, days_to_predict=5):
    """
    预测未来几天的股票价格

    Args:
        model: 训练好的LSTM模型
        data: 原始股票数据
        X_scaled: 归一化后的特征数据
        scaler_y: 用于反归一化预测结果的缩放器
        n_steps: 时间步长
        days_to_predict: 要预测的天数

    Returns:
        future_dates: 未来日期列表
        future_prices: 预测的未来价格列表
    """
    model.eval()
    last_sequence = X_scaled[-n_steps:].reshape(1, n_steps, X_scaled.shape[1])
    last_price = data['收盘'].iloc[-1]

    future_prices = [last_price]
    future_dates = [data.index[-1]]

    # 获取最后一个交易日的日期
    last_date = pd.to_datetime(data.index[-1])
    
    # 简单地生成未来日期，每次加1天
    # 注意：这里没有考虑周末和节假日，实际应用中可能需要更复杂的逻辑
    future_dates_list = [last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]

    with torch.no_grad():
        current_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)

        for i in range(days_to_predict):
            # 预测下一天的价格变化百分比
            pred = model(current_sequence)
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            pred_percentage = scaler_y.inverse_transform(pred_np)[0][0]

            # 计算下一天的价格
            next_price = future_prices[-1] * (1 + pred_percentage)
            future_prices.append(next_price)

            # 更新序列用于下一次预测
            # 注意：在实际应用中，我们需要更新X的所有特征，这里简化处理
            # 实际应用中应该基于新预测的价格重新计算所有技术指标
            new_sequence = np.roll(current_sequence.cpu().numpy(), -1, axis=1)
            # 这里简化处理，实际应用中需要计算新的特征值
            new_sequence[0, -1, :] = new_sequence[0, -2, :]
            current_sequence = torch.tensor(new_sequence, dtype=torch.float32).to(device)

    # 返回未来日期和预测价格（不包括当前价格）
    # 将日期转换为字符串格式，以确保在传递过程中不会丢失日期信息
    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates_list]
    return future_dates_str, future_prices[1:]


def train_and_predict_lstm(ticker, data, X, y, save_dir, n_steps=60, num_epochs=500, batch_size=32,
                           learning_rate=0.001):
    # 数据归一化和准备部分
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps - 1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_val = X_train[split_index - n_steps + 1:]
    y_val = y_train[split_index - n_steps + 1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # PyTorch数据准备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 修改这里，去掉 output_size 参数
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses = []
    val_losses = []

    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # 训练和验证循环
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            scheduler.step()

    # 使用可视化工具绘制损失曲线
    plot_training_loss(ticker, train_losses, val_losses, save_dir)

    # 预测
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]),
                                   dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['收盘'].iloc[i - 2])
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # 使用可视化工具绘制累积收益率曲线
    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}

    # 预测未来5天价格
    future_dates, future_prices = predict_future_days(model, data, X_scaled, scaler_y, n_steps)

    # 保存未来预测结果
    future_df = pd.DataFrame({
        'Date': future_dates,  # 现在future_dates已经是字符串格式
        'Predicted_Price': future_prices
    })
    os.makedirs(os.path.join(save_dir, 'future_predictions'), exist_ok=True)
    future_df.to_csv(os.path.join(save_dir, 'future_predictions', f'{ticker}_future_predictions.csv'))
    print(f"\n未来5天 {ticker} 的预测价格:")
    for date, price in zip(future_dates, future_prices):
        print(f"{date}: {price:.2f}")  # 不需要再调用strftime

    return predict_result, test_indices, predictions, actual_percentages, future_df  # 修改这里，添加future_df作为返回值


def save_predictions_with_indices(ticker, test_indices, predictions, save_dir):
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')


def predict(ticker_name, stock_data, stock_features, save_dir, epochs=500, batch_size=32, learning_rate=0.001):
    all_predictions_lstm = {}
    prediction_metrics = {}

    print(f"\nProcessing {ticker_name}")
    data = stock_data
    X, y = stock_features

    predict_result, test_indices, predictions, actual_percentages, future_df = train_and_predict_lstm(
        ticker_name, data, X, y, save_dir, num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    all_predictions_lstm[ticker_name] = predict_result

    metrics = visualize_predictions(ticker_name, data, predict_result, test_indices, predictions, actual_percentages,
                                    save_dir)
    prediction_metrics[ticker_name] = metrics

    save_predictions_with_indices(ticker_name, test_indices, predictions, save_dir)

    # 保存预测指标
    os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
    metrics_df = pd.DataFrame(prediction_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_metrics.csv'))
    print("\nPrediction metrics summary:")
    print(metrics_df.describe())

    # 使用可视化工具绘制准确度对比图
    plot_accuracy_comparison(prediction_metrics, save_dir)

    # 生成汇总报告
    summary = {
        'Average Accuracy': np.mean([m['accuracy'] * 100 for m in prediction_metrics.values()]),
        'Best Stock': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Worst Stock': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Average RMSE': metrics_df['rmse'].mean(),
        'Average MAE': metrics_df['mae'].mean(),
        'Average R²': metrics_df['r2'].mean(),
        'Average MAPE': metrics_df['mape'].mean(),
        'Average MSE': metrics_df['mse'].mean(),
        'Average Sharpe Ratio': metrics_df['sharpe_ratio'].mean()
    }

    # 保存汇总报告，指定编码为 utf-8
    with open(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_summary.txt'), 'w', encoding='utf-8') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

    print("\nPrediction Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return metrics, future_df


if __name__ == "__main__":
    tickers = [
        '300059'  # 工业
    ]

    save_dir = 'results'  # 设置保存目录
    for ticker_name in tickers:
        stock_data = get_stock_data(ticker_name)
        # 生成增强特征

        stock_features = format_feature(stock_data)
        predict(
            ticker_name=ticker_name,
            stock_data=stock_data,
            stock_features=stock_features,
            save_dir=save_dir
        )

