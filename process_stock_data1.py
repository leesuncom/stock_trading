import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import logging
from datetime import timedelta
import akshare as ak
from visualization import (
    plot_stock_prediction,
    plot_training_loss,
    plot_cumulative_earnings,
    plot_accuracy_comparison
)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def get_trading_dates(start_date, end_date):
    """获取指定日期范围内的交易日历"""
    trade_cal = ak.tool_trade_date_hist_sina()
    trade_dates = pd.to_datetime(trade_cal['date']).dt.date
    return pd.date_range(start=start_date, end=end_date).date.tolist()


def update_features(last_features, new_price):
    """根据最新价格动态更新技术指标"""
    updated_features = last_features.copy()

    # 示例：更新MA5、MA10
    window = 5
    updated_features[0] = (last_features[0] * (window - 1) + new_price) / window  # MA5
    window = 10
    updated_features[1] = (last_features[1] * (window - 1) + new_price) / window  # MA10

    # 其他技术指标更新逻辑...

    return updated_features


def get_stock_data(ticker, data_dir='data'):
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    try:
        data = pd.read_csv(file_path, index_col='日期', parse_dates=True)
        logger.info(f"成功加载股票数据: {ticker}")
        return data
    except Exception as e:
        logger.error(f"加载股票数据失败: {ticker} - {str(e)}")
        raise


def format_feature(data):
    features = [
        '成交量', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    try:
        X = data[features].iloc[1:]
        y = data['收盘'].pct_change().iloc[1:]
        logger.info(f"成功格式化特征数据: {len(X)} samples")
        return X, y
    except KeyError as e:
        logger.error(f"特征列不存在: {str(e)}")
        raise


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

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'r2': r2,
        'mape': mape,
        'mse': mse,
        'sharpe_ratio': sharpe
    }
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)

    return metrics


def predict_future_days(model, data, X_scaled, scaler_y, n_steps, days_to_predict=5):
    model.eval()
    last_sequence = X_scaled[-n_steps:].reshape(1, n_steps, X_scaled.shape[1])
    last_price = data['收盘'].iloc[-1]

    future_prices = [last_price]
    future_dates = [data.index[-1]]

    # 获取交易日历
    last_date = pd.to_datetime(data.index[-1])
    end_date = last_date + timedelta(days=days_to_predict)
    trade_dates = get_trading_dates(last_date, end_date)

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

            # 生成下一个交易日日期
            while True:
                next_date_candidate = last_date + timedelta(days=i + 1)
                if next_date_candidate.date() in trade_dates:
                    future_dates.append(next_date_candidate)
                    break
                i += 1  # 跳过非交易日

            # 更新特征序列
            new_sequence = np.roll(current_sequence.cpu().numpy(), -1, axis=1)
            new_sequence[0, -1, :] = update_features(new_sequence[0, -2, :], next_price)
            current_sequence = torch.tensor(new_sequence, dtype=torch.float32).to(device)

    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates[1:]]
    return future_dates_str, future_prices[1:]


def train_and_predict_lstm(ticker, data, X, y, save_dir, n_steps=60, num_epochs=500, batch_size=32,
                           learning_rate=0.001):
    scaler_y = StandardScaler()
    scaler_X = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
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

    model = LSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    early_stopping = EarlyStopping(patience=50)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                with torch.cuda.amp.autocast():
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

            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{avg_val_loss:.4f}"
            })
            pbar.update(1)
            scheduler.step()

            # 早停检查
            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(save_dir, f'{ticker}_best_model.pth'))
        logger.info(f"Saved best model for {ticker}")

    plot_training_loss(ticker, train_losses, val_losses, save_dir)

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

    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}

    future_dates, future_prices = predict_future_days(model, data, X_scaled, scaler_y, n_steps)

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices
    })
    os.makedirs(os.path.join(save_dir, 'future_predictions'), exist_ok=True)
    future_df.to_csv(os.path.join(save_dir, 'future_predictions', f'{ticker}_future_predictions.csv'))
    logger.info(f"未来5天预测保存成功: {ticker}")

    return predict_result, test_indices, predictions, actual_percentages, future_df


def save_predictions_with_indices(ticker, test_indices, predictions, save_dir):
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)
    logger.info(f"预测结果保存成功: {ticker}")


def predict(ticker_name, stock_data, stock_features, save_dir, epochs=500, batch_size=32, learning_rate=0.001):
    all_predictions_lstm = {}
    prediction_metrics = {}

    logger.info(f"\n开始处理股票: {ticker_name}")
    data = stock_data
    X, y = stock_features

    try:
        predict_result, test_indices, predictions, actual_percentages, future_df = train_and_predict_lstm(
            ticker_name, data, X, y, save_dir, num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        all_predictions_lstm[ticker_name] = predict_result

        metrics = visualize_predictions(ticker_name, data, predict_result, test_indices, predictions,
                                        actual_percentages,
                                        save_dir)
        prediction_metrics[ticker_name] = metrics

        save_predictions_with_indices(ticker_name, test_indices, predictions, save_dir)

        metrics_df = pd.DataFrame(prediction_metrics).T
        metrics_df.to_csv(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_metrics.csv'))
        logger.info(f"预测指标保存成功: {ticker_name}")

        plot_accuracy_comparison(prediction_metrics, save_dir)
        logger.info(f"准确率对比图保存成功: {ticker_name}")

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

        with open(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_summary.txt'), 'w',
                  encoding='utf-8') as f:
            for key, value in summary.items():
                f.write(f'{key}: {value}\n')
        logger.info(f"预测汇总保存成功: {ticker_name}")

        print("\nPrediction Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        return metrics, future_df

    except Exception as e:
        logger.error(f"处理股票失败: {ticker_name} - {str(e)}")
        raise


if __name__ == "__main__":
    tickers = [
        '300059'  # 示例股票代码
    ]

    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'future_predictions'), exist_ok=True)

    for ticker_name in tickers:
        try:
            stock_data = get_stock_data(ticker_name)
            stock_features = format_feature(stock_data)
            predict(
                ticker_name=ticker_name,
                stock_data=stock_data,
                stock_features=stock_features,
                save_dir=save_dir
            )
        except Exception as e:
            logger.error(f"跳过股票: {ticker_name} - {str(e)}")
            continue