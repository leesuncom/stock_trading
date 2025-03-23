import os
import pandas as pd
import numpy as np
import akshare as ak
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime

import pandas as pd


def get_trading_dates(start_date, end_date):
    # 假设 trade_cal 是从文件中加载的交易日期数据
    trade_cal = pd.read_csv('your_trading_calendar_file.csv')  # 替换为实际的文件路径

    # 打印列名，用于调试
    print("trade_cal 列名:", trade_cal.columns)

    try:
        # 尝试获取 'date' 列
        trade_dates = pd.to_datetime(trade_cal['date']).dt.date
    except KeyError:
        print("未找到 'date' 列，请检查列名。")
        return []

    # 过滤出在指定日期范围内的交易日期
    valid_dates = [date for date in trade_dates if start_date <= date <= end_date]
    return valid_dates





def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    计算股票的技术指标

    参数:
        data: DataFrame, 包含OHLCV数据的DataFrame
        start_date: str, 开始日期 (可选，用于相对表现计算)
        end_date: str, 结束日期 (可选，用于相对表现计算)

    返回:
        DataFrame: 添加了技术指标的数据
    """
    # 添加日期特征
    # 基础特征
    data = data.assign(
        Year = data.index.year,
        Month = data.index.month,
        Day = data.index.day,
        Day_of_Week = data.index.dayofweek
    )

    # 动态移动平均计算
    ma_windows = [5, 10, 20, 60]
    for w in ma_windows:
        data[f'MA{w}'] = data['收盘'].shift(1).rolling(w).mean()
        data[f'Std_{w}'] = data['收盘'].rolling(w).std()

    # 移动平均线
    data['MA5'] = data['收盘'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['收盘'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['收盘'].shift(1).rolling(window=20).mean()

    # RSI指标
    delta = data['收盘'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD指标
    exp1 = data['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = data['收盘'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    # VWAP指标
    data['Daily_Date'] = data.index.date
    grouped = data.groupby('Daily_Date')
    data['Cumulative_Volume'] = grouped['成交量'].cumsum()
    data['Cumulative_TPV'] = (data['收盘'] * data['成交量']).groupby(data['Daily_Date']).cumsum()
    data['VWAP'] = data['Cumulative_TPV'] / data['Cumulative_Volume']

    # 布林带
    period = 20
    data['SMA'] = data['收盘'].rolling(window=period).mean()
    data['Std_dev'] = data['收盘'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘表现
    if start_date and end_date:
        benchmark_data = ak.fund_etf_hist_em('510310', period="daily", start_date=start_date, end_date=end_date,
                                             adjust="qfq")
        benchmark_data['日期'] = pd.to_datetime(benchmark_data['日期'])
        benchmark_data = benchmark_data.set_index('日期')
        benchmark_close = benchmark_data['收盘'].reindex(data.index)
        data['Relative_Performance'] = (data['收盘'] / benchmark_close) * 100

    # ROC指标
    data['ROC'] = data['收盘'].pct_change(periods=1) * 100

    # ATR指标
    high_low_range = data['最高'] - data['最低']
    high_close_range = abs(data['最高'] - data['收盘'].shift(1))
    low_close_range = abs(data['最低'] - data['收盘'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['收盘', '开盘', '最高', '最低']].shift(1)

    # 计算对数收益率
    data['log_return'] = np.log(data['收盘'] / data['收盘'].shift(1))

    # 计算交易量与趋势的交互特征
    data['volume_trend_interaction'] = data['成交量'] * (data['收盘'] - data['收盘'].shift(1))

    # 计算波动率与趋势的交互特征
    data['volatility_trend_interaction'] = data['Std_dev'] * (data['收盘'] - data['收盘'].shift(1))

    # 添加月份和星期的周期性特征
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

    # 添加收盘价差分和滞后特征
    data['close_lag_1_diff'] = data['收盘'].diff(1)
    data['close_lag_1'] = data['收盘'].shift(1)
    data['close_lag_2'] = data['收盘'].shift(2)
    data['close_lag_3'] = data['收盘'].shift(3)
    data['close_lag_4'] = data['收盘'].shift(4)
    data['close_lag_5'] = data['收盘'].shift(5)

    # 计算趋势Z分数
    data['trend_z_5'] = (data['收盘'] - data['MA5']) / data['Std_dev']
    data['trend_z_20'] = (data['收盘'] - data['MA20']) / data['Std_dev']
    # 对于60日均线，可能需要先计算
    data['MA60'] = data['收盘'].rolling(window=60).mean()
    data['trend_z_60'] = (data['收盘'] - data['MA60']) / data['Std_dev']

    # 计算趋势比率
    data['trend_ratio_5'] = data['收盘'] / data['MA5']
    data['trend_ratio_20'] = data['收盘'] / data['MA20']
    data['trend_ratio_60'] = data['收盘'] / data['MA60']

    # 市场状态识别
    data['Market_State'] = np.select(
        [
            data['收盘'] > data['MA20'] * 1.05,
            data['收盘'] < data['MA20'] * 0.95,
            (data['收盘'] >= data['MA20'] * 0.97) & (data['收盘'] <= data['MA20'] * 1.03)
        ],
        [2, 0, 1],
        default=1
    )

    # 趋势强度指标
    data['Trend_Strength'] = data['MA5'].rolling(3).corr(data['MA20']) * 100

    # 删除缺失值
    data = data.dropna()

    return data


def get_trading_dates(start_date, end_date):
    """获取指定日期范围内的交易日历"""
    trade_cal = ak.tool_trade_date_hist_sina()
    trade_dates = pd.to_datetime(trade_cal['date']).dt.date
    return pd.date_range(start=start_date, end=end_date).date.tolist()


def get_stock_data(ticker, start_date, end_date, adjust="qfq", period="daily"):
    """
    获取并处理单个股票的数据

    参数:
        ticker: 股票代码
        start_date: 起始日期
        end_date: 结束日期
    返回:
        处理后的股票数据DataFrame
    """
    # 下载股票数据
    data = ak.stock_zh_a_hist(symbol=ticker, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
    # 打印数据以检查结构
    print(data.head())  # 打印前几行数据
    # 确保日期列存在并设置为索引
    if '日期' in data.columns:
        data['日期'] = pd.to_datetime(data['日期'])
        data = data.set_index('日期')
    else:
        raise ValueError("下载的数据中没有 '日期' 列")

    # 计算技术指标
    data = calculate_technical_indicators(data, start_date, end_date)

    return data


def clean_csv_files(file_path):
    df = pd.read_csv(file_path)

    # 删除第二行和第三行
    df = df.drop([0, 1]).reset_index(drop=True)

    # 重命名列
    df = df.rename(columns={'Price': 'Date'})

    # 保存修改后的文件
    df.to_csv(file_path, index=False)
    print("所有文件处理完成！")


def main():
    """主函数：执行数据收集和处理流程"""
    # 股票分类列表
    tickers = [
        '300059'  # 工业
    ]

    # 设置参数
    start_date = "20210601"
    end_date = datetime.now().strftime("%Y%m%d")
    NUM_FEATURES_TO_KEEP = 9

    # 创建数据文件夹
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)

    # 获取并保存所有股票数据
    print("开始下载和处理股票数据...")
    for ticker in tickers:
        try:
            print(f"处理 {ticker} 中...")
            stock_data = get_stock_data(ticker=ticker, start_date=start_date, end_date=end_date, period="daily",
                                        adjust="qfq", )
            # 不再重复设置日期列
            stock_data = calculate_technical_indicators(stock_data, start_date, end_date)
            stock_data.to_csv(f'{data_folder}/{ticker}.csv')
            clean_csv_files(f'{data_folder}/{ticker}.csv')
            print(f"{ticker} 处理完成")
        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")


if __name__ == "__main__":
    main()