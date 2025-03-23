import gradio as gr
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import warnings
from stock_prediction_lstm import predict, format_feature
from RLagent import process_stock
from datetime import datetime
from process_stock_data import get_stock_data, clean_csv_files


# 添加enhanced_feature_engineering函数
def enhanced_feature_engineering(data):
    """
    增强特征工程，创建更多预测所需的特征

    Args:
        data: 包含基本特征的DataFrame

    Returns:
        增强后的DataFrame，包含所有需要的特征
    """
    # 确保索引是日期类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

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

    # 删除NaN值
    data = data.dropna()

    return data


# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'tmp/gradio'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('tmp/gradio/pic', exist_ok=True)
os.makedirs('tmp/gradio/ticker', exist_ok=True)


def get_data(ticker, start_date, end_date, period="daily", progress=gr.Progress()):
    data_folder = 'tmp/gradio/ticker'
    temp_path = f'{data_folder}/{ticker}.csv'
    try:
        # 获取并保存所有股票数据
        progress(0, desc="开始获取股票数据...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        progress(0.4, desc="计算技术指标...")
        stock_data.to_csv(temp_path)
        progress(0.7, desc="处理数据格式...")
        clean_csv_files(temp_path)
        progress(1.0, desc="数据获取完成")
        return temp_path, "数据获取成功"
    except Exception as e:
        return None, f"获取数据出错: {str(e)}"


def process_and_predict(temp_csv_path, epochs, batch_size, learning_rate,
                        window_size, initial_money, agent_iterations, save_dir):
    if not temp_csv_path:
        return [None] * 14  # 更新为14个None，因为我们现在有14个输出

    try:
        ticker = os.path.basename(temp_csv_path).split('.')[0]
        stock_data = pd.read_csv(temp_csv_path, index_col='日期', parse_dates=True)

        # 应用增强特征工程
        stock_data = enhanced_feature_engineering(stock_data)

        # 现在再调用format_feature
        stock_features = format_feature(stock_data)

        metrics, future_df = predict(
            save_dir=save_dir,
            ticker_name=ticker,
            stock_data=stock_data,
            stock_features=stock_features,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # 确保未来日期格式正确
        if isinstance(future_df, pd.DataFrame):
            # 如果日期显示为1970年，说明日期可能是时间戳或格式不正确
            # 我们可以生成新的日期序列，从当前日期开始
            today = datetime.now()
            future_dates = [(today + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(len(future_df))]

            # 替换DataFrame中的日期列
            if 'Date' in future_df.columns:
                future_df['Date'] = future_dates

            # 对价格列进行四舍五入
            if 'Predicted_Price' in future_df.columns:
                future_df['Predicted_Price'] = future_df['Predicted_Price'].round(2)

        trading_results = process_stock(
            ticker,
            save_dir,
            window_size=window_size,
            initial_money=initial_money,
            iterations=agent_iterations
        )

        prediction_plot = Image.open(f"{save_dir}/pic/predictions/{ticker}_prediction.png")
        loss_plot = Image.open(f"{save_dir}/pic/loss/{ticker}_loss.png")
        earnings_plot = Image.open(f"{save_dir}/pic/earnings/{ticker}_cumulative.png")
        trades_plot = Image.open(f"{save_dir}/pic/trades/{ticker}_trades.png")
        transactions_df = pd.read_csv(f"{save_dir}/transactions/{ticker}_transactions.csv")

        # 对交易记录中数值类型的列进行四舍五入处理
        numerical_columns = transactions_df.select_dtypes(include=['number']).columns
        transactions_df[numerical_columns] = transactions_df[numerical_columns].round(2)

        # 对交易指标进行四舍五入处理
        total_gains = round(trading_results['total_gains'], 2)
        investment_return = round(trading_results['investment_return'], 2)

        # 添加夏普比率计算，如果metrics中没有提供
        sharpe_ratio = round(metrics.get('sharpe_ratio', 0), 2)

        return [
            [prediction_plot, loss_plot, earnings_plot, trades_plot],
            round(metrics['accuracy'] * 100, 2),
            round(metrics['rmse'], 2),
            round(metrics['mae'], 2),
            round(metrics['mse'], 2),
            round(metrics['mape'], 2),
            round(metrics['r2'], 2),
            sharpe_ratio,  # 添加夏普比率到返回值列表中
            total_gains,
            investment_return,
            trading_results['trades_buy'],
            trading_results['trades_sell'],
            transactions_df,
            future_df
        ]
    except Exception as e:
        print(f"处理错误: {str(e)}")
        return [None] * 14  # 更新为14个None，因为我们现在有14个输出


with gr.Blocks() as demo:
    gr.Markdown("# 智能股票预测与交易")

    save_dir_state = gr.State(value='tmp/gradio')
    temp_csv_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(label="股票代码 (例如: 300059)")
        with gr.Column(scale=2):
            start_date = gr.Textbox(
                label="开始日期 (YYYYMMDD)",
                value=(datetime.now().replace(year=datetime.now().year - 4).strftime('%Y%m%d'))
            )
        with gr.Column(scale=2):
            end_date = gr.Textbox(
                label="结束日期 (YYYYMMDD)",
                value=datetime.now().strftime('%Y%m%d')
            )
        with gr.Column(scale=1):
            fetch_button = gr.Button("获取数据")

    with gr.Row():
        status_output = gr.Textbox(label="状态信息", interactive=False)

    with gr.Row():
        data_file = gr.File(label="下载股票数据", visible=True, interactive=False)

    with gr.Tabs():
        with gr.TabItem("LSTM预测参数"):
            with gr.Column():
                lstm_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10,
                                        label="LSTM训练轮数")
                lstm_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16,
                                       label="LSTM批次大小")
                learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001,
                                          step=0.0001, label="LSTM训练学习率")

        with gr.TabItem("交易代理参数"):
            with gr.Column():
                window_size = gr.Slider(minimum=10, maximum=100, value=30, step=5,
                                        label="时间窗口大小")
                initial_money = gr.Number(value=10000, label="初始投资金额 ($)")
                agent_iterations = gr.Slider(minimum=100, maximum=1000, value=500,
                                             step=50, label="代理训练迭代次数")

    with gr.Row():
        train_button = gr.Button("开始训练", interactive=False)

    with gr.Row():
        output_gallery = gr.Gallery(label="分析结果可视化", show_label=True,
                                    elem_id="gallery", columns=4, rows=1,
                                    height="auto", object_fit="contain")

    with gr.Row():
        gr.Markdown("### 未来5天预测价格")
        future_prices_df = gr.DataFrame(
            headers=["Date", "Predicted_Price"],
            label="未来5天预测价格"
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 预测指标")
            accuracy_output = gr.Number(label="预测准确率 (%)")
            rmse_output = gr.Number(label="RMSE (均方根误差)")
            mae_output = gr.Number(label="MAE (平均绝对误差)")
            mse_output = gr.Number(label="MSE (均方误差)")
            mape_output = gr.Number(label="MAPE (平均绝对百分比误差)")
            r2_output = gr.Number(label="R² (决定系数)")
            sharpe_ratio = gr.Number(label="Sharpe Ratio (夏普比率)")

        with gr.Column(scale=1):
            gr.Markdown("### 交易指标")
            gains_output = gr.Number(label="总收益 ($)")
            return_output = gr.Number(label="投资回报率 (%)")
            trades_buy_output = gr.Number(label="买入次数")
            trades_sell_output = gr.Number(label="卖出次数")

    with gr.Row():
        gr.Markdown("### 交易记录")
        transactions_df = gr.DataFrame(
            headers=["天", "操作", "价格", "利润", "资金总额"],
            label="交易详细记录"
        )


    def update_interface(csv_path):
        return (
            csv_path if csv_path else None,  # 更新文件下载
            gr.update(interactive=bool(csv_path))  # 更新训练按钮
        )


    # 获取数据按钮事件
    fetch_result = fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date, end_date],
        outputs=[temp_csv_state, status_output]
    )

    # 更新界面状态
    fetch_result.then(
        update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file, train_button]
    )

    # 训练按钮事件
    train_button.click(
        fn=process_and_predict,
        inputs=[
            temp_csv_state,
            lstm_epochs,
            lstm_batch,
            learning_rate,
            window_size,
            initial_money,
            agent_iterations,
            save_dir_state
        ],
        outputs=[
            output_gallery,
            accuracy_output,
            rmse_output,
            mae_output,
            mse_output,
            mape_output,
            r2_output,
            sharpe_ratio,  # 确保这个输出与函数返回值对应
            gains_output,
            return_output,
            trades_buy_output,
            trades_sell_output,
            transactions_df,
            future_prices_df
        ]
    )

demo.launch(server_port=7860, share=True)

