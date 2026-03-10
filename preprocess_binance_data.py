import pandas as pd
import numpy as np
import os


def calculate_technical_indicators(df):
    """计算所有27个技术指标"""
    df = df.copy()
    
    # 重命名列（统一列名）
    if 'volume' in df.columns:
        df['vol'] = df['volume']
    if 'amount' in df.columns:
        df['amt'] = df['amount']
    
    # 基础价格数据
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    amount = df["amt"].values
    
    # 1. 移动平均线
    df["MA5"] = pd.Series(close).rolling(window=5).mean().values
    df["MA10"] = pd.Series(close).rolling(window=10).mean().values
    df["MA20"] = pd.Series(close).rolling(window=20).mean().values
    
    # 2. 乖离率
    df["BIAS20"] = (close / df["MA20"] - 1) * 100
    
    # 3. ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values
    
    # 4. 振幅
    df["AMPLITUDE"] = (high - low) / close * 100
    
    # 5. 成交额移动平均
    df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
    df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values
    df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]
    
    # 6. RSI(14) 和 RSI(7)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain14 = gain.rolling(window=14).mean()
    avg_loss14 = loss.rolling(window=14).mean()
    rs14 = avg_gain14 / avg_loss14
    df["RSI14"] = (100 - (100 / (1 + rs14))).values
    
    avg_gain7 = gain.rolling(window=7).mean()
    avg_loss7 = loss.rolling(window=7).mean()
    rs7 = avg_gain7 / avg_loss7
    df["RSI7"] = (100 - (100 / (1 + rs7))).values
    
    # 7. MACD线 和 MACD柱
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    
    # 8. 价格斜率(5期) 和 (10期)
    df["PRICE_SLOPE5"] = (
        pd.Series(close)
        .rolling(window=5)
        .apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True,
        )
        .values
    )
    df["PRICE_SLOPE10"] = (
        pd.Series(close)
        .rolling(window=10)
        .apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True,
        )
        .values
    )
    
    # 9. 近期高低点
    df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
    df["LOW5"] = pd.Series(low).rolling(window=5).min().values
    df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
    df["LOW10"] = pd.Series(low).rolling(window=10).min().values
    
    # 10. 成交量突破
    df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int)
    df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int)
    
    # 选择需要的27个特征
    feature_list = [
        "open", "high", "low", "close", "vol", "amt", 
        "MA5", "MA10", "MA20",
        "BIAS20",
        "ATR14", "AMPLITUDE",
        "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",
        "RSI14", "RSI7",
        "MACD", "MACD_HIST",
        "PRICE_SLOPE5", "PRICE_SLOPE10",
        "HIGH5", "LOW5", "HIGH10", "LOW10",
        "VOL_BREAKOUT", "VOL_SHRINK"
    ]
    
    # 确保所有列都存在
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0
    
    # 保留时间列 + 特征列，并去除含 NaN 的行
    time_col = None
    if "timestamps" in df.columns:
        time_col = "timestamps"
    elif "datetime" in df.columns:
        time_col = "datetime"
    
    if time_col:
        df = df[[time_col] + feature_list]
    else:
        df = df[feature_list]
    
    df = df.dropna()
    
    return df


def preprocess_data(input_csv, output_csv):
    """处理币安数据并保存"""
    print(f"正在读取: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"原始数据列: {list(df.columns)}")
    print(f"原始数据行数: {len(df)}")
    
    # 计算技术指标
    df_processed = calculate_technical_indicators(df)
    
    print(f"处理后数据列: {list(df_processed.columns)}")
    print(f"处理后数据行数: {len(df_processed)}")
    print(f"处理后数据列数: {len(df_processed.columns)}")
    
    # 保存处理后的文件
    df_processed.to_csv(output_csv, index=False)
    print(f"已保存到: {output_csv}")
    
    return df_processed


if __name__ == "__main__":
    input_file = r"h:\kronos交易\training_data\BTCUSDT_5m.csv"
    output_file = r"h:\kronos交易\training_data\BTCUSDT_5m_with_indicators.csv"
    
    df = preprocess_data(input_file, output_file)
    print("\n处理完成！")
