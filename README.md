# Kronos 币安合约BTC交易机器人

基于Kronos金融时间序列基础模型的币安合约BTC自动交易系统。

## 项目结构

```
kronos交易/
├── Kronos/                      # Kronos官方项目
├── binance_api.py               # 币安API集成模块
├── kronos_analyzer.py           # Kronos分析模块
├── trading_strategy.py          # 交易策略模块
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包
├── .env                         # 配置文件
└── README.md                    # 说明文档
```

## 功能特点

- 集成Kronos金融时间序列基础模型进行价格预测
- 币安合约API集成，支持BTCUSDT合约交易
- 自动交易策略，基于Kronos预测信号
- 支持测试网络和主网络
- 实时价格预测和图表可视化

## 安装步骤

1. 确保已安装Python 3.10+

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 配置说明

在 `.env` 文件中配置你的参数：

```
BINANCE_API_KEY=你的API密钥
BINANCE_API_SECRET=你的API密钥密码
BINANCE_TESTNET=false           # 是否使用测试网
SYMBOL=BTCUSDT                   # 交易对
TIMEFRAME=1m                     # K线时间周期
LEVERAGE=10                      # 杠杆倍数
THRESHOLD=0.5                    # 交易信号阈值(%)
PRED_LEN=120                     # 预测长度
LOOKBACK=512                     # 回顾窗口
RISK_PER_TRADE=0.02              # 每次交易风险比例
CHECK_INTERVAL=300               # 检查间隔(秒)
```

## 使用方法

### 1. 测试币安API连接
```bash
python main.py --mode test
```

### 2. 测试Kronos预测功能
```bash
python main.py --mode predict
```

### 3. 执行一次交易分析
```bash
python main.py --mode once
```

### 4. 连续运行交易机器人
```bash
python main.py --mode continuous
```

## 模块说明

### binance_api.py
币安API封装类，提供：
- K线数据获取
- 账户余额查询
- 持仓管理
- 下单交易
- 杠杆设置

### kronos_analyzer.py
Kronos模型分析类，提供：
- 价格预测
- 趋势分析
- 交易信号生成
- 预测图表可视化

### trading_strategy.py
交易策略类，提供：
- 仓位管理
- 风险控制
- 自动开平仓
- 连续交易循环

## 风险提示

⚠️ **重要提示**：
1. 这是一个演示项目，不构成投资建议
2. 加密货币交易存在高风险，请谨慎使用
3. 建议先在测试网络充分测试
4. 使用前请充分理解Kronos模型和交易策略
5. 请合理控制仓位和风险

## Kronos模型

Kronos是清华大学开源的金融时间序列基础模型，专门用于处理K线数据。它在超过45个全球交易所的数据上进行了预训练，具有出色的预测能力。

更多信息请访问：https://github.com/shiyu-coder/Kronos
