# Kronos AI BTC 5分钟短线专业策略

基于Kronos金融时间序列基础模型的币安合约BTC专业交易系统，完全按照TREA平台专业策略标准实现。

## 📋 策略概述

- **标的**: BTC/USDT（永续合约）
- **周期**: 5分钟
- **杠杆**: 3倍（固定）
- **目标**: 每日1%-4%稳定收益，胜率≥55%，盈亏比≥1.8:1，最大回撤≤5%

## 🎯 核心特性

### Kronos AI信号解析
- 趋势方向识别（LONG/SHORT）
- 趋势强度计算（≥1.2%触发）
- 预测支撑位和阻力位

### 高级入场规则
- 做多：价格≥支撑位×1.001 + K线跌幅≤0.3% + 资金费率≤0.015%
- 做空：价格≤阻力位×0.999 + K线涨幅≤0.3% + 资金费率≥-0.015%

### 分批建仓与分级止盈
- 50%初始仓位 + 50%确认后补仓
- 止盈1：盈利1%（平50%仓位）
- 止盈2：盈利1.8%（平剩余50%仓位）
- 移动止损：止盈1后调整为保本止损

### 完整风控系统
- 单笔风险≤0.8%总资金
- 单日亏损≥3%停止交易
- 连续亏损2笔暂停30分钟
- 单日最大交易8笔
- 极端行情（5分钟涨/跌≥1.5%）过滤

### 交易时段控制
- 优先：20:00-次日02:00（BTC波动活跃时段）
- 其他时段信号过滤强度提升50%

## 📁 项目结构

```
kronos交易/
├── Kronos/                          # Kronos官方项目
├── strategy_config.py               # 专业策略配置
├── enhanced_kronos.py               # 增强版Kronos分析器
├── professional_strategy.py         # 专业交易策略主逻辑
├── professional_main.py             # 专业策略主程序入口
├── binance_api.py                   # 增强版币安API
├── .env                             # 环境配置
├── requirements.txt                 # 依赖包
└── PROFESSIONAL_README.md           # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

`.env` 文件已配置好你的币安API密钥。

### 3. 测试连接

```bash
python professional_main.py --mode test
```

### 4. 测试Kronos信号分析

```bash
python professional_main.py --mode signal
```

### 5. 执行一次策略分析

```bash
python professional_main.py --mode once
```

### 6. 连续运行交易策略

```bash
python professional_main.py --mode continuous
```

## ⚙️ 策略配置

所有策略参数在 `strategy_config.py` 中：

```python
class StrategyConfig:
    SYMBOL = "BTCUSDT"
    TIMEFRAME = "5m"
    LEVERAGE = 3
    
    TREND_STRENGTH_THRESHOLD = 0.012  # 1.2%
    LOOKBACK_PERIOD = 512
    PREDICTION_LENGTH = 120
```

## 📊 策略详解

### 入场条件（做多）
1. Kronos AI输出趋势方向 = LONG
2. 趋势强度 ≥ 0.012
3. 当前BTC价格 ≥ pred_support × 1.001
4. 最近1根5分钟K线跌幅 ≤ 0.3%
5. 资金费率 ≤ 0.015%

### 入场条件（做空）
1. Kronos AI输出趋势方向 = SHORT
2. 趋势强度 ≥ 0.012
3. 当前BTC价格 ≤ pred_resistance × 0.999
4. 最近1根5分钟K线涨幅 ≤ 0.3%
5. 资金费率 ≥ -0.015%

### 止盈止损（做多）
- 止损位：pred_support × 0.996
- 止盈1：入场价 × 1.01（平50%）
- 止盈2：入场价 × 1.018（平剩余50%）
- 移动止损：止盈1后调整为入场价 × 1.001

### 止盈止损（做空）
- 止损位：pred_resistance × 1.004
- 止盈1：入场价 × 0.99（平50%）
- 止盈2：入场价 × 0.982（平剩余50%）
- 移动止损：止盈1后调整为入场价 × 0.999

## ⚠️ 风险提示

1. **模拟盘先行**：首次使用建议先充分测试信号
2. **小额起步**：实盘初始资金≤总可投资金的20%
3. **严格风控**：本策略有多层风控保护，但市场有风险
4. **AI辅助**：Kronos是预测工具，不是保证盈利的神器
5. **自主决策**：所有交易决策请自行负责

## 📈 预期表现

- **胜率**: 55%-65%
- **盈亏比**: 1.8:1
- **每日收益**: 1%-4%
- **最大回撤**: ≤5%

---

**免责声明**: 本策略仅供学习和研究使用，不构成投资建议。加密货币交易存在高风险，请谨慎使用。
