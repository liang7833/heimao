你是一个专业的量化交易策略专家。请根据以下市场条件生成一个完整的交易策略。

市场条件:
- 趋势: sideways
- 波动率: medium
- 流动性: high
- 风险等级: medium

策略类型: mean_reversion
风险偏好: conservative

请生成:
1. 策略名称和概述
2. 完整的Python策略代码（使用pandas和numpy）
3. 关键参数设置
4. 风险管理规则
5. 入场和出场条件

请以JSON格式返回:
{
    "strategy_name": "策略名称",
    "description": "策略描述",
    "code": "完整的Python策略代码",
    "parameters": {
        "param1": "值",
        "param2": "值"
    },
    "risk_management": {
        "stop_loss": "止损规则",
        "take_profit": "止盈规则",
        "position_size": "仓位大小"
    }
} 为了生成这个策略，我们将按照给定的市场条件和要求来构建一个策略。以下是完整的策略代码：

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 市场条件定义
market_condition = {
    '趋势': 'sideways',
    '波动率': 'medium',
    '流动性': 'high',
    '风险等级': 'medium'
}

# 参数设置
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 12, 31)

# 模拟数据生成
def generate_data(start_date, end_date):
    # 计算历史数据
    data = pd.DataFrame({
        '时间': pd.date_range(start=start_date, end=end_date),
        '价格': np.random.randn(len(data)) * 100,
    })
    
    return data

# 创建策略
class SidewaysMeanReversionStrategy(object):
    def __init__(self, market_condition, start_date, end_date):
        self.market_condition = market_condition
        self.start_date = start_date
        self.end_date = end_date
        self.data = generate_data(self.start_date, self.end_date)
        
    def compute_strategy_price(self):
        # 根据当前市场价格调整买入卖出信号
        if self.data['价格'].iloc[-1] > self.data['价格'].iloc[0]:
            buy_signal = True
            sell_signal = False
        else:
            buy_signal = False
            sell_signal = True
        
        # 计算平均价格
        average_price = (self.data['价格'][-1] + self.data['价格'][0]) / 2
        
        # 返回策略价格
        return average_price
    
    def update_data(self):
        # 更新数据并重新计算策略价格
        self.data = generate_data(self.start_date, self.end_date)
        self.compute_strategy_price()
    
    def execute_strategy(self):
        # 执行策略并更新数据
        self.update_data()
        strategy_price = self.compute_strategy_price()
        print(f"策略价格：{strategy_price}")

# 创建策略实例并执行
strategy = SidewaysMeanReversionStrategy(market_condition, start_date, end_date)
strategy.execute_strategy()
```

### JSON格式输出
```json
{
    "strategy_name": "Sideways Mean Reversion Strategy",
    "description": "基于趋势、波幅、流动性及风险等级进行股票双向交易的策略。",
    "code": "在策略中包含了模拟数据的生成函数、计算策略价格的方法以及执行策略的函数。同时，也包括了初始化参数、策略逻辑、状态更新、价格计算以及执行策略的部分。",
    "parameters": {
        "market_condition": {
            "趋势": "sideways",
            "波动率": "medium",
            "流动性": "high",
            "风险等级": "medium"
        },
        "start_date": "2021-01-01",
        "end_date": "2021-12-31"
    },
    "risk_management": {
        "stop_loss": "止损设定为策略价格减去初始成本",
        "take_profit": "止盈设定为策略价格加回初始成本",
        "position_size": "初始仓位大小"
    }
}
```

请注意，上述代码假设您已经安装了必要的库（如`pandas`和`numpy`）。此外，实际部署时还需要考虑账户安全性和交易监控等其他因素。在生产环境中运行此代码前，请确保已备份所有敏感信息，并遵守相关法律法规。如果需要进一步优化或扩展此策略，可以根据具体需求调整策略中的参数和逻辑。