#!/usr/bin/env python
"""回测引擎 - 与实盘交易流程完全一致的回测引擎"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 导入现有模块
try:
    from binance_api import BinanceAPI
    from professional_strategy import ProfessionalTradingStrategy
    from strategy_profiles import StrategyProfiles
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有依赖模块已正确安装")
    sys.exit(1)


class MockBinanceAPI:
    """模拟币安API，用于回测"""
    
    def __init__(self, historical_data, initial_capital=10000.0, commission_rate=0.001):
        self.historical_data = historical_data
        self.current_index = 0
        self.initial_capital = initial_capital
        self.available_balance = initial_capital
        self.position = 0.0  # 正数表示多仓，负数表示空仓
        self.entry_price = 0.0
        self.orders = []
        self.trade_history = []
        self.equity_curve = []
        self.commission_rate = commission_rate  # 手续费率
        
    def get_recent_klines(self, symbol, timeframe, lookback=100):
        """获取历史K线数据"""
        end_idx = min(self.current_index + 1, len(self.historical_data))
        start_idx = max(0, end_idx - lookback)
        return self.historical_data.iloc[start_idx:end_idx].copy()
    
    def get_current_price(self, symbol):
        """获取当前价格"""
        if self.current_index < len(self.historical_data):
            return self.historical_data['close'].iloc[self.current_index]
        return None
    
    def get_funding_rate(self, symbol):
        """获取资金费率（回测中返回0）"""
        return 0.0
    
    def get_current_position_info(self, symbol):
        """获取当前持仓信息"""
        if self.position > 0:
            return "LONG", abs(self.position)
        elif self.position < 0:
            return "SHORT", abs(self.position)
        return None, 0.0
    
    def get_position(self, symbol):
        """获取当前持仓（兼容旧接口）"""
        direction, size = self.get_current_position_info(symbol)
        return {
            'positionAmt': size if direction else 0,
            'positionSide': direction or 'BOTH'
        }
    
    def get_total_balance(self):
        """获取总余额（可用+持仓）"""
        return self.calculate_current_equity()
    
    def get_symbol_info(self, symbol):
        """获取交易对信息"""
        return {
            'filters': [
                {
                    'filterType': 'LOT_SIZE',
                    'stepSize': '0.001',
                    'minQty': '0.001',
                    'maxQty': '1000'
                },
                {
                    'filterType': 'PRICE_FILTER',
                    'tickSize': '0.01'
                },
                {
                    'filterType': 'MIN_NOTIONAL',
                    'minNotional': '100'
                }
            ]
        }
    
    def set_leverage(self, symbol, leverage):
        """设置杠杆（回测中不需要实际操作）"""
        pass
    
    def place_market_buy(self, symbol, quantity):
        """模拟市价买入（平空或开多）"""
        if self.current_index >= len(self.historical_data):
            return False
        
        current_price = self.historical_data['close'].iloc[self.current_index]
        
        # 考虑滑点
        slippage = 0.0005  # 0.05%滑点
        execute_price = current_price * (1 + slippage)
        
        # 如果有空仓，先平空
        if self.position < 0:
            close_size = min(quantity, abs(self.position))
            close_value = close_size * execute_price
            
            # 计算盈亏（空仓盈亏：入场价 - 现价）
            pnl = (self.entry_price - execute_price) * close_size
            
            # 计算平仓手续费
            close_commission = close_value * self.commission_rate
            
            # 更新资金（扣除手续费）
            self.available_balance += close_value + pnl - close_commission
            self.position += close_size  # 空仓是负数，加close_size（正数）减少空仓
            
            # 如果全部平仓，重置入场价
            if self.position == 0:
                self.entry_price = 0.0
            
            # 记录交易
            self._record_trade("CLOSE_SHORT", execute_price, close_size, pnl, close_commission)
            
            # 如果只平空不开多，直接返回
            if quantity <= close_size:
                return True
            
            # 剩余数量用于开多
            quantity = quantity - close_size
        
        # 开多仓
        if quantity > 0:
            position_value = quantity * execute_price
            
            # 计算开仓手续费
            open_commission = position_value * self.commission_rate
            
            # 检查资金是否足够（包括手续费）
            total_cost = position_value + open_commission
            if total_cost > self.available_balance:
                print(f"资金不足: 需要${total_cost:.2f}, 可用${self.available_balance:.2f}")
                return False
            
            # 更新状态（扣除手续费）
            self.position = quantity
            self.entry_price = execute_price
            self.available_balance -= total_cost
            
            # 记录交易
            self._record_trade("OPEN_LONG", execute_price, quantity, 0.0, open_commission)
        
        return True
    
    def place_market_sell(self, symbol, quantity):
        """模拟市价卖出（平多或开空）"""
        if self.current_index >= len(self.historical_data):
            return False
        
        current_price = self.historical_data['close'].iloc[self.current_index]
        
        # 考虑滑点
        slippage = 0.0005  # 0.05%滑点
        execute_price = current_price * (1 - slippage)
        
        # 如果有多仓，先平多
        if self.position > 0:
            close_size = min(quantity, self.position)
            close_value = close_size * execute_price
            
            # 计算盈亏
            pnl = (execute_price - self.entry_price) * close_size
            
            # 计算平仓手续费
            close_commission = close_value * self.commission_rate
            
            # 更新资金（扣除手续费）
            self.available_balance += close_value + pnl - close_commission
            self.position -= close_size
            
            # 如果全部平仓，重置入场价
            if self.position == 0:
                self.entry_price = 0.0
            
            # 记录交易
            self._record_trade("CLOSE_LONG", execute_price, close_size, pnl, close_commission)
            
            # 如果只平多不开空，直接返回
            if quantity <= close_size:
                return True
            
            # 剩余数量用于开空
            quantity = quantity - close_size
        
        # 开空仓
        if quantity > 0:
            position_value = quantity * execute_price
            
            # 计算开仓手续费
            open_commission = position_value * self.commission_rate
            
            # 检查资金是否足够（包括手续费）
            total_cost = position_value + open_commission
            if total_cost > self.available_balance:
                print(f"资金不足: 需要${total_cost:.2f}, 可用${self.available_balance:.2f}")
                return False
            
            # 更新状态（空仓用负数表示，扣除手续费）
            self.position = -quantity
            self.entry_price = execute_price
            self.available_balance -= total_cost
            
            # 记录交易
            self._record_trade("OPEN_SHORT", execute_price, quantity, 0.0, open_commission)
        
        return True
    
    def place_stop_loss_order(self, symbol, side, quantity, stop_price):
        """模拟止损订单（回测中不实际放置）"""
        self.orders.append({
            'type': 'STOP_LOSS',
            'side': side,
            'quantity': quantity,
            'stop_price': stop_price
        })
        return True
    
    def place_take_profit_order(self, symbol, side, quantity, take_profit_price):
        """模拟止盈订单（回测中不实际放置）"""
        self.orders.append({
            'type': 'TAKE_PROFIT',
            'side': side,
            'quantity': quantity,
            'take_profit_price': take_profit_price
        })
        return True
    
    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        self.orders = []
        return True
    
    def cancel_all_algo_orders(self, symbol):
        """取消所有算法订单"""
        return True
    
    def _record_trade(self, trade_type, price, quantity, pnl, commission=0.0):
        """记录交易"""
        timestamp = self.historical_data['timestamps'].iloc[self.current_index] if self.current_index < len(self.historical_data) else datetime.now()
        self.trade_history.append({
            'timestamp': timestamp,
            'type': trade_type,
            'price': price,
            'quantity': quantity,
            'pnl': pnl,
            'commission': commission,
            'available_balance': self.available_balance,
            'position': self.position
        })
    
    def get_wallet_balance(self):
        """获取钱包余额（用于策略初始化）"""
        return self.available_balance
    
    def calculate_current_equity(self):
        """计算当前权益"""
        if self.current_index >= len(self.historical_data):
            return self.available_balance
        
        current_price = self.historical_data['close'].iloc[self.current_index]
        position_value = 0.0
        
        if self.position > 0:
            # 多仓：当前价值 = 数量 × 现价
            position_value = self.position * current_price
        elif self.position < 0:
            # 空仓：当前价值 = 数量 × 入场价 + (入场价 - 现价) × 数量
            # 也就是：入场价值 + 浮盈
            position_value = abs(self.position) * self.entry_price + (self.entry_price - current_price) * abs(self.position)
        
        return self.available_balance + position_value
    
    def advance(self):
        """前进到下一个时间点"""
        if self.current_index < len(self.historical_data) - 1:
            # 记录权益
            equity = self.calculate_current_equity()
            timestamp = self.historical_data['timestamps'].iloc[self.current_index]
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'price': self.historical_data['close'].iloc[self.current_index],
                'position': self.position
            })
            
            self.current_index += 1
            return True
        return False


class BacktestEngine:
    """回测引擎 - 与实盘交易流程完全一致"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 initial_capital: float = 10000.0,
                 timeframe: str = "5m",
                 strategy_profile: str = "trend",
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005,
                 start_date: str = None,
                 end_date: str = None):
        """
        初始化回测引擎
        
        Args:
            symbol: 交易品种
            initial_capital: 初始资金
            timeframe: K线时间周期
            strategy_profile: 策略预设名称
            commission_rate: 交易手续费率
            slippage: 交易滑点
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.strategy_profile = strategy_profile
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 设置回测日期范围
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = pd.to_datetime("2024-01-01")
            
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        
        print(f"初始化回测引擎: {symbol}")
        print(f"策略预设: {strategy_profile}")
        print(f"回测周期: {self.start_date.date()} 到 {self.end_date.date()}")
        print(f"初始资金: ${initial_capital:.2f}")
        print(f"时间周期: {timeframe}")
        print(f"手续费率: {commission_rate*100:.2f}%, 滑点: {slippage*100:.2f}%")
        
        # 初始化组件
        self.historical_data = None
        self.mock_binance = None
        self.strategy = None
        self.performance_metrics = {}
    
    def load_historical_data(self, limit: int = 1000, csv_filepath: str = None) -> pd.DataFrame:
        """加载历史K线数据
        
        Args:
            limit: 加载的K线数量（如果从API获取）
            csv_filepath: CSV文件路径（如果从本地文件加载）
            
        Returns:
            历史K线数据DataFrame
        """
        if csv_filepath:
            return self._load_data_from_csv(csv_filepath)
        
        print(f"正在加载历史数据: {self.symbol} {self.timeframe} {limit}条...")
        
        try:
            # 从币安API获取历史数据
            binance = BinanceAPI()
            df = binance.get_recent_klines(
                self.symbol, 
                self.timeframe, 
                lookback=limit
            )
            
            if df is None or df.empty:
                print("  警告: 获取历史数据失败，生成模拟数据...")
                df = self._generate_mock_data(limit)
            else:
                print(f"  ✓ 加载历史数据成功: {len(df)}条记录")
                print(f"    时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
            
            self.historical_data = df
            return df
            
        except Exception as e:
            print(f"  加载历史数据失败: {e}")
            print("  生成模拟数据进行回测...")
            df = self._generate_mock_data(limit)
            self.historical_data = df
            return df
    
    def _load_data_from_csv(self, csv_filepath: str) -> pd.DataFrame:
        """从本地CSV文件加载历史数据
        
        Args:
            csv_filepath: CSV文件路径
            
        Returns:
            历史K线数据DataFrame
        """
        print(f"正在从CSV文件加载历史数据: {csv_filepath}")
        
        try:
            df = pd.read_csv(csv_filepath)
            print(f"  ✓ CSV文件读取成功: {len(df)}条记录")
            print(f"    原始列: {list(df.columns)}")
            
            # 确保有必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            
            # 处理时间列
            if 'timestamps' in df.columns:
                df['timestamps'] = pd.to_datetime(df['timestamps'])
            elif 'datetime' in df.columns:
                df['timestamps'] = pd.to_datetime(df['datetime'])
                df = df.rename(columns={'datetime': 'timestamps'})
            else:
                print("  警告: 未找到时间列，将使用索引生成时间戳")
                # 使用索引生成时间戳
                end_time = pd.to_datetime("now")
                freq_map = {"5m": "5T", "15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
                freq = freq_map.get(self.timeframe, "5T")
                df['timestamps'] = pd.date_range(end=end_time, periods=len(df), freq=freq)[::-1]
            
            # 确保有volume和amount列
            if 'volume' not in df.columns and 'vol' in df.columns:
                df['volume'] = df['vol']
            if 'amount' not in df.columns and 'amt' in df.columns:
                df['amount'] = df['amt']
            
            # 检查必要列是否都存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  警告: 缺失必要列: {missing_columns}")
                print(f"  生成模拟数据进行回测...")
                return self._generate_mock_data(1000)
            
            # 只保留必要的列（去除技术指标列，回测不需要）
            keep_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            df = df[keep_columns].copy()
            
            # 确保数据按时间升序排列
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # 去除NaN值
            df = df.dropna()
            
            print(f"  ✓ 数据处理完成: {len(df)}条有效记录")
            print(f"    时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
            print(f"    最终列: {list(df.columns)}")
            
            self.historical_data = df
            return df
            
        except Exception as e:
            print(f"  从CSV加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            print("  生成模拟数据进行回测...")
            df = self._generate_mock_data(1000)
            self.historical_data = df
            return df
    
    def _generate_mock_data(self, n_candles: int = 1000) -> pd.DataFrame:
        """生成模拟K线数据
        
        Args:
            n_candles: K线数量
            
        Returns:
            模拟K线数据DataFrame
        """
        print(f"  生成模拟数据: {n_candles}条{self.timeframe}K线")
        
        # 基础价格参数
        base_price = 50000.0
        volatility = 0.02
        
        # 生成时间序列
        end_time = datetime.now()
        freq_map = {
            "5m": "5T",
            "15m": "15T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }
        freq = freq_map.get(self.timeframe, "5T")
        
        timestamps = pd.date_range(
            end=end_time, 
            periods=n_candles, 
            freq=freq
        )[::-1]
        
        # 生成随机价格序列
        np.random.seed(42)
        returns = np.random.normal(0, volatility/np.sqrt(365), n_candles)
        cum_returns = np.exp(np.cumsum(returns))
        prices = base_price * cum_returns
        
        # 生成OHLC数据
        opens = prices * (1 + np.random.normal(0, 0.002, n_candles))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, n_candles)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, n_candles)))
        closes = prices
        
        # 生成成交量
        avg_volume = 1000
        volumes = avg_volume * (1 + np.random.normal(0, 0.5, n_candles))
        volumes = np.maximum(volumes, 10)
        
        # 创建DataFrame
        df = pd.DataFrame({
            "timestamps": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "amount": volumes,
            "volume": volumes
        })
        
        print(f"  ✓ 模拟数据生成完成: {len(df)}条记录")
        print(f"    价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def run_backtest(self, data: pd.DataFrame = None) -> dict:
        """运行回测
        
        Args:
            data: 回测数据 (如果为None则使用加载的历史数据)
            
        Returns:
            回测结果字典
        """
        if data is None:
            if self.historical_data is None:
                print("错误: 没有可用的历史数据")
                return {"error": "没有可用的历史数据"}
            data = self.historical_data
        
        print(f"\n=== 开始回测 ===")
        print(f"数据量: {len(data)}条K线")
        print(f"开始时间: {data['timestamps'].iloc[0]}")
        print(f"结束时间: {data['timestamps'].iloc[-1]}")
        
        # 初始化模拟币安API
        self.mock_binance = MockBinanceAPI(
            data, 
            self.initial_capital,
            commission_rate=self.commission_rate
        )
        
        # 创建策略实例（注入模拟币安API）
        self.strategy = ProfessionalTradingStrategy(
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_type=self.strategy_profile,
            binance=self.mock_binance,
            backtest_mode=True
        )
        
        # 回测主循环
        total_candles = len(data)
        for i in range(60, total_candles):
            # 更新进度
            if i % 100 == 0:
                progress = (i / total_candles) * 100
                print(f"  回测进度: {progress:.1f}% ({i}/{total_candles})", end="\r")
            
            # 设置模拟API的当前索引
            self.mock_binance.current_index = i
            
            # 设置策略的回测时间
            self.strategy.backtest_current_time = data['timestamps'].iloc[i]
            
            # 运行策略的一次迭代
            try:
                self.strategy.run_once()
            except Exception as e:
                print(f"\n  策略执行出错: {e}")
                import traceback
                traceback.print_exc()
            
            # 前进到下一个时间点
            self.mock_binance.advance()
        
        print(f"\n✓ 回测完成: {len(data)}条K线分析完毕")
        
        # 计算性能指标
        self._calculate_performance_metrics()
        
        # 生成回测报告
        report = self._generate_backtest_report()
        
        return report
    
    def _calculate_performance_metrics(self):
        """计算回测性能指标"""
        if not self.mock_binance or not self.mock_binance.equity_curve:
            return
        
        # 提取权益曲线
        equity_values = [e["equity"] for e in self.mock_binance.equity_curve]
        
        if len(equity_values) < 2:
            return
        
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # 基础指标
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (365 / len(equity_values)) if len(equity_values) > 0 else 0
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # 回撤分析
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # 交易统计
        trades = self.mock_binance.trade_history
        open_trades = [t for t in trades if t["type"] in ["OPEN_LONG", "OPEN_SHORT"]]
        close_trades = [t for t in trades if t["type"] in ["CLOSE_LONG", "CLOSE_SHORT"]]
        total_trades = len(close_trades)
        winning_trades = len([t for t in close_trades if t.get("pnl", 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏统计
        total_pnl = sum(t.get("pnl", 0) for t in close_trades)
        winning_pnls = [t.get("pnl", 0) for t in close_trades if t.get("pnl", 0) > 0]
        losing_pnls = [t.get("pnl", 0) for t in close_trades if t.get("pnl", 0) < 0]
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if (losing_trades > 0 and avg_loss != 0) else float('inf')
        
        self.performance_metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_equity": equity_values[-1] if equity_values else self.initial_capital,
            "initial_capital": self.initial_capital
        }
    
    def _generate_backtest_report(self) -> dict:
        """生成回测报告
        
        Returns:
            回测报告字典
        """
        report = {
            "summary": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy_profile": self.strategy_profile,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "initial_capital": self.initial_capital,
                "final_equity": self.performance_metrics.get("final_equity", self.initial_capital),
                "total_return_pct": self.performance_metrics.get("total_return", 0) * 100,
                "annual_return_pct": self.performance_metrics.get("annual_return", 0) * 100,
                "max_drawdown_pct": self.performance_metrics.get("max_drawdown", 0) * 100,
                "sharpe_ratio": self.performance_metrics.get("sharpe_ratio", 0),
                "total_trades": self.performance_metrics.get("total_trades", 0),
                "win_rate_pct": self.performance_metrics.get("win_rate", 0) * 100,
            },
            "performance_metrics": self.performance_metrics,
            "trade_history": self.mock_binance.trade_history[-20:] if self.mock_binance else [],
            "equity_curve_sample": self.mock_binance.equity_curve[::max(1, len(self.mock_binance.equity_curve)//100)] if self.mock_binance else [],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def print_report(self, report: dict = None):
        """打印回测报告
        
        Args:
            report: 回测报告
        """
        if report is None:
            if not self.performance_metrics:
                print("错误: 没有可用的回测报告")
                return
            report = self._generate_backtest_report()
        
        summary = report.get("summary", {})
        metrics = report.get("performance_metrics", {})
        
        print("\n" + "="*80)
        print("回测报告摘要")
        print("="*80)
        print(f"交易品种: {summary.get('symbol', 'N/A')}")
        print(f"策略预设: {summary.get('strategy_profile', 'N/A')}")
        print(f"时间周期: {summary.get('timeframe', 'N/A')}")
        print(f"回测周期: {summary.get('start_date', 'N/A')} 到 {summary.get('end_date', 'N/A')}")
        print(f"初始资金: ${summary.get('initial_capital', 0):.2f}")
        print(f"最终权益: ${summary.get('final_equity', 0):.2f}")
        print(f"总收益率: {summary.get('total_return_pct', 0):.2f}%")
        print(f"年化收益率: {summary.get('annual_return_pct', 0):.2f}%")
        print(f"最大回撤: {summary.get('max_drawdown_pct', 0):.2f}%")
        print(f"夏普比率: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"总交易次数: {summary.get('total_trades', 0)}")
        print(f"胜率: {summary.get('win_rate_pct', 0):.2f}%")
        print("\n" + "-"*80)
        print("详细指标:")
        print(f"  波动率: {metrics.get('volatility', 0):.4f}")
        print(f"  总盈亏: ${metrics.get('total_pnl', 0):.2f}")
        print(f"  平均盈利: ${metrics.get('avg_win', 0):.2f}")
        print(f"  平均亏损: ${metrics.get('avg_loss', 0):.2f}")
        print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
        print(f"  盈利交易: {metrics.get('winning_trades', 0)}")
        print(f"  亏损交易: {metrics.get('losing_trades', 0)}")
        print("="*80)
    
    def export_report(self, report: dict = None, filepath: str = None):
        """导出回测报告到文件
        
        Args:
            report: 回测报告
            filepath: 文件路径
        """
        if report is None:
            report = self._generate_backtest_report()
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtest_report_{self.symbol}_{timestamp}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"✓ 回测报告已导出到: {filepath}")
        except Exception as e:
            print(f"✗ 导出报告失败: {e}")


# 使用示例
if __name__ == "__main__":
    print("=== 回测引擎测试 ===")
    
    # 创建回测引擎
    engine = BacktestEngine(
        symbol="BTCUSDT",
        initial_capital=10000.0,
        timeframe="5m",
        strategy_profile="trend",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # 加载历史数据
    data = engine.load_historical_data(limit=500)
    
    # 运行回测
    report = engine.run_backtest(data=data)
    
    # 打印报告
    engine.print_report(report)
    
    # 导出报告
    engine.export_report(report)
    
    print("\n回测测试完成！")
