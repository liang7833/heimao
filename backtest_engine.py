#!/usr/bin/env python
"""回测引擎 - 集成策略协调器，进行历史数据回测和实盘交易"""

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
    from strategy_coordinator import StrategyCoordinator
    from enhanced_kronos import EnhancedKronosAnalyzer
    from fingpt_analyzer import FinGPTSentimentAnalyzer
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有依赖模块已正确安装")

class BacktestEngine:
    """回测引擎 - 加密货币交易策略回测"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 initial_capital: float = 10000.0,
                 timeframe: str = "5m",
                 use_fingpt: bool = True,
                 commission_rate: float = 0.001,  # 默认手续费率0.1%
                 slippage: float = 0.0005,        # 默认滑点0.05%
                 start_date: str = None,
                 end_date: str = None):
        """
        初始化回测引擎
        
        Args:
            symbol: 交易品种 (BTCUSDT, ETHUSDT等)
            initial_capital: 初始资金 (USDT)
            timeframe: K线时间周期
            use_fingpt: 是否使用FinGPT舆情分析
            commission_rate: 交易手续费率
            slippage: 交易滑点
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.use_fingpt = use_fingpt
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 设置回测日期范围
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = pd.to_datetime("2024-01-01")  # 默认开始日期
            
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))  # 默认结束日期
        
        print(f"初始化回测引擎: {symbol}")
        print(f"回测周期: {self.start_date.date()} 到 {self.end_date.date()}")
        print(f"初始资金: ${initial_capital:.2f}")
        print(f"时间周期: {timeframe}")
        print(f"手续费率: {commission_rate*100:.2f}%, 滑点: {slippage*100:.2f}%")
        
        # 初始化策略协调器
        self.strategy_coordinator = None
        self.binance_api = None
        self.historical_data = None
        
        # 回测状态变量
        self.capital = initial_capital
        self.position = 0.0  # 当前持仓数量
        self.position_entry_price = 0.0  # 持仓入场价格
        self.trade_history = []  # 交易历史记录
        self.equity_curve = []  # 权益曲线
        self.signals = []  # 信号记录
        
        # 性能指标
        self.performance_metrics = {}
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化回测所需组件"""
        try:
            print("  初始化策略协调器...")
            self.strategy_coordinator = StrategyCoordinator(
                kronos_model_name="kronos-small",
                use_fingpt=self.use_fingpt,
                symbol=self.symbol.replace("USDT", "")  # 移除USDT后缀
            )
            
            print("  初始化币安API...")
            self.binance_api = BinanceAPI()  # 回测使用默认API配置
            
            print("  ✓ 回测引擎初始化完成")
        except Exception as e:
            print(f"  ✗ 组件初始化失败: {e}")
            raise
    
    def load_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """加载历史K线数据
        
        Args:
            limit: 加载的K线数量
            
        Returns:
            历史K线数据DataFrame
        """
        print(f"正在加载历史数据: {self.symbol} {self.timeframe} {limit}条...")
        
        try:
            # 从币安API获取历史数据
            df = self.binance_api.get_recent_klines(
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
    
    def _generate_mock_data(self, n_candles: int = 1000) -> pd.DataFrame:
        """生成模拟K线数据（当无法获取真实数据时使用）
        
        Args:
            n_candles: K线数量
            
        Returns:
            模拟K线数据DataFrame
        """
        print(f"  生成模拟数据: {n_candles}条{self.timeframe}K线")
        
        # 基础价格参数
        base_price = 50000.0  # BTC基础价格
        volatility = 0.02     # 日波动率
        
        # 生成时间序列
        end_time = datetime.now()
        if self.timeframe == "5m":
            freq = "5T"
        elif self.timeframe == "15m":
            freq = "15T"
        elif self.timeframe == "1h":
            freq = "1H"
        elif self.timeframe == "4h":
            freq = "4H"
        elif self.timeframe == "1d":
            freq = "1D"
        else:
            freq = "5T"
        
        timestamps = pd.date_range(
            end=end_time, 
            periods=n_candles, 
            freq=freq
        )[::-1]  # 反转使时间递增
        
        # 生成随机价格序列
        np.random.seed(42)  # 固定随机种子以便复现
        returns = np.random.normal(0, volatility/np.sqrt(365), n_candles)
        
        # 累积收益率
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
        volumes = np.maximum(volumes, 10)  # 确保最小成交量
        
        # 创建DataFrame
        df = pd.DataFrame({
            "timestamps": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "amount": volumes
        })
        
        # 添加技术指标（简单版本）
        df["volume"] = df["amount"]
        
        print(f"  ✓ 模拟数据生成完成: {len(df)}条记录")
        print(f"    价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def run_backtest(self, 
                     data: pd.DataFrame = None,
                     position_size_pct: float = 0.1) -> dict:
        """运行回测
        
        Args:
            data: 回测数据 (如果为None则使用加载的历史数据)
            position_size_pct: 每次交易的仓位比例
            
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
        print(f"仓位比例: {position_size_pct*100:.1f}%")
        print(f"开始时间: {data['timestamps'].iloc[0]}")
        print(f"结束时间: {data['timestamps'].iloc[-1]}")
        
        # 重置回测状态
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_entry_price = 0.0
        self.trade_history = []
        self.equity_curve = []
        self.signals = []
        
        # 回测主循环
        total_candles = len(data)
        for i in range(60, total_candles):  # 从第60根K线开始，确保有足够历史数据
            # 更新进度
            if i % 100 == 0:
                progress = (i / total_candles) * 100
                print(f"  回测进度: {progress:.1f}% ({i}/{total_candles})", end="\r")
            
            # 获取当前K线数据
            current_time = data['timestamps'].iloc[i]
            current_price = data['close'].iloc[i]
            
            # 获取历史窗口数据用于分析
            lookback_data = data.iloc[max(0, i-100):i+1].copy()
            
            # 生成交易信号
            signal = self._generate_signal(lookback_data)
            
            # 记录信号
            signal_record = {
                "timestamp": current_time,
                "price": current_price,
                "signal": signal.get("action", "HOLD"),
                "confidence": signal.get("confidence", 0.0),
                "position_size": signal.get("position_size", 0.0)
            }
            self.signals.append(signal_record)
            
            # 执行交易逻辑
            self._execute_trading_logic(
                signal=signal,
                current_price=current_price,
                current_time=current_time,
                position_size_pct=position_size_pct
            )
            
            # 更新权益曲线
            current_equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                "timestamp": current_time,
                "equity": current_equity,
                "price": current_price,
                "position": self.position
            })
        
        print(f"\n✓ 回测完成: {len(data)}条K线分析完毕")
        
        # 计算性能指标
        self._calculate_performance_metrics()
        
        # 生成回测报告
        report = self._generate_backtest_report()
        
        return report
    
    def _generate_signal(self, data: pd.DataFrame) -> dict:
        """生成交易信号
        
        Args:
            data: 历史K线数据
            
        Returns:
            交易信号字典
        """
        if self.strategy_coordinator is None:
            return {"action": "HOLD", "confidence": 0.0}
        
        try:
            # 使用策略协调器分析市场
            analysis_result = self.strategy_coordinator.analyze_market(data)
            
            # 提取交易建议
            recommendation = analysis_result.get("trading_recommendation", {})
            
            # 简化信号格式
            signal = {
                "action": recommendation.get("action", "HOLD"),
                "confidence": recommendation.get("confidence", 0.0),
                "position_size": recommendation.get("position_size", 0.0),
                "reasoning": recommendation.get("reasoning", []),
                "signal_strength": recommendation.get("signal_strength", 0.0),
                "risk_level": recommendation.get("risk_level", "LOW")
            }
            
            return signal
            
        except Exception as e:
            print(f"  信号生成失败: {e}")
            return {"action": "HOLD", "confidence": 0.0}
    
    def _execute_trading_logic(self, 
                              signal: dict, 
                              current_price: float,
                              current_time: datetime,
                              position_size_pct: float):
        """执行交易逻辑
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            current_time: 当前时间
            position_size_pct: 仓位比例
        """
        action = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0.0)
        
        # 置信度阈值过滤
        if confidence < 0.3:
            action = "HOLD"
        
        # 风险等级过滤
        risk_level = signal.get("risk_level", "LOW")
        if risk_level == "HIGH":
            action = "HOLD"
        
        # 执行交易
        if action == "BUY" and self.position <= 0:
            # 开多仓
            self._open_long_position(current_price, current_time, position_size_pct)
            
        elif action == "SELL" and self.position >= 0:
            # 开空仓 (简化处理，实际加密货币交易可能需要借贷)
            # 这里我们简化为平多仓并开空仓
            if self.position > 0:
                self._close_position(current_price, current_time, "信号反转")
            # 开空仓
            self._open_short_position(current_price, current_time, position_size_pct)
            
        elif action == "HOLD":
            # 持仓管理：检查止损止盈
            self._manage_position(current_price, current_time)
    
    def _open_long_position(self, price: float, time: datetime, position_size_pct: float):
        """开多仓
        
        Args:
            price: 入场价格
            time: 入场时间
            position_size_pct: 仓位比例
        """
        if self.position > 0:
            return  # 已有持仓
            
        # 计算交易数量（基于仓位比例）
        position_value = self.capital * position_size_pct
        position_amount = position_value / price
        
        # 考虑手续费和滑点
        effective_price = price * (1 + self.slippage)
        commission = position_value * self.commission_rate
        
        # 检查资金是否足够
        total_cost = position_value + commission
        if total_cost > self.capital:
            print(f"  资金不足: 需要${total_cost:.2f}, 可用${self.capital:.2f}")
            return
        
        # 更新持仓
        self.position = position_amount
        self.position_entry_price = effective_price
        self.capital -= total_cost
        
        # 记录交易
        trade_record = {
            "time": time,
            "type": "LONG",
            "entry_price": effective_price,
            "amount": position_amount,
            "position_value": position_value,
            "commission": commission,
            "current_capital": self.capital,
            "reason": "策略信号开多仓"
        }
        self.trade_history.append(trade_record)
        
        print(f"  [{time.strftime('%Y-%m-%d %H:%M')}] 开多仓: {position_amount:.6f} BTC @ ${effective_price:.2f}")
    
    def _open_short_position(self, price: float, time: datetime, position_size_pct: float):
        """开空仓（简化版本，实际需要借贷）
        
        Args:
            price: 入场价格
            time: 入场时间
            position_size_pct: 仓位比例
        """
        if self.position < 0:
            return  # 已有空仓
            
        # 如果有持仓，先平仓
        if self.position > 0:
            self._close_position(price, time, "平多开空")
        
        # 计算交易数量（基于仓位比例）
        position_value = self.capital * position_size_pct
        position_amount = position_value / price
        
        # 考虑手续费和滑点
        effective_price = price * (1 - self.slippage)  # 做空时价格较低有利
        commission = position_value * self.commission_rate
        
        # 检查资金是否足够
        total_cost = position_value * 0.5 + commission  # 做空需要保证金
        if total_cost > self.capital:
            print(f"  资金不足: 需要${total_cost:.2f}, 可用${self.capital:.2f}")
            return
        
        # 更新持仓（负值表示空仓）
        self.position = -position_amount
        self.position_entry_price = effective_price
        self.capital -= total_cost  # 扣除保证金和手续费
        
        # 记录交易
        trade_record = {
            "time": time,
            "type": "SHORT",
            "entry_price": effective_price,
            "amount": position_amount,
            "position_value": position_value,
            "commission": commission,
            "current_capital": self.capital,
            "reason": "策略信号开空仓"
        }
        self.trade_history.append(trade_record)
        
        print(f"  [{time.strftime('%Y-%m-%d %H:%M')}] 开空仓: {position_amount:.6f} BTC @ ${effective_price:.2f}")
    
    def _close_position(self, price: float, time: datetime, reason: str = ""):
        """平仓
        
        Args:
            price: 平仓价格
            time: 平仓时间
            reason: 平仓原因
        """
        if self.position == 0:
            return  # 没有持仓
            
        # 计算平仓收益
        position_amount = abs(self.position)
        entry_price = self.position_entry_price
        
        if self.position > 0:  # 平多仓
            # 考虑手续费和滑点
            effective_price = price * (1 - self.slippage)
            pnl = (effective_price - entry_price) * position_amount
        else:  # 平空仓
            # 考虑手续费和滑点
            effective_price = price * (1 + self.slippage)
            pnl = (entry_price - effective_price) * position_amount
        
        # 手续费
        position_value = position_amount * effective_price
        commission = position_value * self.commission_rate
        
        # 净收益
        net_pnl = pnl - commission
        
        # 更新资金和持仓
        self.capital += (position_amount * entry_price) + net_pnl  # 返还本金 + 净收益
        self.position = 0.0
        self.position_entry_price = 0.0
        
        # 记录交易
        trade_record = {
            "time": time,
            "type": "CLOSE",
            "exit_price": effective_price,
            "amount": position_amount,
            "pnl": pnl,
            "commission": commission,
            "net_pnl": net_pnl,
            "current_capital": self.capital,
            "reason": reason
        }
        self.trade_history.append(trade_record)
        
        position_type = "多仓" if self.position > 0 else "空仓"
        print(f"  [{time.strftime('%Y-%m-%d %H:%M')}] 平{position_type}: 盈利${net_pnl:.2f} ({pnl/position_amount/entry_price*100:.2f}%)")
    
    def _manage_position(self, current_price: float, current_time: datetime):
        """持仓管理：止损止盈检查
        
        Args:
            current_price: 当前价格
            current_time: 当前时间
        """
        if self.position == 0:
            return
        
        entry_price = self.position_entry_price
        
        # 计算盈亏百分比
        if self.position > 0:  # 多仓
            pnl_pct = (current_price - entry_price) / entry_price
            # 止损：亏损超过5%
            if pnl_pct < -0.05:
                self._close_position(current_price, current_time, "多仓止损")
            # 止盈：盈利超过10%
            elif pnl_pct > 0.10:
                self._close_position(current_price, current_time, "多仓止盈")
        else:  # 空仓
            pnl_pct = (entry_price - current_price) / entry_price
            # 止损：亏损超过5%
            if pnl_pct < -0.05:
                self._close_position(current_price, current_time, "空仓止损")
            # 止盈：盈利超过10%
            elif pnl_pct > 0.10:
                self._close_position(current_price, current_time, "空仓止盈")
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算当前权益
        
        Args:
            current_price: 当前价格
            
        Returns:
            当前总权益
        """
        position_value = 0.0
        if self.position > 0:  # 多仓
            position_value = self.position * current_price
        elif self.position < 0:  # 空仓
            position_value = abs(self.position) * (self.position_entry_price - current_price) + abs(self.position) * self.position_entry_price
        
        total_equity = self.capital + position_value
        return total_equity
    
    def _calculate_performance_metrics(self):
        """计算回测性能指标"""
        if not self.equity_curve:
            return
        
        # 提取权益曲线
        equity_values = [e["equity"] for e in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        if len(returns) == 0:
            return
        
        # 基础指标
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (365 / len(self.equity_curve)) if len(self.equity_curve) > 0 else 0
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # 回撤分析
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # 交易统计
        trades = self.trade_history
        total_trades = len([t for t in trades if t["type"] in ["LONG", "SHORT"]])
        winning_trades = len([t for t in trades if t.get("net_pnl", 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏统计
        total_pnl = sum(t.get("net_pnl", 0) for t in trades)
        avg_win = np.mean([t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
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
            "trade_history": self.trade_history[-20:],  # 返回最近20笔交易
            "equity_curve_sample": self.equity_curve[::max(1, len(self.equity_curve)//100)],  # 采样100个点
            "signals_sample": self.signals[::max(1, len(self.signals)//50)],  # 采样50个信号
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def print_report(self, report: dict = None):
        """打印回测报告
        
        Args:
            report: 回测报告 (如果为None则使用最新报告)
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
            filepath: 文件路径 (如果为None则生成默认路径)
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
        use_fingpt=True,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # 加载历史数据
    data = engine.load_historical_data(limit=500)
    
    # 运行回测
    report = engine.run_backtest(data=data, position_size_pct=0.1)
    
    # 打印报告
    engine.print_report(report)
    
    # 导出报告
    engine.export_report(report)
    
    print("\n回测测试完成！")