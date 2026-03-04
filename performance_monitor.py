#!/usr/bin/env python
"""性能监控器 - 实时监控交易表现，触发自动优化"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Optional, Callable
import warnings
warnings.filterwarnings("ignore")

class PerformanceMonitor:
    """交易表现监控器 - 实时监控交易性能并计算关键指标"""
    
    def __init__(self, 
                 strategy_instance=None,
                 check_interval_seconds: int = 900,  # 每15分钟检查一次
                 min_trades_for_analysis: int = 10,
                 performance_history_file: str = None,
                 judgment_mode: str = "threshold",  # threshold / ai / hybrid
                 qwen_optimizer=None):
        """
        初始化性能监控器
        
        Args:
            strategy_instance: 专业策略实例 (ProfessionalTradingStrategy)
            check_interval_seconds: 检查间隔秒数
            min_trades_for_analysis: 分析所需的最小交易数量
            performance_history_file: 性能历史记录文件
            judgment_mode: 判断模式 ('threshold'固定阈值 / 'ai'AI判断 / 'hybrid'混合模式)
            qwen_optimizer: Qwen优化器实例（用于AI判断模式）
        """
        # 兼容 PyInstaller 打包环境
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(__file__)
        
        if performance_history_file is None:
            self.history_file = os.path.join(base_dir, "performance_history.json")
        else:
            if os.path.isabs(performance_history_file):
                self.history_file = performance_history_file
            else:
                self.history_file = os.path.join(base_dir, performance_history_file)
        
        self.strategy_instance = strategy_instance
        self.check_interval = check_interval_seconds
        self.min_trades = min_trades_for_analysis
        self.judgment_mode = judgment_mode
        self.qwen_optimizer = qwen_optimizer
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # 性能历史记录
        self.performance_history = []
        self.current_performance = {}
        
        # 阈值配置 (可调整)
        self.thresholds = {
            "sharpe_ratio": 1.0,      # 夏普比率低于1.0触发优化
            "win_rate": 0.45,         # 胜率低于45%触发优化
            "max_drawdown": -0.10,    # 最大回撤超过-10%触发优化
            "profit_factor": 1.1,     # 盈亏比低于1.1触发优化
            "consecutive_losses": 3,  # 连续亏损3次触发优化
            "days_no_profit": 3       # 连续3天无盈利触发优化
        }
        
        # 回调函数
        self.on_threshold_breach = None  # 阈值突破回调
        self.on_performance_update = None  # 性能更新回调
        
        # 加载历史记录
        self.load_performance_history()
        
        mode_desc = {
            "threshold": "固定阈值",
            "ai": "AI判断",
            "hybrid": "混合模式"
        }
        print(f"性能监控器初始化完成 (检查间隔: {check_interval_seconds}秒, 判断模式: {mode_desc.get(judgment_mode, judgment_mode)})")
    
    def start_monitoring(self):
        """开始监控交易表现"""
        if self.is_monitoring:
            print("性能监控器已在运行中")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ 性能监控器已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("性能监控器未运行")
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("⏹️ 性能监控器已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        mode_desc = {
            "threshold": "固定阈值",
            "ai": "AI判断",
            "hybrid": "混合模式"
        }
        print(f"性能监控循环开始 (间隔: {self.check_interval}秒, 模式: {mode_desc.get(self.judgment_mode, self.judgment_mode)})")
        
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                # 计算当前性能指标
                performance = self.calculate_performance()
                
                if performance:
                    # 更新当前性能
                    self.current_performance = performance
                    
                    # 保存到历史记录
                    self._save_performance_to_history(performance)
                    
                    # 触发性能更新回调
                    if self.on_performance_update:
                        self.on_performance_update(performance)
                    
                    # 根据判断模式检查是否需要优化
                    should_optimize = False
                    optimize_reasons = []
                    
                    if self.judgment_mode == "threshold":
                        # 固定阈值模式
                        threshold_breaches = self.check_thresholds(performance)
                        should_optimize = len(threshold_breaches) > 0
                        optimize_reasons = threshold_breaches
                    elif self.judgment_mode == "ai":
                        # 纯AI判断模式
                        should_optimize, optimize_reasons = self.ai_should_optimize(performance)
                    elif self.judgment_mode == "hybrid":
                        # 混合模式：先检查固定阈值，再由AI确认
                        threshold_breaches = self.check_thresholds(performance)
                        if threshold_breaches:
                            print(f"📊 固定阈值检测到突破，请求AI确认...")
                            should_optimize, optimize_reasons = self.ai_should_optimize(performance)
                        else:
                            should_optimize = False
                    
                    if should_optimize:
                        print(f"⚠️ 检测到需要优化: {optimize_reasons}")
                        
                        # 触发阈值突破回调
                        if self.on_threshold_breach:
                            self.on_threshold_breach(optimize_reasons, performance)
                
                # 等待下一次检查
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                print(f"性能监控循环错误: {e}")
                time.sleep(self.check_interval)  # 出错时等待
    
    def calculate_performance(self) -> Dict:
        """计算当前交易表现指标"""
        if not self.strategy_instance:
            print("⚠️ 策略实例未设置，无法计算性能")
            return {}
        
        try:
            # 获取交易历史
            trade_history = getattr(self.strategy_instance, 'trade_history', [])
            
            if len(trade_history) < self.min_trades:
                # print(f"交易数量不足 ({len(trade_history)} < {self.min_trades})，跳过性能计算")
                return {}
            
            # 解析交易历史为DataFrame
            trades_df = self._parse_trade_history(trade_history)
            
            if trades_df.empty:
                return {}
            
            # 计算基础指标
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 盈亏统计
            total_pnl = trades_df['pnl'].sum()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # 盈亏比
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
            
            # 计算权益曲线 (简化版)
            equity_curve = []
            current_balance = 10000  # 假设初始资金
            for _, trade in trades_df.iterrows():
                current_balance += trade['pnl']
                equity_curve.append(current_balance)
            
            # 计算回报率序列
            if len(equity_curve) > 1:
                equity_values = np.array(equity_curve)
                returns = np.diff(equity_values) / equity_values[:-1]
                
                # 夏普比率 (简化版)
                sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0
                
                # 最大回撤
                peak = np.maximum.accumulate(equity_values)
                drawdown = (equity_values - peak) / peak
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            # 计算连续亏损
            consecutive_losses = self._calculate_consecutive_losses(trades_df)
            
            # 计算无盈利天数
            days_no_profit = self._calculate_days_no_profit(trades_df)
            
            # 构建性能报告
            performance = {
                "timestamp": datetime.now().isoformat(),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "consecutive_losses": consecutive_losses,
                "days_no_profit": days_no_profit,
                "current_balance": equity_curve[-1] if equity_curve else 10000,
                "analysis_valid": True
            }
            
            return performance
            
        except Exception as e:
            print(f"计算性能指标失败: {e}")
            return {}
    
    def _parse_trade_history(self, trade_history: List) -> pd.DataFrame:
        """解析交易历史为DataFrame"""
        records = []
        
        for trade in trade_history:
            # 检查是否有盈亏信息 - 支持 pnl 和 pnl_pct 字段
            pnl = None
            
            # 优先使用 pnl 字段
            if 'pnl' in trade:
                pnl = trade['pnl']
            elif 'pnl_pct' in trade:
                pnl = trade['pnl_pct']  # 百分比值
            elif 'action' in trade and trade['action'].startswith('CLOSE'):
                # 如果是平仓记录但没有盈亏信息，尝试从其他字段计算
                entry_price = trade.get('entry_price')
                close_price = trade.get('price')
                if entry_price and close_price:
                    if 'LONG' in trade['action']:
                        pnl = (close_price - entry_price) / entry_price * 100  # 百分比
                    elif 'SHORT' in trade['action']:
                        pnl = (entry_price - close_price) / entry_price * 100  # 百分比
            
            if pnl is not None:
                records.append({
                    'timestamp': trade.get('timestamp', trade.get('time', datetime.now())),
                    'action': trade.get('action', 'UNKNOWN'),
                    'price': trade.get('price', 0),
                    'size': trade.get('size', 0),
                    'pnl': pnl,  # 可能是百分比值
                    'pnl_pct': pnl if isinstance(pnl, (int, float)) else 0,
                    'close_reason': trade.get('reason', trade.get('close_reason', 'UNKNOWN'))
                })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # 确保时间戳是datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _calculate_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """计算连续亏损次数"""
        if trades_df.empty:
            return 0
        
        # 获取最近的交易
        recent_trades = trades_df.tail(20)  # 检查最近20笔交易
        
        consecutive = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for _, trade in recent_trades.iterrows():
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_days_no_profit(self, trades_df: pd.DataFrame) -> int:
        """计算连续无盈利天数"""
        if trades_df.empty:
            return 0
        
        # 按日期分组计算每日盈亏
        trades_df['date'] = trades_df['timestamp'].dt.date
        daily_pnl = trades_df.groupby('date')['pnl'].sum()
        
        # 检查最近日期
        if len(daily_pnl) == 0:
            return 0
        
        # 计算连续无盈利天数
        consecutive_days = 0
        for date in sorted(daily_pnl.index, reverse=True):  # 从最新日期开始
            if daily_pnl[date] <= 0:
                consecutive_days += 1
            else:
                break
        
        return consecutive_days
    
    def check_thresholds(self, performance: Dict) -> List[str]:
        """检查性能指标是否突破阈值"""
        breaches = []
        
        # 检查各指标阈值
        if performance.get('sharpe_ratio', 0) < self.thresholds['sharpe_ratio']:
            breaches.append(f"夏普比率过低: {performance.get('sharpe_ratio', 0):.2f} < {self.thresholds['sharpe_ratio']}")
        
        if performance.get('win_rate', 0) < self.thresholds['win_rate']:
            breaches.append(f"胜率过低: {performance.get('win_rate', 0):.2f} < {self.thresholds['win_rate']}")
        
        if performance.get('max_drawdown', 0) < self.thresholds['max_drawdown']:
            breaches.append(f"最大回撤过大: {performance.get('max_drawdown', 0):.2f} < {self.thresholds['max_drawdown']}")
        
        if performance.get('profit_factor', float('inf')) < self.thresholds['profit_factor']:
            breaches.append(f"盈亏比过低: {performance.get('profit_factor', 0):.2f} < {self.thresholds['profit_factor']}")
        
        if performance.get('consecutive_losses', 0) >= self.thresholds['consecutive_losses']:
            breaches.append(f"连续亏损: {performance.get('consecutive_losses', 0)} >= {self.thresholds['consecutive_losses']}")
        
        if performance.get('days_no_profit', 0) >= self.thresholds['days_no_profit']:
            breaches.append(f"连续无盈利天数: {performance.get('days_no_profit', 0)} >= {self.thresholds['days_no_profit']}")
        
        return breaches
    
    def set_judgment_mode(self, mode: str, qwen_optimizer=None):
        """设置判断模式
        
        Args:
            mode: 判断模式 ('threshold'固定阈值 / 'ai'AI判断 / 'hybrid'混合模式
            qwen_optimizer: Qwen优化器实例（用于AI判断模式）
        """
        self.judgment_mode = mode
        if qwen_optimizer:
            self.qwen_optimizer = qwen_optimizer
        
        mode_desc = {
            "threshold": "固定阈值",
            "ai": "AI判断",
            "hybrid": "混合模式"
        }
        print(f"性能监控器判断模式已切换为: {mode_desc.get(mode, mode)}")
    
    def ai_should_optimize(self, performance: Dict) -> tuple[bool, List[str]]:
        """使用AI判断是否需要优化
        
        Args:
            performance: 当前性能数据
            
        Returns:
            (是否需要优化, 优化原因列表)
        """
        if not self.qwen_optimizer or not self.qwen_optimizer.is_loaded:
            print("⚠️ Qwen优化器未加载，回退到固定阈值模式")
            breaches = self.check_thresholds(performance)
            return (len(breaches) > 0), breaches
        
        try:
            # 获取最近的性能历史用于AI分析
            recent_history = self.performance_history[-20:] if len(self.performance_history) else []
            
            # 构建AI提示词
            prompt = self._build_ai_judgment_prompt(performance, recent_history)
            
            # 调用Qwen进行判断
            response = self.qwen_optimizer.generate(
                prompt,
                max_tokens=512,
                temperature=0.3
            )
            
            # 解析AI响应
            should_optimize, reasons = self._parse_ai_response(response)
            
            return should_optimize, reasons
            
        except Exception as e:
            print(f"⚠️ AI判断失败: {e}，回退到固定阈值模式")
            breaches = self.check_thresholds(performance)
            return (len(breaches) > 0), breaches
    
    def _build_ai_judgment_prompt(self, performance: Dict, recent_history: List[Dict]) -> str:
        """构建AI判断提示词"""
        perf_summary = f"""
【基本交易数据】
总交易数: {performance.get('total_trades', 0)}
盈利交易数: {performance.get('winning_trades', 0)}
亏损交易数: {performance.get('losing_trades', 0)}
胜率: {performance.get('win_rate', 0):.1%}

【收益表现】
总盈亏: ${performance.get('total_pnl', 0):.2f}
平均盈利: ${performance.get('avg_win', 0):.2f}
平均亏损: ${performance.get('avg_loss', 0):.2f}
盈亏比: {performance.get('profit_factor', 0):.2f}
当前余额: ${performance.get('current_balance', 0):.2f}

【风险指标】
夏普比率: {performance.get('sharpe_ratio', 0):.2f}
最大回撤: {performance.get('max_drawdown', 0):.2%}
连续亏损: {performance.get('consecutive_losses', 0)}次
连续无盈利天数: {performance.get('days_no_profit', 0)}天
"""
        
        history_summary = ""
        if recent_history:
            history_summary = "\n【最近性能趋势（最后5次记录）】\n"
            for i, hist in enumerate(recent_history[-5:], 1):
                history_summary += f"  记录{i}: 胜率{hist.get('win_rate',0):.1%}, 夏普{hist.get('sharpe_ratio',0):.2f}, 最大回撤{hist.get('max_drawdown',0):.2%}, 盈亏比{hist.get('profit_factor',0):.2f}\n"
        
        prompt = f"""你是一个专业的量化交易策略专家。请综合分析以下交易表现数据，判断是否需要进行策略优化。

{perf_summary}
{history_summary}

【分析要求】
1. 请综合考虑所有指标，不要只看单一指标
2. 考虑性能趋势（是在变好还是变差）
3. 权衡收益和风险
4. 判断当前策略是否需要优化

请以JSON格式返回:
{{
    "should_optimize": true/false,
    "reasons": ["原因1", "原因2", "原因3"],
    "confidence": 0.95
}}

只返回JSON，不要其他文字。
"""
        return prompt
    
    def _parse_ai_response(self, response: str) -> tuple[bool, List[str]]:
        """解析AI响应"""
        import re
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                import json
                result = json.loads(json_match.group())
                should_optimize = result.get('should_optimize', False)
                reasons = result.get('reasons', [])
                return should_optimize, reasons
        except Exception as e:
            print(f"解析AI响应失败: {e}")
        
        return False, []
    
    def _save_performance_to_history(self, performance: Dict):
        """保存性能数据到历史记录"""
        self.performance_history.append(performance)
        
        # 限制历史记录长度
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # 定期保存到文件
        if len(self.performance_history) % 10 == 0:  # 每10次保存一次
            self.save_performance_history()
    
    def save_performance_history(self):
        """保存性能历史到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"保存性能历史失败: {e}")
    
    def load_performance_history(self):
        """从文件加载性能历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.performance_history = json.load(f)
                print(f"加载性能历史: {len(self.performance_history)}条记录")
        except Exception as e:
            print(f"加载性能历史失败: {e}")
            self.performance_history = []
    
    def get_performance_summary(self) -> str:
        """获取性能摘要字符串"""
        if not self.current_performance:
            return "暂无性能数据"
        
        perf = self.current_performance
        summary = []
        summary.append("=== 当前交易表现 ===")
        summary.append(f"总交易数: {perf.get('total_trades', 0)}")
        summary.append(f"胜率: {perf.get('win_rate', 0):.1%}")
        summary.append(f"夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
        summary.append(f"最大回撤: {perf.get('max_drawdown', 0):.2%}")
        summary.append(f"盈亏比: {perf.get('profit_factor', 0):.2f}")
        summary.append(f"总盈亏: ${perf.get('total_pnl', 0):.2f}")
        summary.append(f"连续亏损: {perf.get('consecutive_losses', 0)}次")
        summary.append(f"连续无盈利天数: {perf.get('days_no_profit', 0)}天")
        
        return "\n".join(summary)
    
    def set_strategy_instance(self, strategy_instance):
        """设置策略实例"""
        self.strategy_instance = strategy_instance
        print(f"性能监控器已连接到策略实例: {type(strategy_instance).__name__}")
    
    def set_threshold(self, metric: str, value: float):
        """设置阈值"""
        if metric in self.thresholds:
            old_value = self.thresholds[metric]
            self.thresholds[metric] = value
            print(f"阈值更新: {metric} {old_value} -> {value}")
        else:
            print(f"未知指标: {metric}")
    
    def set_callback(self, callback_type: str, callback_func: Callable):
        """设置回调函数"""
        if callback_type == "threshold_breach":
            self.on_threshold_breach = callback_func
            print("阈值突破回调已设置")
        elif callback_type == "performance_update":
            self.on_performance_update = callback_func
            print("性能更新回调已设置")
        else:
            print(f"未知回调类型: {callback_type}")


# 测试函数
if __name__ == "__main__":
    print("测试性能监控器...")
    
    # 创建模拟策略实例
    class MockStrategy:
        def __init__(self):
            self.trade_history = []
            
            # 生成模拟交易记录
            import random
            base_price = 50000
            for i in range(50):
                pnl = random.uniform(-100, 200)
                self.trade_history.append({
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'action': 'CLOSE_LONG' if pnl > 0 else 'CLOSE_SHORT',
                    'price': base_price + random.uniform(-1000, 1000),
                    'size': 0.01,
                    'pnl': pnl,
                    'close_reason': 'TAKE_PROFIT' if pnl > 0 else 'STOP_LOSS'
                })
    
    mock_strategy = MockStrategy()
    
    # 创建监控器
    monitor = PerformanceMonitor(
        strategy_instance=mock_strategy,
        check_interval_seconds=10,
        min_trades_for_analysis=5
    )
    
    # 测试性能计算
    print("\n计算性能指标...")
    performance = monitor.calculate_performance()
    
    if performance:
        print(f"交易数: {performance.get('total_trades', 0)}")
        print(f"胜率: {performance.get('win_rate', 0):.1%}")
        print(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {performance.get('max_drawdown', 0):.2%}")
        
        # 测试阈值检查
        print("\n检查阈值...")
        breaches = monitor.check_thresholds(performance)
        if breaches:
            print("阈值突破:")
            for breach in breaches:
                print(f"  • {breach}")
        else:
            print("所有指标均在阈值范围内")
    else:
        print("无法计算性能指标")
    
    print("\n✅ 性能监控器测试完成")