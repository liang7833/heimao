#!/usr/bin/env python
"""策略适配器 - 将策略协调器信号转换为具体的交易指令"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 导入现有模块
try:
    from strategy_coordinator import StrategyCoordinator
    from enhanced_kronos import EnhancedKronosAnalyzer
    from fingpt_analyzer import FinGPTSentimentAnalyzer
except ImportError as e:
    print(f"模块导入失败: {e}")


class StrategyAdapter:
    """策略适配器 - 将AI信号转换为具体交易指令"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 use_fingpt: bool = True,
                 risk_level: str = "MEDIUM"):
        """
        初始化策略适配器
        
        Args:
            symbol: 交易品种
            use_fingpt: 是否使用FinGPT舆情分析
            risk_level: 风险等级 (LOW, MEDIUM, HIGH)
        """
        self.symbol = symbol
        self.use_fingpt = use_fingpt
        self.risk_level = risk_level
        
        # 策略协调器
        self.coordinator = None
        
        # 风险参数配置 (根据风险等级调整)
        self.risk_params = self._get_risk_params(risk_level)
        
        # 交易参数
        self.position_size_multiplier = 1.0
        self.stop_loss_pct = 0.05  # 默认止损5%
        self.take_profit_pct = 0.10  # 默认止盈10%
        self.confidence_threshold = 0.3  # 信号置信度阈值
        
        # 状态跟踪
        self.last_signal = None
        self.signal_history = []
        self.trade_log = []
        
        print(f"策略适配器初始化: {symbol} (风险等级: {risk_level})")
        self._initialize_coordinator()
    
    def _get_risk_params(self, risk_level: str) -> Dict:
        """根据风险等级获取风险参数"""
        risk_configs = {
            "LOW": {
                "position_size": 0.05,  # 5%仓位
                "stop_loss": 0.03,      # 3%止损
                "take_profit": 0.06,    # 6%止盈
                "max_daily_trades": 5,
                "confidence_threshold": 0.4
            },
            "MEDIUM": {
                "position_size": 0.10,  # 10%仓位
                "stop_loss": 0.05,      # 5%止损
                "take_profit": 0.10,    # 10%止盈
                "max_daily_trades": 10,
                "confidence_threshold": 0.3
            },
            "HIGH": {
                "position_size": 0.20,  # 20%仓位
                "stop_loss": 0.08,      # 8%止损
                "take_profit": 0.15,    # 15%止盈
                "max_daily_trades": 20,
                "confidence_threshold": 0.2
            }
        }
        
        return risk_configs.get(risk_level, risk_configs["MEDIUM"])
    
    def _initialize_coordinator(self):
        """初始化策略协调器"""
        try:
            self.coordinator = StrategyCoordinator(
                kronos_model_name="kronos-small",
                use_fingpt=self.use_fingpt,
                symbol=self.symbol.replace("USDT", "")  # 移除USDT后缀
            )
            print("  ✓ 策略协调器初始化成功")
        except Exception as e:
            print(f"  ✗ 策略协调器初始化失败: {e}")
            self.coordinator = None
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """分析市场数据，生成交易决策
        
        Args:
            market_data: 市场K线数据
            
        Returns:
            交易决策字典
        """
        if self.coordinator is None:
            return self._generate_default_decision()
        
        try:
            # 使用策略协调器分析市场
            analysis_result = self.coordinator.analyze_market(market_data)
            
            # 提取关键信息
            signal = analysis_result.get("trading_recommendation", {})
            kronos_signal = analysis_result.get("kronos_signal", {})
            sentiment_analysis = analysis_result.get("sentiment_analysis", {})
            
            # 生成交易决策
            decision = self._generate_trading_decision(
                signal=signal,
                kronos_signal=kronos_signal,
                sentiment_analysis=sentiment_analysis,
                market_data=market_data
            )
            
            # 记录信号历史
            self._record_signal_history(decision, analysis_result)
            
            return decision
            
        except Exception as e:
            print(f"市场分析失败: {e}")
            return self._generate_default_decision()
    
    def _generate_trading_decision(self, 
                                  signal: Dict,
                                  kronos_signal: Dict,
                                  sentiment_analysis: Dict,
                                  market_data: pd.DataFrame) -> Dict:
        """生成具体交易决策
        
        Args:
            signal: 策略协调器信号
            kronos_signal: Kronos原始信号
            sentiment_analysis: 舆情分析结果
            market_data: 市场数据
            
        Returns:
            交易决策字典
        """
        # 基础决策
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "action": "HOLD",  # 默认持有
            "confidence": 0.0,
            "position_size": 0.0,
            "reasoning": [],
            "risk_level": self.risk_level,
            "entry_price_range": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "signal_details": {
                "kronos_direction": kronos_signal.get("trend_direction", "NEUTRAL"),
                "kronos_strength": kronos_signal.get("trend_strength", 0.0),
                "sentiment": sentiment_analysis.get("overall_sentiment", "NEUTRAL"),
                "sentiment_score": sentiment_analysis.get("sentiment_score", 0.0),
                "risk_assessment": sentiment_analysis.get("risk_level", "LOW")
            }
        }
        
        # 提取信号信息
        action = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0.0)
        signal_strength = signal.get("signal_strength", 0.0)
        reasoning = signal.get("reasoning", [])
        
        # 应用风险参数
        position_size = self.risk_params["position_size"]
        confidence_threshold = self.risk_params["confidence_threshold"]
        
        # 置信度过滤
        if confidence < confidence_threshold:
            action = "HOLD"
            reasoning.append(f"置信度过低 ({confidence:.2f} < {confidence_threshold})")
        
        # 风险等级过滤
        risk_assessment = sentiment_analysis.get("risk_level", "LOW")
        if risk_assessment == "HIGH":
            action = "HOLD"
            reasoning.append("高风险环境，暂停交易")
        
        # 更新决策
        decision["action"] = action
        decision["confidence"] = confidence
        decision["reasoning"] = reasoning
        
        # 生成具体的交易指令
        if action in ["BUY", "SELL"]:
            current_price = market_data["close"].iloc[-1] if not market_data.empty else 0
            
            # 计算仓位大小（根据风险等级调整）
            risk_adjusted_size = position_size * min(confidence / 0.5, 1.0)  # 置信度调整
            decision["position_size"] = risk_adjusted_size
            
            # 生成入场价格范围
            if kronos_signal:
                pred_support = kronos_signal.get("pred_support", current_price * 0.99)
                pred_resistance = kronos_signal.get("pred_resistance", current_price * 1.01)
                
                if action == "BUY":
                    entry_min = pred_support
                    entry_max = current_price * 1.005
                    stop_loss = pred_support * 0.99
                    take_profit = current_price * (1 + self.risk_params["take_profit"])
                else:  # SELL
                    entry_min = current_price * 0.995
                    entry_max = pred_resistance
                    stop_loss = pred_resistance * 1.01
                    take_profit = current_price * (1 - self.risk_params["take_profit"])
                
                decision["entry_price_range"] = [float(entry_min), float(entry_max)]
                decision["stop_loss_price"] = float(stop_loss)
                decision["take_profit_price"] = float(take_profit)
        
        # 检查拐点信息（如果有）
        if kronos_signal and kronos_signal.get("has_turning_point", False):
            recent_turn_type = kronos_signal.get("recent_turn_type")
            recent_turn_price = kronos_signal.get("recent_turn_price")
            
            if recent_turn_type and recent_turn_price:
                decision["signal_details"]["turning_point"] = {
                    "type": recent_turn_type,
                    "price": recent_turn_price,
                    "time_offset": kronos_signal.get("recent_turn_time_offset", 0)
                }
                
                # 拐点一致性检查
                if action == "BUY" and recent_turn_type == "VALLEY":
                    decision["reasoning"].append("近期检测到价格谷值，支持买入信号")
                elif action == "SELL" and recent_turn_type == "PEAK":
                    decision["reasoning"].append("近期检测到价格峰值，支持卖出信号")
        
        return decision
    
    def _generate_default_decision(self) -> Dict:
        """生成默认决策（当分析失败时）"""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "action": "HOLD",
            "confidence": 0.0,
            "position_size": 0.0,
            "reasoning": ["系统初始化失败或数据不足"],
            "risk_level": self.risk_level,
            "entry_price_range": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "signal_details": {}
        }
    
    def _record_signal_history(self, decision: Dict, analysis_result: Dict):
        """记录信号历史"""
        signal_record = {
            "timestamp": decision["timestamp"],
            "action": decision["action"],
            "confidence": decision["confidence"],
            "position_size": decision["position_size"],
            "reasoning": decision["reasoning"],
            "kronos_signal": analysis_result.get("kronos_signal", {}),
            "sentiment_analysis": analysis_result.get("sentiment_analysis", {}),
            "performance_stats": analysis_result.get("performance_stats", {})
        }
        
        self.signal_history.append(signal_record)
        self.last_signal = signal_record
        
        # 保留最近1000条记录
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def generate_trade_instructions(self, decision: Dict, account_balance: float) -> List[Dict]:
        """根据决策生成具体的交易指令
        
        Args:
            decision: 交易决策
            account_balance: 账户余额 (USDT)
            
        Returns:
            交易指令列表
        """
        instructions = []
        
        action = decision.get("action", "HOLD")
        position_size = decision.get("position_size", 0.0)
        
        if action == "HOLD" or position_size <= 0:
            return instructions
        
        # 计算交易金额
        trade_amount = account_balance * position_size
        
        # 生成指令
        if action == "BUY":
            instruction = {
                "symbol": self.symbol,
                "side": "BUY",
                "type": "MARKET",  # 市价单
                "quantity": trade_amount,  # 交易金额 (USDT)
                "position_size_pct": position_size * 100,
                "stop_loss": decision.get("stop_loss_price"),
                "take_profit": decision.get("take_profit_price"),
                "reasoning": decision.get("reasoning", []),
                "confidence": decision.get("confidence", 0.0),
                "timestamp": decision.get("timestamp")
            }
            instructions.append(instruction)
            
        elif action == "SELL":
            instruction = {
                "symbol": self.symbol,
                "side": "SELL",
                "type": "MARKET",  # 市价单
                "quantity": trade_amount,  # 交易金额 (USDT)
                "position_size_pct": position_size * 100,
                "stop_loss": decision.get("stop_loss_price"),
                "take_profit": decision.get("take_profit_price"),
                "reasoning": decision.get("reasoning", []),
                "confidence": decision.get("confidence", 0.0),
                "timestamp": decision.get("timestamp")
            }
            instructions.append(instruction)
        
        return instructions
    
    def log_trade(self, 
                  instruction: Dict, 
                  execution_result: Dict,
                  current_price: float):
        """记录交易日志
        
        Args:
            instruction: 交易指令
            execution_result: 执行结果
            current_price: 当前价格
        """
        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "symbol": instruction.get("symbol"),
            "side": instruction.get("side"),
            "quantity": instruction.get("quantity"),
            "position_size_pct": instruction.get("position_size_pct"),
            "instruction_price": instruction.get("price", current_price),
            "execution_price": execution_result.get("price", current_price),
            "stop_loss": instruction.get("stop_loss"),
            "take_profit": instruction.get("take_profit"),
            "confidence": instruction.get("confidence"),
            "reasoning": instruction.get("reasoning", []),
            "execution_status": execution_result.get("status", "UNKNOWN"),
            "execution_message": execution_result.get("message", ""),
            "current_price": current_price
        }
        
        self.trade_log.append(trade_log)
        
        # 保留最近500笔交易记录
        if len(self.trade_log) > 500:
            self.trade_log = self.trade_log[-500:]
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.trade_log:
            return {"total_trades": 0, "recent_performance": "无交易记录"}
        
        # 计算简单统计
        total_trades = len(self.trade_log)
        successful_trades = len([t for t in self.trade_log if t.get("execution_status") == "SUCCESS"])
        buy_trades = len([t for t in self.trade_log if t.get("side") == "BUY"])
        sell_trades = len([t for t in self.trade_log if t.get("side") == "SELL"])
        
        # 最近10笔交易
        recent_trades = self.trade_log[-10:] if total_trades >= 10 else self.trade_log
        
        # 平均置信度
        avg_confidence = np.mean([t.get("confidence", 0) for t in self.trade_log])
        
        return {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "success_rate": successful_trades / total_trades if total_trades > 0 else 0,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_confidence": avg_confidence,
            "recent_trades_count": len(recent_trades),
            "recent_trades": recent_trades,
            "signal_history_count": len(self.signal_history),
            "last_signal_time": self.last_signal.get("timestamp") if self.last_signal else None
        }
    
    def update_risk_level(self, new_risk_level: str):
        """更新风险等级
        
        Args:
            new_risk_level: 新的风险等级 (LOW, MEDIUM, HIGH)
        """
        if new_risk_level not in ["LOW", "MEDIUM", "HIGH"]:
            print(f"警告: 无效的风险等级 {new_risk_level}")
            return
        
        self.risk_level = new_risk_level
        self.risk_params = self._get_risk_params(new_risk_level)
        
        print(f"风险等级已更新: {new_risk_level}")
        print(f"  仓位比例: {self.risk_params['position_size']*100:.1f}%")
        print(f"  止损: {self.risk_params['stop_loss']*100:.1f}%")
        print(f"  止盈: {self.risk_params['take_profit']*100:.1f}%")
        print(f"  置信度阈值: {self.risk_params['confidence_threshold']:.2f}")
    
    def reset_history(self):
        """重置历史记录"""
        self.signal_history = []
        self.trade_log = []
        self.last_signal = None
        print("历史记录已重置")


# 使用示例
if __name__ == "__main__":
    print("=== 策略适配器测试 ===")
    
    # 创建策略适配器
    adapter = StrategyAdapter(
        symbol="BTCUSDT",
        use_fingpt=True,
        risk_level="MEDIUM"
    )
    
    # 生成模拟市场数据
    print("\n生成模拟市场数据...")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5T")
    sample_data = pd.DataFrame({
        "timestamps": dates,
        "open": np.random.normal(50000, 1000, 100),
        "high": np.random.normal(50500, 1200, 100),
        "low": np.random.normal(49500, 1200, 100),
        "close": np.random.normal(50000, 1000, 100),
        "amount": np.random.normal(1000, 200, 100)
    })
    
    # 分析市场
    print("分析市场数据...")
    decision = adapter.analyze_market(sample_data)
    
    # 显示决策结果
    print(f"\n交易决策:")
    print(f"  动作: {decision.get('action')}")
    print(f"  置信度: {decision.get('confidence'):.3f}")
    print(f"  仓位比例: {decision.get('position_size')*100:.1f}%")
    print(f"  风险等级: {decision.get('risk_level')}")
    
    if decision.get("reasoning"):
        print(f"  推理: {'; '.join(decision['reasoning'])}")
    
    # 生成交易指令
    if decision.get("action") != "HOLD":
        print("\n生成交易指令...")
        account_balance = 10000.0
        instructions = adapter.generate_trade_instructions(decision, account_balance)
        
        for i, instr in enumerate(instructions):
            print(f"  指令 {i+1}:")
            print(f"    方向: {instr.get('side')}")
            print(f"    金额: ${instr.get('quantity'):.2f}")
            print(f"    仓位: {instr.get('position_size_pct'):.1f}%")
            if instr.get("stop_loss"):
                print(f"    止损: ${instr.get('stop_loss'):.2f}")
            if instr.get("take_profit"):
                print(f"    止盈: ${instr.get('take_profit'):.2f}")
    
    # 获取性能摘要
    print("\n性能摘要:")
    performance = adapter.get_performance_summary()
    print(f"  总交易次数: {performance.get('total_trades', 0)}")
    print(f"  平均置信度: {performance.get('avg_confidence', 0):.3f}")
    
    print("\n策略适配器测试完成！")