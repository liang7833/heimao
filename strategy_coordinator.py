#!/usr/bin/env python
"""策略协调器 - 整合Kronos预测信号和FinGPT舆情信号，生成最终交易决策"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 导入现有模块
try:
    from enhanced_kronos import EnhancedKronosAnalyzer
except ImportError:
    print("警告: enhanced_kronos模块导入失败，部分功能受限")

try:
    from fingpt_analyzer import FinGPTSentimentAnalyzer
except ImportError:
    print("警告: fingpt_analyzer模块导入失败，部分功能受限")




class StrategyCoordinator:
    """策略协调器 - 整合多源信号，生成优化交易决策"""
    
    def __init__(self, 
                 kronos_model_name: str = "kronos-small",
                 use_fingpt: bool = True,
                 symbol: str = "BTC",
                 kronos_analyzer = None,
                 fingpt_analyzer = None):
        """
        初始化策略协调器
        
        Args:
            kronos_model_name: Kronos模型名称
            use_fingpt: 是否使用FinGPT舆情分析
            symbol: 交易品种符号
            kronos_analyzer: 可选的外部Kronos分析器实例（避免循环创建）
            fingpt_analyzer: 可选的外部FinGPT分析器实例（避免循环创建）
        """
        self.symbol = symbol
        self.use_fingpt = use_fingpt
        
        print(f"初始化策略协调器 ({symbol})...")
        
        # 初始化Kronos分析器（如果外部没有提供）
        if kronos_analyzer is not None:
            print("  使用外部提供的Kronos分析器...")
            self.kronos_analyzer = kronos_analyzer
            self.kronos_available = True
            print("  ✓ Kronos分析器就绪")
        else:
            print("  加载Kronos分析器...")
            try:
                self.kronos_analyzer = EnhancedKronosAnalyzer(model_name=kronos_model_name)
                self.kronos_available = True
                print("  ✓ Kronos分析器就绪")
            except Exception as e:
                print(f"  ✗ Kronos分析器加载失败: {e}")
                self.kronos_available = False
                self.kronos_analyzer = None
        
        # 初始化FinGPT舆情分析器（如果外部没有提供）
        if fingpt_analyzer is not None:
            print("  使用外部提供的FinGPT分析器...")
            self.fingpt_analyzer = fingpt_analyzer
            self.fingpt_available = True
            print("  ✓ FinGPT舆情分析器就绪")
        else:
            self.fingpt_analyzer = None
            if use_fingpt:
                print("  加载FinGPT舆情分析器...")
                try:
                    self.fingpt_analyzer = FinGPTSentimentAnalyzer(
                        use_local_model=True
                    )
                    self.fingpt_available = True
                    print("  ✓ FinGPT舆情分析器就绪")
                except Exception as e:
                    print(f"  ✗ FinGPT舆情分析器加载失败: {e}")
                    self.fingpt_available = False
            else:
                self.fingpt_available = False
        
        # 策略配置
        self.config = {
            "min_signal_strength": 0.0025,  # 最小信号强度（与Kronos趋势强度范围匹配：0.002-0.01）
            "max_position_size": 0.1,    # 最大仓位比例
            "sentiment_weight": 0.3,     # 舆情权重
            "technical_weight": 0.7,     # 技术权重
            "black_swan_threshold": "HIGH",  # 黑天鹅风险阈值
            "enable_adaptive_filtering": True,  # 启用自适应过滤
        }
        
        # 性能跟踪
        self.performance_stats = {
            "total_signals": 0,
            "filtered_signals": 0,
            "sentiment_filtered": 0,
            "risk_filtered": 0,
            "strength_filtered": 0,
            "last_update": datetime.now().isoformat()
        }
        
        print(f"策略协调器初始化完成 (Kronos: {self.kronos_available}, FinGPT: {self.fingpt_available})")
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """综合分析市场数据，生成交易决策
        
        Args:
            market_data: 市场K线数据
            
        Returns:
            综合交易决策
        """
        if market_data.empty or len(market_data) < 20:
            return {
                "error": "市场数据不足",
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"\n=== 市场综合分析开始 ===")
        print(f"数据范围: {len(market_data)}条K线")
        
        # 步骤1: Kronos技术分析
        kronos_signal = None
        if self.kronos_available and self.kronos_analyzer:
            try:
                print("  1. Kronos技术分析...")
                kronos_signal = self.kronos_analyzer.get_enhanced_signal(market_data)
                if kronos_signal:
                    print(f"    - 方向: {kronos_signal.get('trend_direction', '未知')}")
                    print(f"    - 强度: {kronos_signal.get('trend_strength', 0):.3f}")
                    print(f"    - 拐点检测: {kronos_signal.get('has_turning_point', False)}")
            except Exception as e:
                print(f"    ✗ Kronos分析失败: {e}")
                kronos_signal = None
        
        # 步骤2: FinGPT舆情分析
        sentiment_analysis = None
        if self.fingpt_available and self.fingpt_analyzer:
            try:
                print("  2. FinGPT舆情分析...")
                sentiment_analysis = self.fingpt_analyzer.analyze_market_sentiment(self.symbol)
                if sentiment_analysis:
                    print(f"    - 情绪: {sentiment_analysis.get('overall_sentiment', '未知')}")
                    print(f"    - 风险等级: {sentiment_analysis.get('risk_level', '未知')}")
                    print(f"    - 建议: {sentiment_analysis.get('recommendation', '未知')}")
            except Exception as e:
                print(f"    ✗ FinGPT分析失败: {e}")
                sentiment_analysis = None
        
        # 步骤3: 信号整合与过滤
        print("  3. 信号整合与过滤...")
        final_decision = self._integrate_signals(kronos_signal, sentiment_analysis)
        
        # 步骤4: 生成交易建议
        print("  4. 生成交易建议...")
        trading_recommendation = self._generate_trading_recommendation(final_decision)
        
        # 更新性能统计
        self.performance_stats["total_signals"] += 1
        if final_decision.get("signal_filtered", False):
            self.performance_stats["filtered_signals"] += 1
        if final_decision.get("filtered_by_sentiment", False):
            self.performance_stats["sentiment_filtered"] += 1
        if final_decision.get("filtered_by_risk", False):
            self.performance_stats["risk_filtered"] += 1
        if final_decision.get("filtered_by_strength", False):
            self.performance_stats["strength_filtered"] += 1
        self.performance_stats["last_update"] = datetime.now().isoformat()
        
        # 组合最终结果
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "analysis_summary": {
                "kronos_available": self.kronos_available,
                "fingpt_available": self.fingpt_available,
                "signal_integrated": kronos_signal is not None or sentiment_analysis is not None
            },
            "kronos_signal": kronos_signal,
            "sentiment_analysis": sentiment_analysis,
            "integration_result": final_decision,
            "trading_recommendation": trading_recommendation,
            "performance_stats": self.performance_stats.copy()
        }
        
        print(f"=== 分析完成 ===")
        print(f"最终决策: {trading_recommendation.get('action', '未知')}")
        print(f"置信度: {trading_recommendation.get('confidence', 0):.2f}")
        
        return result
    
    def _integrate_signals(self, 
                          kronos_signal: Optional[Dict], 
                          sentiment_analysis: Optional[Dict]) -> Dict:
        """整合Kronos信号和舆情信号
        
        Args:
            kronos_signal: Kronos技术信号
            sentiment_analysis: 舆情分析结果
            
        Returns:
            整合后的信号
        """
        integration_result = {
            "kronos_signal_present": kronos_signal is not None,
            "sentiment_analysis_present": sentiment_analysis is not None,
            "signal_filtered": False,
            "filter_reason": None,
            "filtered_by_sentiment": False,
            "filtered_by_risk": False,
            "filtered_by_strength": False,
            "final_signal": None,
            "integration_method": "加权融合",
            "integration_timestamp": datetime.now().isoformat()
        }
        
        # 如果没有Kronos信号，无法生成有效决策
        if not kronos_signal:
            integration_result.update({
                "signal_filtered": True,
                "filter_reason": "Kronos信号缺失",
                "final_signal": None
            })
            return integration_result
        
        # 检查信号强度是否满足最小要求
        trend_strength = kronos_signal.get("trend_strength", 0.0)
        min_strength = self.config.get("min_signal_strength", 0.0025)
        if trend_strength < min_strength:
            integration_result.update({
                "signal_filtered": True,
                "filter_reason": f"信号强度过低 ({trend_strength:.4f} < {min_strength:.4f})",
                "filtered_by_strength": True,
                "final_signal": None,
                "integration_method": "信号强度过滤"
            })
            print(f"    ⚠️  信号被过滤: 信号强度过低 ({trend_strength:.4f} < {min_strength:.4f})")
            return integration_result
        
        # 复制Kronos信号作为基础，保留原始信号有效性
        integrated_signal = kronos_signal.copy()
        
        # 如果没有舆情分析，直接使用Kronos信号
        if not sentiment_analysis:
            integration_result.update({
                "final_signal": integrated_signal,
                "integration_method": "仅Kronos信号"
            })
            return integration_result
        
        # 检查风险等级：HIGH 风险时直接过滤信号
        risk_level = sentiment_analysis.get("risk_level", "LOW")
        
        # 根据风险等级决定是否过滤
        if risk_level == "HIGH":
            integration_result.update({
                "signal_filtered": True,
                "filter_reason": "FinGPT检测到高风险，建议观望",
                "filtered_by_risk": True,
                "final_signal": None,
                "integration_method": "高风险过滤"
            })
            print(f"    ⚠️  信号被过滤: 检测到高风险({risk_level})，建议观望")
            return integration_result
        
        # 使用FinGPT调整信号强度（中低风险时）
        if self.fingpt_analyzer:
            try:
                # 信号融合：结合技术信号和舆情信号（但保留原始信号有效性）
                integrated_signal = self._fuse_signals(integrated_signal, sentiment_analysis)
                integration_result["final_signal"] = integrated_signal
                integration_result["integration_method"] = "加权融合（保留原始信号有效性）"
                
            except Exception as e:
                print(f"信号整合失败: {e}")
                integration_result.update({
                    "final_signal": kronos_signal,
                    "integration_method": "整合失败，使用原始信号"
                })
        else:
            integration_result.update({
                "final_signal": kronos_signal,
                "integration_method": "FinGPT不可用，使用原始信号"
            })
        
        # 最后检查：融合后的信号强度是否仍满足要求
        final_signal = integration_result.get("final_signal")
        if final_signal:
            final_strength = final_signal.get("trend_strength", 0.0)
            min_strength = self.config.get("min_signal_strength", 0.0025)
            if final_strength < min_strength:
                integration_result.update({
                    "signal_filtered": True,
                    "filter_reason": f"融合后信号强度过低 ({final_strength:.4f} < {min_strength:.4f})",
                    "filtered_by_strength": True,
                    "final_signal": None,
                    "integration_method": "融合后信号强度过滤"
                })
                print(f"    ⚠️  信号被过滤: 融合后信号强度过低 ({final_strength:.4f} < {min_strength:.4f})")
        
        return integration_result
    
    def _fuse_signals(self, 
                     technical_signal: Dict, 
                     sentiment_analysis: Dict) -> Dict:
        """融合技术信号和舆情信号
        
        Args:
            technical_signal: 技术分析信号
            sentiment_analysis: 舆情分析结果
            
        Returns:
            融合后的信号
        """
        fused_signal = technical_signal.copy()
        
        # 提取关键指标
        tech_strength = technical_signal.get("trend_strength", 0.0)
        sentiment_score = sentiment_analysis.get("sentiment_score", 0.0)
        risk_level = sentiment_analysis.get("risk_level", "LOW")
        overall_sentiment = sentiment_analysis.get("overall_sentiment", "NEUTRAL")
        
        # 方向一致性检查
        tech_direction = technical_signal.get("trend_direction", "NEUTRAL")
        sentiment_direction = "BULLISH" if sentiment_score > 0.1 else "BEARISH" if sentiment_score < -0.1 else "NEUTRAL"
        
        direction_consistent = (
            (tech_direction == "LONG" and sentiment_direction in ["BULLISH", "NEUTRAL"]) or
            (tech_direction == "SHORT" and sentiment_direction in ["BEARISH", "NEUTRAL"]) or
            tech_direction == "NEUTRAL"
        )
        
        # 计算融合权重
        sentiment_weight = self.config["sentiment_weight"]
        technical_weight = self.config["technical_weight"]
        
        # 根据风险等级调整权重
        risk_multiplier = {
            "LOW": 1.0,
            "MEDIUM": 0.7,
            "HIGH": 0.3
        }.get(risk_level, 1.0)
        
        # 调整信号强度
        if direction_consistent:
            # 方向一致，增强信号
            sentiment_boost = abs(sentiment_score) * sentiment_weight
            fused_strength = tech_strength * (1 + sentiment_boost)
        else:
            # 方向不一致，减弱信号
            sentiment_penalty = abs(sentiment_score) * sentiment_weight
            fused_strength = tech_strength * (1 - sentiment_penalty)
        
        # 应用风险乘数
        fused_strength *= risk_multiplier
        
        # 更新信号
        fused_signal["trend_strength"] = fused_strength
        fused_signal["original_tech_strength"] = tech_strength
        fused_signal["sentiment_score"] = sentiment_score
        fused_signal["risk_level"] = risk_level
        fused_signal["direction_consistent"] = direction_consistent
        fused_signal["fusion_method"] = "加权融合"
        fused_signal["fusion_weights"] = {
            "technical": technical_weight,
            "sentiment": sentiment_weight,
            "risk_multiplier": risk_multiplier
        }
        fused_signal["fusion_timestamp"] = datetime.now().isoformat()
        
        # 保留原始Kronos的信号有效性，不重新评估
        # 只添加融合信息，但不改变原始信号有效性
        
        return fused_signal
    
    def _generate_trading_recommendation(self, integration_result: Dict) -> Dict:
        """根据整合结果生成交易建议
        
        Args:
            integration_result: 信号整合结果
            
        Returns:
            交易建议
        """
        recommendation = {
            "action": "HOLD",  # HOLD, BUY, SELL
            "confidence": 0.0,
            "position_size": 0.0,
            "risk_level": "LOW",
            "entry_price_range": None,
            "stop_loss": None,
            "take_profit": None,
            "reasoning": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 检查信号是否被过滤
        if integration_result.get("signal_filtered", False):
            recommendation["reasoning"].append(f"信号被过滤: {integration_result.get('filter_reason', '未知原因')}")
            recommendation["action"] = "HOLD"
            recommendation["confidence"] = 0.0
            return recommendation
        
        final_signal = integration_result.get("final_signal")
        if not final_signal:
            recommendation["reasoning"].append("无有效信号")
            return recommendation
        
        # 提取信号信息（保留原始Kronos的signal_valid）
        signal_valid = final_signal.get("signal_valid", True)
        trend_direction = final_signal.get("trend_direction", "NEUTRAL")
        trend_strength = final_signal.get("trend_strength", 0.0)
        risk_level = final_signal.get("risk_level", "LOW")
        
        # 趋势强度归一化：0.002→0.2, 0.01→1.0（提前计算，供后面使用）
        normalized_strength = min(trend_strength / 0.01, 1.0)  # 归一化到0-1范围
        
        # 确定交易动作（统一方向标识：LONG/BUY = 做多，SHORT/SELL = 做空）
        if trend_direction in ["LONG", "BUY"]:
            action = "BUY"
            confidence = normalized_strength  # 用归一化后的值作为置信度
        elif trend_direction in ["SHORT", "SELL"]:
            action = "SELL"
            confidence = normalized_strength  # 用归一化后的值作为置信度
        else:
            action = "HOLD"
            confidence = 0.0
        
        # 计算仓位大小（基于信号强度归一化后的值）
        base_position = normalized_strength * self.config["max_position_size"]
        
        # 根据风险等级调整仓位
        risk_adjustment = {
            "LOW": 1.0,
            "MEDIUM": 0.5,
            "HIGH": 0.1
        }.get(risk_level, 0.1)
        
        position_size = base_position * risk_adjustment
        
        # 生成入场价格范围
        current_price = final_signal.get("current_price")
        pred_support = final_signal.get("pred_support")
        pred_resistance = final_signal.get("pred_resistance")
        
        if current_price and pred_support and pred_resistance:
            if action == "BUY":
                entry_range = [pred_support, current_price * 1.005]
                stop_loss = pred_support * 0.99
                take_profit = pred_resistance * 1.01
            else:  # SELL
                entry_range = [current_price * 0.995, pred_resistance]
                stop_loss = pred_resistance * 1.01
                take_profit = pred_support * 0.99
        else:
            entry_range = None
            stop_loss = None
            take_profit = None
        
        # 构建推理链条
        reasoning = []
        if normalized_strength > 0.5:
            reasoning.append(f"强趋势信号 (归一化强度: {normalized_strength:.2f})")
        elif normalized_strength > 0.2:
            reasoning.append(f"中等趋势信号 (归一化强度: {normalized_strength:.2f})")
        else:
            reasoning.append(f"弱趋势信号 (归一化强度: {normalized_strength:.2f})")
        
        if risk_level == "LOW":
            reasoning.append("低风险环境")
        elif risk_level == "MEDIUM":
            reasoning.append("中等风险，谨慎交易")
        else:
            reasoning.append("高风险，建议观望")
        
        # 检查拐点信息
        if final_signal.get("has_turning_point", False):
            recent_turn_type = final_signal.get("recent_turn_type")
            if recent_turn_type == "PEAK" and action == "SELL":
                reasoning.append("近期检测到峰值，支持卖出信号")
            elif recent_turn_type == "VALLEY" and action == "BUY":
                reasoning.append("近期检测到谷值，支持买入信号")
        
        # 组合最终建议
        recommendation.update({
            "action": action,
            "confidence": confidence,
            "position_size": position_size,
            "risk_level": risk_level,
            "entry_price_range": entry_range,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning,
            "signal_strength": trend_strength,
            "trend_direction": trend_direction
        })
        
        return recommendation
    
    def get_combined_signal(self, market_data: pd.DataFrame = None, kronos_signal: Dict = None) -> Dict:
        """获取综合信号（兼容接口）
        
        接受预计算的Kronos信号，避免循环调用
        
        Args:
            market_data: 市场K线数据（可选，如果提供了kronos_signal则不需要）
            kronos_signal: 预计算的Kronos信号（可选，避免重复计算）
            
        Returns:
            标准化的综合信号
        """
        # 如果没有提供Kronos信号，且Kronos分析器可用，则计算
        if kronos_signal is None and self.kronos_available and self.kronos_analyzer and market_data is not None:
            try:
                print("  1. Kronos技术分析...")
                kronos_signal = self.kronos_analyzer.get_enhanced_signal(market_data)
                if kronos_signal:
                    print(f"    - 方向: {kronos_signal.get('trend_direction', '未知')}")
                    print(f"    - 强度: {kronos_signal.get('trend_strength', 0):.3f}")
            except Exception as e:
                print(f"    ✗ Kronos分析失败: {e}")
                kronos_signal = None
        
        # FinGPT舆情分析
        sentiment_analysis = None
        if self.fingpt_available and self.fingpt_analyzer:
            try:
                print("  2. FinGPT舆情分析...")
                sentiment_analysis = self.fingpt_analyzer.analyze_market_sentiment(self.symbol)
                if sentiment_analysis:
                    print(f"    - 情绪: {sentiment_analysis.get('overall_sentiment', '未知')}")
                    print(f"    - 风险等级: {sentiment_analysis.get('risk_level', '未知')}")
            except Exception as e:
                print(f"    ✗ FinGPT分析失败: {e}")
                sentiment_analysis = None
        
        # 信号整合与过滤
        print("  3. 信号整合与过滤...")
        final_decision = self._integrate_signals(kronos_signal, sentiment_analysis)
        
        # 生成交易建议
        print("  4. 生成交易建议...")
        trading_recommendation = self._generate_trading_recommendation(final_decision)
        
        # 转换为标准格式
        integration = final_decision
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'kronos_signal': kronos_signal.get('trend_direction', 'NEUTRAL') if kronos_signal else 'NEUTRAL',
            'final_signal': trading_recommendation.get('action', 'HOLD'),
            'signal_strength': kronos_signal.get('trend_strength', 0) if kronos_signal else 0,
            'recommendation': '; '.join(trading_recommendation.get('reasoning', [])) if trading_recommendation.get('reasoning') else '正常交易',
            'sentiment': sentiment_analysis,
            'filtered': integration.get('signal_filtered', False),
            'filter_reason': integration.get('filter_reason', ''),
            'analysis_summary': {
                'kronos_available': kronos_signal is not None,
                'fingpt_available': sentiment_analysis is not None
            }
        }
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "modules": {
                "kronos": {
                    "available": self.kronos_available,
                    "model_name": self.kronos_analyzer.model_name if self.kronos_analyzer else None
                },
                "fingpt": {
                    "available": self.fingpt_available,
                    "use_local_model": self.fingpt_analyzer.use_local_model if self.fingpt_analyzer else False
                }
            },
            "config": self.config,
            "performance": self.performance_stats,
            "memory_usage": "待集成",  # 可添加内存使用监控
            "last_analysis_time": self.performance_stats["last_update"]
        }
    
    def update_config(self, new_config: Dict):
        """更新配置参数
        
        Args:
            new_config: 新的配置字典
        """
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
                print(f"配置更新: {key} = {value}")
    
    def reset_statistics(self):
        """重置性能统计"""
        self.performance_stats = {
            "total_signals": 0,
            "filtered_signals": 0,
            "sentiment_filtered": 0,
            "risk_filtered": 0,
            "strength_filtered": 0,
            "last_update": datetime.now().isoformat()
        }
        print("性能统计已重置")


# 使用示例
if __name__ == "__main__":
    # 创建策略协调器
    coordinator = StrategyCoordinator(
        kronos_model_name="kronos-small",
        use_fingpt=True,
        symbol="BTC"
    )
    
    # 获取系统状态
    status = coordinator.get_system_status()
    print(f"系统状态: Kronos={status['modules']['kronos']['available']}, "
          f"FinGPT={status['modules']['fingpt']['available']}")
    
    # 创建示例市场数据
    print("\n正在生成示例市场数据...")
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
    result = coordinator.analyze_market(sample_data)
    
    # 显示结果摘要
    print(f"\n=== 分析结果摘要 ===")
    recommendation = result.get("trading_recommendation", {})
    print(f"交易动作: {recommendation.get('action', '未知')}")
    print(f"置信度: {recommendation.get('confidence', 0):.2f}")
    print(f"仓位建议: {recommendation.get('position_size', 0):.3f}")
    print(f"风险等级: {recommendation.get('risk_level', '未知')}")
    
    if recommendation.get("reasoning"):
        print(f"推理: {'; '.join(recommendation['reasoning'])}")
    
    # 显示性能统计
    stats = result.get("performance_stats", {})
    print(f"\n性能统计:")
    print(f"总信号数: {stats.get('total_signals', 0)}")
    print(f"过滤信号数: {stats.get('filtered_signals', 0)}")
    print(f"舆情过滤: {stats.get('sentiment_filtered', 0)}")
    print(f"风险过滤: {stats.get('risk_filtered', 0)}")