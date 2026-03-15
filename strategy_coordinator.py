#!/usr/bin/env python
"""策略协调器 - 整合Kronos预测信号、Qwen分析和FinGPT舆情信号，生成最终交易决策"""

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

try:
    from qwen_analyzer import QwenAnalyzer, get_qwen_analyzer
except ImportError:
    print("警告: qwen_analyzer模块导入失败，部分功能受限")




class StrategyCoordinator:
    """策略协调器 - 整合多源信号，生成优化交易决策"""
    
    def __init__(self, 
                 kronos_model_name: str = "kronos-small",
                 use_fingpt: bool = True,
                 use_qwen: bool = False,
                 symbol: str = "BTC",
                 kronos_analyzer = None,
                 fingpt_analyzer = None,
                 qwen_analyzer = None):
        """
        初始化策略协调器
        
        Args:
            kronos_model_name: Kronos模型名称
            use_fingpt: 是否使用FinGPT舆情分析
            use_qwen: 是否使用Qwen分析
            symbol: 交易品种符号
            kronos_analyzer: 可选的外部Kronos分析器实例（避免循环创建）
            fingpt_analyzer: 可选的外部FinGPT分析器实例（避免循环创建）
            qwen_analyzer: 可选的外部Qwen分析器实例（避免循环创建）
        """
        self.symbol = symbol
        self.use_fingpt = use_fingpt
        self.use_qwen = use_qwen
        
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
        
        # 初始化Qwen分析器（如果外部没有提供）
        if qwen_analyzer is not None:
            print("  使用外部提供的Qwen分析器...")
            self.qwen_analyzer = qwen_analyzer
            self.qwen_available = True
            print("  ✓ Qwen分析器就绪")
        else:
            self.qwen_analyzer = None
            if use_qwen:
                print("  加载Qwen分析器...")
                try:
                    self.qwen_analyzer = get_qwen_analyzer(
                        symbol=symbol,
                        use_local_model=True
                    )
                    self.qwen_available = True
                    print("  ✓ Qwen分析器就绪")
                except Exception as e:
                    print(f"  ✗ Qwen分析器加载失败: {e}")
                    self.qwen_available = False
            else:
                self.qwen_available = False
        
        # 策略配置
        self.config = {
            "min_signal_strength": 0.0025,  # 最小信号强度（与Kronos趋势强度范围匹配：0.002-0.01）
            "max_position_size": 0.1,    # 最大仓位比例
            "kronos_qwen_ratio": 0.7,      # Kronos在Kronos+Qwen融合中的权重（默认0.7，即7:3）
            "tech_fingpt_ratio": 0.8,      # 技术分析在技术+FinGPT融合中的权重（默认0.8，即8:2）
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
        
        print(f"策略协调器初始化完成 (Kronos: {self.kronos_available}, Qwen: {self.qwen_available}, FinGPT: {self.fingpt_available})")
        if self.use_qwen:
            print(f"  ✓ Qwen分析已启用")
            print(f"  ✓ 权重配置: Kronos:Qwen={self.config.get('kronos_qwen_ratio'):.1f}:{1 - self.config.get('kronos_qwen_ratio'):.1f}, 技术:FinGPT={self.config.get('tech_fingpt_ratio'):.1f}:{1 - self.config.get('tech_fingpt_ratio'):.1f}")
        else:
            print(f"  ✗ Qwen分析已禁用")
            print(f"  ✓ 权重配置: 技术:FinGPT={self.config.get('tech_fingpt_ratio'):.1f}:{1 - self.config.get('tech_fingpt_ratio'):.1f}")
    
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
        
        # 步骤2: Qwen技术分析
        qwen_signal = None
        print(f"  [Qwen] 状态检查 - use_qwen={self.use_qwen}, qwen_available={self.qwen_available}")
        if self.use_qwen and self.qwen_available and self.qwen_analyzer:
            try:
                print("  2. Qwen技术分析...")
                qwen_signal = self.qwen_analyzer.get_enhanced_signal(market_data)
                if qwen_signal:
                    print(f"    ✓ Qwen分析成功")
                    print(f"    - 方向: {qwen_signal.get('trend_direction', '未知')}")
                    print(f"    - 强度: {qwen_signal.get('trend_strength', 0):.3f}")
                    print(f"    - 方法: {qwen_signal.get('analysis_method', '未知')}")
                    # 如果Qwen没有止盈止损，使用Kronos的
                    if kronos_signal and qwen_signal:
                        need_update = False
                        if "pred_support" not in qwen_signal or qwen_signal["pred_support"] is None or qwen_signal["pred_support"] == 0:
                            qwen_signal["pred_support"] = kronos_signal.get("pred_support")
                            need_update = True
                        if "pred_resistance" not in qwen_signal or qwen_signal["pred_resistance"] is None or qwen_signal["pred_resistance"] == 0:
                            qwen_signal["pred_resistance"] = kronos_signal.get("pred_resistance")
                            need_update = True
                        if "current_price" not in qwen_signal or qwen_signal["current_price"] is None or qwen_signal["current_price"] == 0:
                            qwen_signal["current_price"] = kronos_signal.get("current_price")
                            need_update = True
                        if need_update:
                            print(f"    ✓ 使用Kronos的止盈止损值")
            except Exception as e:
                print(f"    ✗ Qwen分析失败: {e}")
                import traceback
                traceback.print_exc()
                qwen_signal = None
        else:
            print(f"  ⏭️  跳过Qwen分析 (未启用或不可用)")
        
        # 步骤3: FinGPT舆情分析
        sentiment_analysis = None
        if self.fingpt_available and self.fingpt_analyzer:
            try:
                step_num = 4 if qwen_signal else 3
                print(f"  {step_num}. FinGPT舆情分析...")
                sentiment_analysis = self.fingpt_analyzer.analyze_market_sentiment(self.symbol)
                if sentiment_analysis:
                    print(f"    - 情绪: {sentiment_analysis.get('overall_sentiment', '未知')}")
                    print(f"    - 风险等级: {sentiment_analysis.get('risk_level', '未知')}")
                    print(f"    - 建议: {sentiment_analysis.get('recommendation', '未知')}")
            except Exception as e:
                print(f"    ✗ FinGPT分析失败: {e}")
                sentiment_analysis = None
        
        # 步骤4: 信号整合与过滤
        step_num = 5 if qwen_signal and sentiment_analysis else (4 if (qwen_signal or sentiment_analysis) else 3)
        print(f"  {step_num}. 信号整合与过滤...")
        final_decision = self._integrate_signals(kronos_signal, qwen_signal, sentiment_analysis)
        
        # 步骤5: 生成交易建议
        step_num += 1
        print(f"  {step_num}. 生成交易建议...")
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
                "qwen_available": self.qwen_available,
                "fingpt_available": self.fingpt_available,
                "signal_integrated": kronos_signal is not None or qwen_signal is not None or sentiment_analysis is not None
            },
            "kronos_signal": kronos_signal,
            "qwen_signal": qwen_signal,
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
                          qwen_signal: Optional[Dict] = None,
                          sentiment_analysis: Optional[Dict] = None) -> Dict:
        """整合Kronos信号、Qwen分析和舆情信号
        
        Args:
            kronos_signal: Kronos技术信号
            qwen_signal: Qwen分析信号
            sentiment_analysis: 舆情分析结果
            
        Returns:
            整合后的信号
        """
        integration_result = {
            "kronos_signal_present": kronos_signal is not None,
            "qwen_signal_present": qwen_signal is not None,
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
        
        # 检查风险等级：HIGH 风险时直接过滤信号
        if sentiment_analysis:
            risk_level = sentiment_analysis.get("risk_level", "LOW")
            if risk_level == "HIGH":
                integration_result.update({
                    "signal_filtered": True,
                    "filter_reason": "FinGPT检测到高风险，建议观望",
                    "filtered_by_risk": True,
                    "integration_method": "高风险过滤"
                })
                print(f"    ⚠️  信号被过滤: 检测到高风险({risk_level})，建议观望")
                return integration_result
        
        # 三模型加权融合（先融合再检查强度）
        try:
            integrated_signal = self._fuse_signals(kronos_signal, qwen_signal, sentiment_analysis)
            integration_result["final_signal"] = integrated_signal
            integration_result["integration_method"] = "三模型加权融合"
        except Exception as e:
            print(f"信号融合失败: {e}")
            integration_result.update({
                "final_signal": kronos_signal,
                "integration_method": "融合失败，使用Kronos原始信号"
            })
        
        # 检查融合后的信号强度是否满足要求
        final_signal = integration_result.get("final_signal")
        if final_signal:
            final_strength = final_signal.get("trend_strength", 0.0)
            min_strength = self.config.get("min_signal_strength", 0.0025)
            if final_strength < min_strength:
                integration_result.update({
                    "signal_filtered": True,
                    "filter_reason": f"融合后信号强度过低 ({final_strength:.4f} < {min_strength:.4f})",
                    "filtered_by_strength": True,
                    "integration_method": "融合后信号强度过滤"
                })
                print(f"    ⚠️  信号被过滤: 融合后信号强度过低 ({final_strength:.4f} < {min_strength:.4f})")
        
        return integration_result
    
    def _fuse_signals(self, 
                     kronos_signal: Dict, 
                     qwen_signal: Optional[Dict] = None,
                     sentiment_analysis: Optional[Dict] = None) -> Dict:
        """融合信号（使用配置的权重）
        
        Qwen启用时: 
            第一步: Kronos + Qwen (使用kronos_qwen_ratio)
            第二步: 第一步结果 + FinGPT (使用tech_fingpt_ratio)
        
        Qwen不启用时:
            Kronos + FinGPT (使用tech_fingpt_ratio)
        
        Args:
            kronos_signal: Kronos技术分析信号
            qwen_signal: Qwen分析信号
            sentiment_analysis: FinGPT舆情分析结果
            
        Returns:
            融合后的信号
        """
        fused_signal = kronos_signal.copy()
        
        # 从配置获取权重
        kronos_qwen_ratio = self.config.get("kronos_qwen_ratio", 0.7)
        tech_fingpt_ratio = self.config.get("tech_fingpt_ratio", 0.8)
        
        # 将方向转换为数值信号（LONG=1, SHORT=-1, NEUTRAL=0）
        def direction_to_value(direction):
            if direction == "LONG":
                return 1
            elif direction == "SHORT":
                return -1
            else:
                return 0
        
        # 第一步：Kronos + Qwen（使用kronos_qwen_ratio）
        kronos_strength = kronos_signal.get("trend_strength", 0.0)
        kronos_direction = kronos_signal.get("trend_direction", "NEUTRAL")
        kronos_dir_value = direction_to_value(kronos_direction)
        # 将强度乘以方向值：LONG为正，SHORT为负，NEUTRAL为0
        kronos_signed_strength = kronos_strength * kronos_dir_value
        print(f"    [Kronos] 信号: {kronos_direction} (强度: {kronos_strength:.4f}, 带符号强度: {kronos_signed_strength:.4f})")
        
        step1_signed_strength = kronos_signed_strength
        step1_direction_value = kronos_dir_value
        
        if qwen_signal and qwen_signal.get("signal_valid", True):
            qwen_strength = qwen_signal.get("trend_strength", 0.0)
            qwen_direction = qwen_signal.get("trend_direction", "NEUTRAL")
            qwen_dir_value = direction_to_value(qwen_direction)
            # 将强度乘以方向值：LONG为正，SHORT为负，NEUTRAL为0
            qwen_signed_strength = qwen_strength * qwen_dir_value
            print(f"    [Qwen] 信号: {qwen_direction} (强度: {qwen_strength:.4f}, 带符号强度: {qwen_signed_strength:.4f})")
            
            # Kronos:Qwen = kronos_qwen_ratio : (1-kronos_qwen_ratio) 融合（使用带符号强度）
            q_ratio = 1 - kronos_qwen_ratio
            step1_signed_strength = (kronos_signed_strength * kronos_qwen_ratio) + (qwen_signed_strength * q_ratio)
            step1_direction_value = (kronos_dir_value * kronos_qwen_ratio) + (qwen_dir_value * q_ratio)
            # 最终强度取绝对值
            step1_strength = abs(step1_signed_strength)
            print(f"    [第一步] Kronos+Qwen融合({kronos_qwen_ratio:.1f}:{q_ratio:.1f}): 带符号强度={step1_signed_strength:.4f}, 最终强度={step1_strength:.4f}, 方向值={step1_direction_value:.3f}")
        else:
            step1_strength = abs(step1_signed_strength)
        
        # 第二步：第一步结果 + FinGPT（使用tech_fingpt_ratio）
        # 第一步的带符号强度用于第二步融合
        final_signed_strength = step1_signed_strength
        final_direction_value = step1_direction_value
        
        if sentiment_analysis:
            sentiment_score = sentiment_analysis.get("sentiment_score", 0.0)
            risk_level = sentiment_analysis.get("risk_level", "LOW")
            
            # 将FinGPT情绪转换为方向值
            fingpt_direction_value = sentiment_score * 2 - 1  # 0-1 -> -1-1
            # FinGPT的强度：情绪偏离0.5的程度 × 0.01（基准强度）
            fingpt_strength = abs(sentiment_score - 0.5) * 2 * 0.01
            # FinGPT的带符号强度
            fingpt_signed_strength = fingpt_strength * fingpt_direction_value
            
            # 技术分析:FinGPT = tech_fingpt_ratio : (1-tech_fingpt_ratio) 融合（使用带符号强度）
            f_ratio = 1 - tech_fingpt_ratio
            final_signed_strength = (step1_signed_strength * tech_fingpt_ratio) + (fingpt_signed_strength * f_ratio)
            final_direction_value = (step1_direction_value * tech_fingpt_ratio) + (fingpt_direction_value * f_ratio)
            # 最终强度取绝对值
            final_strength = abs(final_signed_strength)
            
            # 根据风险等级调整
            risk_multiplier = {"LOW": 1.0, "MEDIUM": 0.8, "HIGH": 0.5}.get(risk_level, 1.0)
            final_strength = final_strength * risk_multiplier
            
            print(f"    [FinGPT] 情绪: {sentiment_score:.3f}, 风险: {risk_level}, 方向值: {fingpt_direction_value:.3f}, 强度: {fingpt_strength:.4f}, 带符号强度: {fingpt_signed_strength:.4f}")
            print(f"    [第二步] +FinGPT融合({tech_fingpt_ratio:.1f}:{f_ratio:.1f}): 带符号强度={final_signed_strength:.4f}, 最终强度={final_strength:.4f}, 方向值={final_direction_value:.3f}, 风险乘数={risk_multiplier}")
        else:
            final_strength = abs(final_signed_strength)
        
        # 确定最终方向
        if final_direction_value > 0.1:
            final_direction = "LONG"
        elif final_direction_value < -0.1:
            final_direction = "SHORT"
        else:
            final_direction = "NEUTRAL"
        
        # 不强制强度范围，保持融合后的真实值，但确保不小于0
        final_strength = max(0.0, min(final_strength, 0.01))
        
        # ============================================
        # 拐点增强逻辑：利用Qwen的拐点检测
        # ============================================
        has_turning_point = False
        turning_point_boost = False
        turning_point_boost_pct = 0.0
        
        if qwen_signal:
            has_turning_point = qwen_signal.get("has_turning_point", False)
            
            if has_turning_point and final_direction != "NEUTRAL":
                qwen_direction = qwen_signal.get("trend_direction", "NEUTRAL")
                
                # 检查Qwen方向是否与最终融合方向一致
                direction_match = False
                if (final_direction in ["LONG", "BUY"] and qwen_direction in ["LONG", "BUY"]) or \
                   (final_direction in ["SHORT", "SELL"] and qwen_direction in ["SHORT", "SELL"]):
                    direction_match = True
                
                if direction_match:
                    # 方向一致时，增强信号强度30%
                    original_strength = final_strength
                    turning_point_boost_pct = 0.3
                    final_strength = final_strength * (1 + turning_point_boost_pct)
                    final_strength = min(final_strength, 0.01)  # 不超过最大强度
                    turning_point_boost = True
                    print(f"    [拐点增强] Qwen检测到拐点且方向一致，信号强度增强30%: {original_strength:.4f} → {final_strength:.4f}")
        
        # 更新融合信号
        fused_signal["trend_strength"] = final_strength
        fused_signal["trend_direction"] = final_direction
        fused_signal["original_kronos_strength"] = kronos_strength
        fused_signal["fusion_method"] = "两步融合" if qwen_signal else "单步融合"
        fused_signal["fusion_timestamp"] = datetime.now().isoformat()
        fused_signal["has_turning_point"] = has_turning_point
        fused_signal["turning_point_boost"] = turning_point_boost
        fused_signal["turning_point_boost_pct"] = turning_point_boost_pct
        
        print(f"    [融合] 最终信号: {final_direction} (强度: {final_strength:.4f})")
        
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
        
        # Qwen分析
        qwen_signal = None
        if self.use_qwen and self.qwen_available and self.qwen_analyzer and market_data is not None:
            try:
                print("  2. Qwen技术分析...")
                qwen_signal = self.qwen_analyzer.get_enhanced_signal(market_data)
                if qwen_signal:
                    print(f"    - 方向: {qwen_signal.get('trend_direction', '未知')}")
                    print(f"    - 强度: {qwen_signal.get('trend_strength', 0):.3f}")
            except Exception as e:
                print(f"    ✗ Qwen分析失败: {e}")
                qwen_signal = None
        
        # FinGPT舆情分析
        sentiment_analysis = None
        if self.fingpt_available and self.fingpt_analyzer:
            try:
                step_num = 3 if qwen_signal else 2
                print(f"  {step_num}. FinGPT舆情分析...")
                sentiment_analysis = self.fingpt_analyzer.analyze_market_sentiment(self.symbol)
                if sentiment_analysis:
                    print(f"    - 情绪: {sentiment_analysis.get('overall_sentiment', '未知')}")
                    print(f"    - 风险等级: {sentiment_analysis.get('risk_level', '未知')}")
            except Exception as e:
                print(f"    ✗ FinGPT分析失败: {e}")
                sentiment_analysis = None
        
        # 信号整合与过滤
        step_num = 4 if qwen_signal and sentiment_analysis else (3 if (qwen_signal or sentiment_analysis) else 2)
        print(f"  {step_num}. 信号整合与过滤...")
        final_decision = self._integrate_signals(kronos_signal, qwen_signal, sentiment_analysis)
        
        # 生成交易建议
        step_num += 1
        print(f"  {step_num}. 生成交易建议...")
        trading_recommendation = self._generate_trading_recommendation(final_decision)
        
        # 转换为标准格式
        integration = final_decision
        final_signal_data = integration.get('final_signal', {})
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'kronos_signal': kronos_signal.get('trend_direction', 'NEUTRAL') if kronos_signal else 'NEUTRAL',
            'qwen_signal': qwen_signal.get('trend_direction', 'NEUTRAL') if qwen_signal else 'NEUTRAL',
            'final_signal': trading_recommendation.get('action', 'HOLD'),
            'signal_strength': final_signal_data.get('trend_strength', 0) if final_signal_data else (kronos_signal.get('trend_strength', 0) if kronos_signal else 0),
            'recommendation': '; '.join(trading_recommendation.get('reasoning', [])) if trading_recommendation.get('reasoning') else '正常交易',
            'sentiment': sentiment_analysis,
            'filtered': integration.get('signal_filtered', False),
            'filter_reason': integration.get('filter_reason', ''),
            'has_turning_point': final_signal_data.get('has_turning_point', False),
            'turning_point_boost': final_signal_data.get('turning_point_boost', False),
            'turning_point_boost_pct': final_signal_data.get('turning_point_boost_pct', 0.0),
            'analysis_summary': {
                'kronos_available': kronos_signal is not None,
                'qwen_available': qwen_signal is not None,
                'fingpt_available': sentiment_analysis is not None
            }
        }
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        modules_status = {
            "kronos": {
                "available": self.kronos_available,
                "model_name": self.kronos_analyzer.model_name if self.kronos_analyzer else None
            },
            "fingpt": {
                "available": self.fingpt_available,
                "use_local_model": self.fingpt_analyzer.use_local_model if self.fingpt_analyzer else False
            }
        }
        
        if hasattr(self, 'qwen_available'):
            modules_status["qwen"] = {
                "available": self.qwen_available,
                "use_local_model": self.qwen_analyzer.use_local_model if hasattr(self, 'qwen_analyzer') and self.qwen_analyzer else False
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "modules": modules_status,
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