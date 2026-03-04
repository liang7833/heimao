#!/usr/bin/env python
"""策略优化器 - 基于回测结果调整风险收益参数"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class StrategyOptimizer:
    """基于回测结果的策略优化器"""
    
    def __init__(self, config_path: str = "strategy_config.py"):
        """
        初始化策略优化器
        
        Args:
            config_path: 策略配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.optimization_history = []
        
        print("策略优化器初始化完成")
    
    def _load_config(self) -> dict:
        """加载策略配置"""
        try:
            # 简单解析Python配置文件
            config_dict = {}
            with open(self.config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            # 尝试解析为Python对象
                            config_dict[key] = eval(value)
                        except:
                            config_dict[key] = value
            
            return config_dict
        except Exception as e:
            print(f"配置加载失败: {e}")
            return {}
    
    def optimize_parameters(self, 
                           backtest_results: Dict,
                           target_metrics: List[str] = None) -> Dict:
        """
        基于回测结果优化策略参数
        
        Args:
            backtest_results: 回测结果
            target_metrics: 目标指标列表
            
        Returns:
            优化后的参数建议
        """
        if not backtest_results:
            return {"error": "无效的回测结果"}
        
        summary = backtest_results.get("summary", {})
        metrics = backtest_results.get("performance_metrics", {})
        
        # 目标指标
        if target_metrics is None:
            target_metrics = ["sharpe_ratio", "max_drawdown", "win_rate"]
        
        # 分析当前表现
        current_performance = self._analyze_performance(summary, metrics)
        
        # 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(
            current_performance, target_metrics
        )
        
        # 计算推荐参数
        recommended_params = self._calculate_recommended_params(
            current_performance, optimization_suggestions
        )
        
        result = {
            "current_performance": current_performance,
            "optimization_suggestions": optimization_suggestions,
            "recommended_parameters": recommended_params,
            "timestamp": datetime.now().isoformat()
        }
        
        # 记录优化历史
        self.optimization_history.append(result)
        
        return result
    
    def _analyze_performance(self, summary: Dict, metrics: Dict) -> Dict:
        """分析当前表现"""
        performance = {
            "profitability": {
                "total_return": summary.get("total_return_pct", 0),
                "annual_return": metrics.get("annual_return", 0) * 100,
                "final_equity": summary.get("final_equity", 0),
                "initial_capital": summary.get("initial_capital", 0)
            },
            "risk": {
                "max_drawdown": abs(metrics.get("max_drawdown", 0)) * 100,
                "volatility": metrics.get("volatility", 0) * 100,
                "risk_level": self._assess_risk_level(metrics)
            },
            "trading": {
                "total_trades": summary.get("total_trades", 0),
                "win_rate": summary.get("win_rate_pct", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "avg_win": metrics.get("avg_win", 0),
                "avg_loss": metrics.get("avg_loss", 0)
            },
            "risk_adjusted": {
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0)
            }
        }
        
        return performance
    
    def _assess_risk_level(self, metrics: Dict) -> str:
        """评估风险等级"""
        max_dd = abs(metrics.get("max_drawdown", 0))
        
        if max_dd < 0.05:
            return "LOW"
        elif max_dd < 0.10:
            return "MEDIUM"
        elif max_dd < 0.15:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _generate_optimization_suggestions(self, 
                                          performance: Dict,
                                          target_metrics: List[str]) -> List[Dict]:
        """生成优化建议"""
        suggestions = []
        
        # 收益率优化建议
        if performance["profitability"]["total_return"] < 10:
            suggestions.append({
                "category": "profitability",
                "priority": "HIGH",
                "issue": "总收益率低于10%",
                "suggestion": "考虑增加仓位比例或优化入场策略",
                "recommended_action": "增加position_size_pct参数"
            })
        
        # 风险优化建议
        if performance["risk"]["max_drawdown"] > 10:
            suggestions.append({
                "category": "risk",
                "priority": "CRITICAL",
                "issue": f"最大回撤{performance['risk']['max_drawdown']:.1f}%过高",
                "suggestion": "降低单笔交易风险或优化止损策略",
                "recommended_action": "减少single_trade_risk参数至0.005或更低"
            })
        
        # 胜率优化建议
        if performance["trading"]["win_rate"] < 50:
            suggestions.append({
                "category": "trading",
                "priority": "HIGH",
                "issue": f"胜率{performance['trading']['win_rate']:.1f}%偏低",
                "suggestion": "优化入场条件或增加过滤器",
                "recommended_action": "提高TREND_STRENGTH_THRESHOLD参数"
            })
        
        # 夏普比率优化建议
        if performance["risk_adjusted"]["sharpe_ratio"] < 1.0:
            suggestions.append({
                "category": "risk_adjusted",
                "priority": "MEDIUM",
                "issue": f"夏普比率{performance['risk_adjusted']['sharpe_ratio']:.2f}偏低",
                "suggestion": "优化风险收益比或降低波动率",
                "recommended_action": "调整TAKE_PROFIT和STOP_LOSS参数"
            })
        
        # 盈亏比优化建议
        if performance["trading"]["profit_factor"] < 1.2:
            suggestions.append({
                "category": "trading",
                "priority": "MEDIUM",
                "issue": f"盈亏比{performance['trading']['profit_factor']:.2f}偏低",
                "suggestion": "增加止盈倍数或减少止损幅度",
                "recommended_action": "调整TAKE_PROFIT倍数参数"
            })
        
        return suggestions
    
    def _calculate_recommended_params(self, 
                                     performance: Dict,
                                     suggestions: List[Dict]) -> Dict:
        """计算推荐参数"""
        recommended = {}
        
        # 根据建议计算推荐参数
        for suggestion in suggestions:
            category = suggestion.get("category")
            action = suggestion.get("recommended_action", "")
            
            if "position_size_pct" in action:
                # 调整仓位比例
                if performance["risk"]["max_drawdown"] > 10:
                    recommended["position_size_pct"] = 0.05  # 降低到5%
                else:
                    recommended["position_size_pct"] = 0.10  # 保持10%
            
            if "single_trade_risk" in action:
                # 调整单笔交易风险
                if performance["risk"]["max_drawdown"] > 10:
                    recommended["single_trade_risk"] = 0.005
                else:
                    recommended["single_trade_risk"] = 0.008
            
            if "TREND_STRENGTH_THRESHOLD" in action:
                # 调整趋势强度阈值
                if performance["trading"]["win_rate"] < 50:
                    recommended["TREND_STRENGTH_THRESHOLD"] = 0.012
                else:
                    recommended["TREND_STRENGTH_THRESHOLD"] = 0.008
            
            if "TAKE_PROFIT" in action:
                # 调整止盈参数
                if performance["trading"]["profit_factor"] < 1.2:
                    recommended["take_profit_multiplier"] = 1.025
                else:
                    recommended["take_profit_multiplier"] = 1.015
        
        # 默认推荐值
        if not recommended:
            recommended = {
                "position_size_pct": 0.10,
                "single_trade_risk": 0.008,
                "TREND_STRENGTH_THRESHOLD": 0.008,
                "take_profit_multiplier": 1.015
            }
        
        return recommended
    
    def get_recommendations_summary(self, optimization_result: Dict) -> str:
        """获取优化建议摘要"""
        if not optimization_result:
            return "无优化结果"
        
        suggestions = optimization_result.get("optimization_suggestions", [])
        recommended = optimization_result.get("recommended_parameters", {})
        
        summary_lines = []
        summary_lines.append("=== 策略优化建议 ===")
        summary_lines.append("")
        
        # 优先级排序
        critical_suggestions = [s for s in suggestions if s.get("priority") == "CRITICAL"]
        high_suggestions = [s for s in suggestions if s.get("priority") == "HIGH"]
        medium_suggestions = [s for s in suggestions if s.get("priority") == "MEDIUM"]
        
        if critical_suggestions:
            summary_lines.append("【紧急优化】")
            for s in critical_suggestions:
                summary_lines.append(f"  • {s.get('issue')}: {s.get('suggestion')}")
            summary_lines.append("")
        
        if high_suggestions:
            summary_lines.append("【重要优化】")
            for s in high_suggestions:
                summary_lines.append(f"  • {s.get('issue')}: {s.get('suggestion')}")
            summary_lines.append("")
        
        if medium_suggestions:
            summary_lines.append("【建议优化】")
            for s in medium_suggestions:
                summary_lines.append(f"  • {s.get('issue')}: {s.get('suggestion')}")
            summary_lines.append("")
        
        # 推荐参数
        if recommended:
            summary_lines.append("【推荐参数】")
            for param, value in recommended.items():
                summary_lines.append(f"  • {param}: {value}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def save_recommendations(self, optimization_result: Dict, filepath: str = None):
        """保存优化建议"""
        if filepath is None:
            filepath = f"optimization_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(optimization_result, f, indent=2, ensure_ascii=False)
        
        print(f"优化建议已保存到: {filepath}")
        return filepath
    
    def get_history(self) -> List[Dict]:
        """获取优化历史"""
        return self.optimization_history
    
    def clear_history(self):
        """清空优化历史"""
        self.optimization_history = []
        print("优化历史已清空")


# 使用示例
if __name__ == "__main__":
    print("=== 策略优化器测试 ===")
    
    # 初始化优化器
    optimizer = StrategyOptimizer()
    
    # 模拟回测结果
    sample_backtest = {
        "summary": {
            "initial_capital": 10000.0,
            "final_equity": 11500.0,
            "total_return_pct": 15.0,
            "max_drawdown_pct": 12.0,
            "total_trades": 25,
            "win_rate_pct": 48.0
        },
        "performance_metrics": {
            "sharpe_ratio": 0.8,
            "sortino_ratio": 1.1,
            "calmar_ratio": 1.25,
            "annual_return": 0.18,
            "volatility": 0.15,
            "max_drawdown": 0.12,
            "profit_factor": 1.1,
            "avg_win": 0.012,
            "avg_loss": -0.008
        }
    }
    
    # 优化参数
    result = optimizer.optimize_parameters(sample_backtest)
    
    # 打印结果
    print("\n当前表现:")
    perf = result.get("current_performance", {})
    print(f"  总收益率: {perf['profitability']['total_return']:.1f}%")
    print(f"  最大回撤: {perf['risk']['max_drawdown']:.1f}%")
    print(f"  胜率: {perf['trading']['win_rate']:.1f}%")
    print(f"  夏普比率: {perf['risk_adjusted']['sharpe_ratio']:.2f}")
    
    print("\n优化建议:")
    for suggestion in result.get("optimization_suggestions", []):
        print(f"  [{suggestion['priority']}] {suggestion['issue']}")
        print(f"      建议: {suggestion['suggestion']}")
    
    print("\n推荐参数:")
    for param, value in result.get("recommended_parameters", {}).items():
        print(f"  {param}: {value}")
    
    # 保存建议
    filepath = optimizer.save_recommendations(result)
    print(f"\n优化建议已保存到: {filepath}")
    
    print("\n=== 测试完成 ===")