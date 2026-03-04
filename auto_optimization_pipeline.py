#!/usr/bin/env python
"""自动化优化管道 - 基于交易表现自动触发策略优化"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import warnings
warnings.filterwarnings("ignore")

# 导入现有模块
try:
    from backtest_engine import BacktestEngine
    from strategy_optimizer import StrategyOptimizer
    from qwen3_optimizer import Qwen3Optimizer
    from binance_api import BinanceAPI
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有依赖模块已正确安装")

class AutoOptimizationPipeline:
    """自动化优化管道 - 处理阈值突破事件并执行完整优化流程"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 config_path: str = "strategy_config.py",
                 qwen_model_name: str = "Qwen/Qwen3.5-0.8B-Instruct",  # Qwen3.5-0.8B参数，更轻量快速
                 enable_qwen: bool = True,
                 optimization_history_file: str = "auto_optimization_history.json"):
        """
        初始化自动化优化管道
        
        Args:
            symbol: 交易品种
            config_path: 策略配置文件路径
            qwen_model_name: Qwen3模型名称
            enable_qwen: 是否启用Qwen3优化
            optimization_history_file: 优化历史记录文件
        """
        self.symbol = symbol
        self.config_path = config_path
        self.qwen_model_name = qwen_model_name
        self.enable_qwen = enable_qwen
        self.history_file = optimization_history_file
        
        # 优化组件
        self.backtest_engine = None
        self.strategy_optimizer = None
        self.qwen_optimizer = None
        self.binance_api = None
        
        # 优化状态
        self.is_optimizing = False
        self.optimization_thread = None
        self.last_optimization_time = None
        self.optimization_count = 0
        
        # 优化历史
        self.optimization_history = []
        
        # 回调函数
        self.on_optimization_start = None
        self.on_optimization_progress = None
        self.on_optimization_complete = None
        self.on_optimization_error = None
        
        # 初始化组件
        self._initialize_components()
        
        print(f"自动化优化管道初始化完成: {symbol}")
        print(f"Qwen3优化: {'启用' if enable_qwen else '禁用'}")
    
    def _initialize_components(self):
        """初始化优化组件"""
        try:
            print("  初始化币安API...")
            self.binance_api = BinanceAPI()
            
            print("  初始化策略优化器...")
            self.strategy_optimizer = StrategyOptimizer(self.config_path)
            
            if self.enable_qwen:
                print(f"  初始化Qwen3优化器 ({self.qwen_model_name})...")
                try:
                    # 兼容 PyInstaller 打包环境
                    if getattr(sys, 'frozen', False):
                        base_dir = os.path.dirname(sys.executable)
                    else:
                        base_dir = os.path.dirname(__file__)
                    
                    self.qwen_optimizer = Qwen3Optimizer(
                        model_path=os.path.join(base_dir, "models", "Qwen3.5-0.8B-Instruct"),
                        device=None,  # 自动选择
                        max_length=2048
                    )
                    if not self.qwen_optimizer.is_loaded:
                        print("  ⚠️ Qwen3模型加载失败: 模型未在本地缓存")
                        print(f"  模型名称: {self.qwen_model_name}")
                        print(f"  下载命令: huggingface-cli download Qwen/Qwen3.5-0.8B-Instruct --local-dir models/Qwen3.5-0.8B-Instruct")
                        print("  ⚠️ 将使用传统优化器（无AI优化）")
                        self.enable_qwen = False
                except Exception as e:
                    print(f"  ⚠️ Qwen3初始化失败: {e}")
                    self.enable_qwen = False
            
            print("  ✓ 优化组件初始化完成")
            
        except Exception as e:
            print(f"  ✗ 组件初始化失败: {e}")
            raise
    
    def trigger_optimization(self, 
                            threshold_breaches: List[str] = None,
                            performance_data: Dict = None,
                            force_optimization: bool = False):
        """触发自动化优化流程
        
        Args:
            threshold_breaches: 阈值突破列表
            performance_data: 当前性能数据
            force_optimization: 强制优化（忽略冷却时间）
        """
        if self.is_optimizing:
            print("⚠️ 优化正在进行中，请等待完成")
            return False
        
        # 检查冷却时间（最小间隔1小时）
        if not force_optimization and self.last_optimization_time:
            time_since_last = datetime.now() - self.last_optimization_time
            if time_since_last < timedelta(hours=1):
                print(f"⚠️ 距离上次优化仅 {time_since_last.seconds//60} 分钟，跳过本次优化")
                return False
        
        # 启动优化线程
        self.is_optimizing = True
        self.optimization_thread = threading.Thread(
            target=self._run_optimization_pipeline,
            args=(threshold_breaches, performance_data),
            daemon=True
        )
        self.optimization_thread.start()
        
        print("🚀 自动化优化流程已启动")
        return True
    
    def _run_optimization_pipeline(self, threshold_breaches: List[str], performance_data: Dict):
        """运行完整的优化管道"""
        try:
            # 触发优化开始回调
            if self.on_optimization_start:
                self.on_optimization_start(threshold_breaches, performance_data)
            
            print("\n" + "="*80)
            print("自动化优化管道开始运行")
            print("="*80)
            
            # 记录优化触发原因
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "threshold_breaches": threshold_breaches or ["手动触发"],
                "performance_data": performance_data,
                "steps": []
            }
            
            # 步骤1: 收集历史数据
            step1_result = self._step1_collect_data()
            if not step1_result["success"]:
                raise Exception(f"数据收集失败: {step1_result.get('error')}")
            
            optimization_record["steps"].append(step1_result)
            print(f"✓ 步骤1完成: {step1_result.get('description')}")
            
            # 步骤2: 运行回测分析
            step2_result = self._step2_run_backtest(step1_result["data"])
            if not step2_result["success"]:
                raise Exception(f"回测失败: {step2_result.get('error')}")
            
            optimization_record["steps"].append(step2_result)
            print(f"✓ 步骤2完成: {step2_result.get('description')}")
            
            # 步骤3: 传统参数优化
            step3_result = self._step3_traditional_optimization(step2_result["backtest_results"])
            if not step3_result["success"]:
                raise Exception(f"传统优化失败: {step3_result.get('error')}")
            
            optimization_record["steps"].append(step3_result)
            print(f"✓ 步骤3完成: {step3_result.get('description')}")
            
            # 步骤4: AI优化（如果启用）
            step4_result = None
            if self.enable_qwen and self.qwen_optimizer and self.qwen_optimizer.is_loaded:
                step4_result = self._step4_ai_optimization(step2_result["backtest_results"])
                if step4_result and step4_result.get("success"):
                    optimization_record["steps"].append(step4_result)
                    print(f"✓ 步骤4完成: {step4_result.get('description')}")
                else:
                    print("⚠️ AI优化跳过或失败，继续使用传统优化结果")
            
            # 步骤5: 整合优化结果
            step5_result = self._step5_integrate_results(step3_result, step4_result)
            optimization_record["steps"].append(step5_result)
            print(f"✓ 步骤5完成: {step5_result.get('description')}")
            
            # 步骤6: 生成优化报告
            step6_result = self._step6_generate_report(
                step2_result["backtest_results"],
                step3_result,
                step4_result,
                step5_result
            )
            optimization_record["steps"].append(step6_result)
            optimization_record["final_report"] = step6_result.get("report", {})
            
            # 步骤7: 应用优化参数（可选）
            step7_result = self._step7_apply_parameters(step5_result.get("integrated_parameters", {}))
            if step7_result.get("success"):
                optimization_record["steps"].append(step7_result)
                print(f"✓ 步骤7完成: {step7_result.get('description')}")
            else:
                print(f"⚠️ 参数应用失败: {step7_result.get('error')}")
            
            # 完成优化
            optimization_record["success"] = True
            optimization_record["completion_time"] = datetime.now().isoformat()
            self.optimization_count += 1
            self.last_optimization_time = datetime.now()
            
            # 保存优化历史
            self.optimization_history.append(optimization_record)
            self._save_optimization_history()
            
            print("\n" + "="*80)
            print("✅ 自动化优化管道完成！")
            print(f"总耗时: {optimization_record.get('completion_time')}")
            print(f"优化报告已保存")
            print("="*80)
            
            # 触发优化完成回调
            if self.on_optimization_complete:
                self.on_optimization_complete(optimization_record)
            
        except Exception as e:
            error_msg = f"优化管道运行失败: {e}"
            print(f"❌ {error_msg}")
            
            # 记录错误
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "success": False,
                "error": str(e),
                "threshold_breaches": threshold_breaches
            }
            self.optimization_history.append(optimization_record)
            
            # 触发错误回调
            if self.on_optimization_error:
                self.on_optimization_error(e, optimization_record)
        
        finally:
            self.is_optimizing = False
    
    def _step1_collect_data(self) -> Dict:
        """步骤1: 收集历史数据"""
        try:
            print("\n[步骤1] 收集历史数据...")
            
            if not self.binance_api:
                return {"success": False, "error": "币安API未初始化"}
            
            # 获取最近1000根K线数据
            df = self.binance_api.get_recent_klines(
                self.symbol, 
                "5m",  # 使用5分钟数据
                lookback=1000
            )
            
            if df is None or df.empty:
                return {"success": False, "error": "无法获取历史数据"}
            
            result = {
                "success": True,
                "description": f"收集到 {len(df)} 条K线数据 ({df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]})",
                "data": df,
                "data_points": len(df)
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step2_run_backtest(self, data: pd.DataFrame) -> Dict:
        """步骤2: 运行回测分析"""
        try:
            print("\n[步骤2] 运行回测分析...")
            
            # 初始化回测引擎
            backtest_engine = BacktestEngine(
                symbol=self.symbol,
                initial_capital=10000.0,
                timeframe="5m",
                use_fingpt=True
            )
            
            # 运行回测
            report = backtest_engine.run_backtest(data=data, position_size_pct=0.1)
            
            if "error" in report:
                return {"success": False, "error": report.get("error")}
            
            result = {
                "success": True,
                "description": f"回测完成: {report.get('summary', {}).get('total_trades', 0)} 笔交易，收益率 {report.get('summary', {}).get('total_return_pct', 0):.1f}%",
                "backtest_results": report,
                "summary": report.get("summary", {}),
                "performance_metrics": report.get("performance_metrics", {})
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step3_traditional_optimization(self, backtest_results: Dict) -> Dict:
        """步骤3: 传统参数优化"""
        try:
            print("\n[步骤3] 传统参数优化...")
            
            if not self.strategy_optimizer:
                return {"success": False, "error": "策略优化器未初始化"}
            
            # 使用策略优化器进行优化
            optimization_result = self.strategy_optimizer.optimize_parameters(
                backtest_results=backtest_results,
                target_metrics=["sharpe_ratio", "max_drawdown", "win_rate"]
            )
            
            result = {
                "success": True,
                "description": "传统参数优化完成",
                "optimization_result": optimization_result,
                "recommended_parameters": optimization_result.get("recommended_parameters", {}),
                "optimization_suggestions": optimization_result.get("optimization_suggestions", [])
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step4_ai_optimization(self, backtest_results: Dict) -> Dict:
        """步骤4: AI优化（Qwen3）"""
        try:
            print("\n[步骤4] AI优化分析...")
            
            if not self.qwen_optimizer or not self.qwen_optimizer.is_loaded:
                return {"success": False, "error": "Qwen3优化器不可用"}
            
            # 使用Qwen3分析回测结果
            analysis_result = self.qwen_optimizer.analyze_backtest_results(backtest_results)
            
            if not analysis_result.get("success"):
                return {"success": False, "error": analysis_result.get("error", "AI分析失败")}
            
            # 使用Qwen3优化参数
            optimization_result = self.qwen_optimizer.optimize_parameters(
                backtest_results=backtest_results,
                strategy_type="trend_following",  # 默认策略类型
                target_metric="sharpe_ratio"
            )
            
            result = {
                "success": True,
                "description": "AI优化分析完成",
                "analysis_result": analysis_result,
                "optimization_result": optimization_result,
                "recommended_parameters": optimization_result.get("recommended_parameters", {}),
                "analysis_suggestions": analysis_result.get("analysis", {})
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step5_integrate_results(self, traditional_result: Dict, ai_result: Dict = None) -> Dict:
        """步骤5: 整合优化结果"""
        try:
            print("\n[步骤5] 整合优化结果...")
            
            # 基础推荐参数
            base_params = traditional_result.get("recommended_parameters", {})
            
            # 如果AI优化结果可用，整合AI建议
            integrated_params = base_params.copy()
            
            if ai_result and ai_result.get("success"):
                ai_params = ai_result.get("recommended_parameters", {})
                
                # 合并参数（AI参数优先）
                for key, value in ai_params.items():
                    integrated_params[key] = value
            
            # 添加时间戳
            integrated_params["_last_optimized"] = datetime.now().isoformat()
            integrated_params["_optimization_count"] = self.optimization_count + 1
            
            result = {
                "success": True,
                "description": f"整合完成: {len(integrated_params)} 个参数",
                "integrated_parameters": integrated_params,
                "traditional_parameters": traditional_result.get("recommended_parameters", {}),
                "ai_parameters": ai_result.get("recommended_parameters", {}) if ai_result else {}
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step6_generate_report(self, backtest_results: Dict, 
                              traditional_result: Dict, 
                              ai_result: Dict, 
                              integration_result: Dict) -> Dict:
        """步骤6: 生成优化报告"""
        try:
            print("\n[步骤6] 生成优化报告...")
            
            backtest_summary = backtest_results.get("summary", {})
            
            report = {
                "optimization_report": {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": self.symbol,
                    "backtest_performance": {
                        "total_return_pct": backtest_summary.get("total_return_pct", 0),
                        "max_drawdown_pct": backtest_summary.get("max_drawdown_pct", 0),
                        "sharpe_ratio": backtest_summary.get("sharpe_ratio", 0),
                        "win_rate_pct": backtest_summary.get("win_rate_pct", 0),
                        "total_trades": backtest_summary.get("total_trades", 0)
                    },
                    "traditional_optimization": {
                        "suggestions": traditional_result.get("optimization_suggestions", []),
                        "recommended_parameters": traditional_result.get("recommended_parameters", {})
                    },
                    "ai_optimization": {
                        "available": ai_result is not None and ai_result.get("success"),
                        "suggestions": ai_result.get("analysis_suggestions", {}) if ai_result else {},
                        "recommended_parameters": ai_result.get("recommended_parameters", {}) if ai_result else {}
                    },
                    "final_recommendations": {
                        "integrated_parameters": integration_result.get("integrated_parameters", {}),
                        "summary": self._generate_summary_text(backtest_summary, integration_result)
                    }
                }
            }
            
            # 保存报告到文件
            report_filename = f"auto_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            result = {
                "success": True,
                "description": f"优化报告已保存到 {report_filename}",
                "report": report,
                "report_file": report_filename
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _step7_apply_parameters(self, parameters: Dict) -> Dict:
        """步骤7: 应用优化参数（可选步骤）"""
        try:
            print("\n[步骤7] 应用优化参数...")
            
            if not parameters:
                return {"success": False, "error": "无参数可应用"}
            
            # 这里应该将参数应用到策略配置中
            # 当前实现仅保存参数到文件，实际集成需要根据具体策略框架
            
            # 保存参数到配置文件
            params_filename = f"optimized_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(params_filename, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=2, ensure_ascii=False)
            
            # TODO: 实际集成到策略中需要根据具体框架实现
            # 例如：更新strategy_config.py或专业策略的参数
            
            result = {
                "success": True,
                "description": f"优化参数已保存到 {params_filename}",
                "parameters_file": params_filename,
                "parameters": parameters,
                "note": "参数需要手动集成到策略中"
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_summary_text(self, backtest_summary: Dict, integration_result: Dict) -> str:
        """生成优化摘要文本"""
        summary = []
        summary.append("=== 自动化优化报告 ===")
        summary.append("")
        summary.append("【回测表现】")
        summary.append(f"总收益率: {backtest_summary.get('total_return_pct', 0):.1f}%")
        summary.append(f"最大回撤: {backtest_summary.get('max_drawdown_pct', 0):.1f}%")
        summary.append(f"夏普比率: {backtest_summary.get('sharpe_ratio', 0):.2f}")
        summary.append(f"胜率: {backtest_summary.get('win_rate_pct', 0):.1f}%")
        summary.append("")
        summary.append("【优化建议】")
        
        params = integration_result.get("integrated_parameters", {})
        if params:
            summary.append("推荐参数调整:")
            for key, value in params.items():
                if not key.startswith('_'):  # 跳过内部字段
                    summary.append(f"  • {key}: {value}")
        else:
            summary.append("无参数调整建议")
        
        summary.append("")
        summary.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(summary)
    
    def _save_optimization_history(self):
        """保存优化历史到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"保存优化历史失败: {e}")
    
    def load_optimization_history(self):
        """从文件加载优化历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.optimization_history = json.load(f)
                print(f"加载优化历史: {len(self.optimization_history)}条记录")
        except Exception as e:
            print(f"加载优化历史失败: {e}")
            self.optimization_history = []
    
    def set_callback(self, callback_type: str, callback_func: Callable):
        """设置回调函数"""
        if callback_type == "optimization_start":
            self.on_optimization_start = callback_func
            print("优化开始回调已设置")
        elif callback_type == "optimization_progress":
            self.on_optimization_progress = callback_func
            print("优化进度回调已设置")
        elif callback_type == "optimization_complete":
            self.on_optimization_complete = callback_func
            print("优化完成回调已设置")
        elif callback_type == "optimization_error":
            self.on_optimization_error = callback_func
            print("优化错误回调已设置")
        else:
            print(f"未知回调类型: {callback_type}")
    
    def get_optimization_status(self) -> Dict:
        """获取优化状态"""
        return {
            "is_optimizing": self.is_optimizing,
            "optimization_count": self.optimization_count,
            "last_optimization_time": self.last_optimization_time,
            "enable_qwen": self.enable_qwen,
            "history_count": len(self.optimization_history)
        }
    
    def get_recent_optimizations(self, limit: int = 5) -> List[Dict]:
        """获取最近优化记录"""
        return self.optimization_history[-limit:] if self.optimization_history else []


# 测试函数
if __name__ == "__main__":
    print("测试自动化优化管道...")
    
    try:
        # 创建优化管道
        pipeline = AutoOptimizationPipeline(
            symbol="BTCUSDT",
            enable_qwen=False,  # 测试时禁用Qwen3
            optimization_history_file="test_optimization_history.json"
        )
        
        # 获取优化状态
        status = pipeline.get_optimization_status()
        print(f"优化状态: {status}")
        
        # 测试触发优化
        print("\n触发测试优化...")
        success = pipeline.trigger_optimization(
            threshold_breaches=["测试: 夏普比率过低"],
            performance_data={"sharpe_ratio": 0.8, "win_rate": 0.4},
            force_optimization=True
        )
        
        if success:
            print("优化已触发，等待完成...")
            
            # 等待优化完成（测试环境）
            import time
            max_wait = 120  # 最多等待2分钟
            wait_count = 0
            
            while pipeline.is_optimizing and wait_count < max_wait:
                time.sleep(1)
                wait_count += 1
                if wait_count % 10 == 0:
                    print(f"等待优化完成... ({wait_count}秒)")
            
            if not pipeline.is_optimizing:
                print("✅ 优化测试完成")
                
                # 获取最近优化记录
                recent = pipeline.get_recent_optimizations(1)
                if recent:
                    print(f"优化记录: {recent[-1].get('success', False)}")
            else:
                print("⚠️ 优化超时")
        else:
            print("❌ 优化触发失败")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("\n✅ 自动化优化管道测试完成")