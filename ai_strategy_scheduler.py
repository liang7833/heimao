#!/usr/bin/env python
"""AI策略中心调度器 - 定期分析市场并动态优化所有交易参数"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json


class AIStrategyScheduler:
    """AI策略中心调度器 - 定期执行市场分析和参数优化"""
    
    def __init__(self,
                 qwen_optimizer=None,
                 parameter_integrator=None,
                 strategy_coordinator=None,
                 check_interval_minutes: int = 60):
        """
        初始化AI策略调度器
        
        Args:
            qwen_optimizer: Qwen优化器实例
            parameter_integrator: 参数集成器实例
            strategy_coordinator: 策略协调器实例
            check_interval_minutes: 检查间隔（分钟）
        """
        self.qwen_optimizer = qwen_optimizer
        self.parameter_integrator = parameter_integrator
        self.strategy_coordinator = strategy_coordinator
        
        self.check_interval_minutes = check_interval_minutes
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # 优化历史
        self.optimization_history = []
        self.max_history_size = 50
        
        # 回调函数
        self.on_optimization_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # 市场数据缓存
        self.market_conditions = {}
        self.performance_data = {}
        
        print(f"AI策略中心调度器初始化完成 (检查间隔: {check_interval_minutes}分钟)")
    
    def start(self):
        """启动调度器"""
        if self.is_running:
            print("⚠️ 调度器已在运行中")
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="AIStrategyScheduler"
        )
        self.scheduler_thread.start()
        
        print("✅ AI策略中心调度器已启动")
        
        # 立即执行一次优化
        threading.Thread(target=self._run_optimization, daemon=True).start()
        
        return True
    
    def stop(self):
        """停止调度器"""
        if not self.is_running:
            print("⚠️ 调度器未在运行")
            return False
        
        print("正在停止AI策略中心调度器...")
        self.is_running = False
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        print("✅ AI策略中心调度器已停止")
        return True
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 等待下次检查
                if self.stop_event.wait(timeout=self.check_interval_minutes * 60):
                    break
                
                if self.is_running:
                    self._run_optimization()
                    
            except Exception as e:
                print(f"调度器循环错误: {e}")
                if self.on_error:
                    self.on_error(e)
                time.sleep(60)
    
    def _run_optimization(self):
        """执行一次完整的优化流程"""
        print(f"\n{'='*60}")
        print(f"🤖 AI策略中心 - 开始市场分析和参数优化")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            # 检查Qwen优化器是否可用
            if not self.qwen_optimizer or not self.qwen_optimizer.is_loaded:
                print("❌ Qwen优化器未加载，跳过优化")
                return
            
            # 步骤1: 收集市场数据
            print("\n步骤1: 收集市场数据...")
            market_data = self._collect_market_data()
            print("✓ 市场数据已收集")
            
            # 步骤2: 收集性能数据
            print("\n步骤2: 收集性能数据...")
            perf_data = self._collect_performance_data()
            print("✓ 性能数据已收集")
            
            # 步骤3: Qwen全面分析和优化
            print("\n步骤3: Qwen智能分析和优化...")
            result = self.qwen_optimizer.analyze_and_optimize_all_parameters(
                market_conditions=market_data,
                performance_data=perf_data,
                risk_profile="balanced"
            )
            
            if not result.get("success"):
                error = result.get("error", "未知错误")
                print(f"❌ Qwen优化失败: {error}")
                return
            
            print("✓ Qwen优化完成")
            
            # 显示优化结果
            full_params = result.get("full_parameters", {})
            
            if "market_analysis" in full_params:
                print(f"\n市场分析: {full_params['market_analysis']}")
            
            if "optimization_reasoning" in full_params:
                print(f"优化理由: {full_params['optimization_reasoning']}")
            
            # 步骤4: 应用参数
            print("\n步骤4: 应用优化参数...")
            self._apply_optimized_parameters(full_params)
            
            # 记录优化历史
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "market_conditions": market_data,
                "performance_data": perf_data,
                "optimized_parameters": full_params,
                "success": True
            }
            
            self.optimization_history.append(optimization_record)
            
            # 保持历史记录大小
            if len(self.optimization_history) > self.max_history_size:
                self.optimization_history = self.optimization_history[-self.max_history_size:]
            
            # 触发回调
            if self.on_optimization_complete:
                self.on_optimization_complete(optimization_record)
            
            print(f"\n{'='*60}")
            print("✅ AI策略中心 - 优化完成！")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ 优化流程失败: {e}")
            import traceback
            traceback.print_exc()
            
            if self.on_error:
                self.on_error(e)
    
    def _collect_market_data(self) -> Dict:
        """收集市场数据"""
        # 这里应该从实际数据源获取
        # 先使用缓存或默认数据
        if self.market_conditions:
            return self.market_conditions
        
        # 默认市场数据
        return {
            "trend": "sideways",
            "volatility": "medium",
            "liquidity": "high",
            "current_price": 50000.0,
            "price_change_24h": 0.0
        }
    
    def _collect_performance_data(self) -> Dict:
        """收集性能数据"""
        # 这里应该从性能监控器获取
        if self.performance_data:
            return self.performance_data
        
        # 默认性能数据
        return {
            "win_rate": 0.55,
            "profit_factor": 1.3,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
            "total_trades": 100
        }
    
    def _apply_optimized_parameters(self, full_params: Dict):
        """应用优化后的参数"""
        applied_count = 0
        
        # 1. 应用协调器参数
        coordinator_params = full_params.get("coordinator_parameters", {})
        if coordinator_params and self.strategy_coordinator:
            try:
                self.strategy_coordinator.update_config(coordinator_params)
                print(f"  ✓ 协调器参数已应用 ({len(coordinator_params)}个)")
                applied_count += len(coordinator_params)
            except Exception as e:
                print(f"  ✗ 协调器参数应用失败: {e}")
        
        # 2. 应用策略配置参数
        if self.parameter_integrator:
            try:
                # 构建扁平的参数字典
                flat_params = {}
                
                # 基础参数
                basic_params = full_params.get("basic_parameters", {})
                flat_params.update(basic_params)
                
                # 入场过滤
                entry_filter = full_params.get("entry_filter", {})
                for k, v in entry_filter.items():
                    flat_params[k] = v
                
                # 止损
                stop_loss = full_params.get("stop_loss", {})
                for k, v in stop_loss.items():
                    flat_params[k] = v
                
                # 止盈
                take_profit = full_params.get("take_profit", {})
                for k, v in take_profit.items():
                    flat_params[k] = v
                
                # 风险管理
                risk_mgmt = full_params.get("risk_management", {})
                for k, v in risk_mgmt.items():
                    flat_params[k] = v
                
                # 交易频率
                trade_freq = full_params.get("trade_frequency", {})
                for k, v in trade_freq.items():
                    flat_params[k] = v
                
                # 仓位管理
                pos_mgmt = full_params.get("position_management", {})
                for k, v in pos_mgmt.items():
                    flat_params[k] = v
                
                if flat_params:
                    result = self.parameter_integrator.integrate_parameters(flat_params)
                    if result.get("success"):
                        print(f"  ✓ 策略配置参数已应用 ({result.get('parameters_updated', 0)}个)")
                        applied_count += result.get('parameters_updated', 0)
                    else:
                        print(f"  ✗ 策略配置参数应用失败: {result.get('error')}")
            except Exception as e:
                print(f"  ✗ 策略配置参数应用异常: {e}")
        
        return applied_count
    
    def update_market_conditions(self, conditions: Dict):
        """更新市场条件"""
        self.market_conditions = conditions
        print(f"市场条件已更新: {conditions}")
    
    def update_performance_data(self, data: Dict):
        """更新性能数据"""
        self.performance_data = data
        print(f"性能数据已更新: {data}")
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """获取优化历史"""
        return self.optimization_history[-limit:] if self.optimization_history else []
    
    def trigger_optimization_now(self):
        """立即触发一次优化"""
        if self.is_running:
            threading.Thread(target=self._run_optimization, daemon=True).start()
            return True
        return False


# 测试代码
if __name__ == "__main__":
    print("=== AI策略中心调度器测试 ===")
    
    # 创建调度器（不依赖其他模块）
    scheduler = AIStrategyScheduler(check_interval_minutes=1)
    
    print("\n调度器已创建")
    print("按Ctrl+C停止\n")
    
    try:
        scheduler.start()
        
        # 保持主线程运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n正在停止...")
        scheduler.stop()
        print("测试完成")
