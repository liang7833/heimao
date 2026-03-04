#!/usr/bin/env python
"""参数集成器 - 将优化后的参数自动集成到策略配置中"""

import os
import re
import json
import ast
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")

class ParameterIntegrator:
    """参数集成器 - 更新策略配置文件并通知策略重新加载"""
    
    def __init__(self, 
                 config_file: str = "strategy_config.py",
                 backup_dir: str = "config_backups"):
        """
        初始化参数集成器
        
        Args:
            config_file: 策略配置文件路径
            backup_dir: 配置文件备份目录
        """
        self.config_file = config_file
        self.backup_dir = backup_dir
        
        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)
        
        # 配置参数映射（优化参数名 -> 配置文件变量名）
        self.parameter_mapping = {
            # 趋势阈值
            "TREND_STRENGTH_THRESHOLD": "TREND_STRENGTH_THRESHOLD",
            "trend_strength_threshold": "TREND_STRENGTH_THRESHOLD",
            
            # 仓位管理
            "position_size_pct": "RISK_MANAGEMENT['single_trade_risk']",
            "single_trade_risk": "RISK_MANAGEMENT['single_trade_risk']",
            
            # 止损止盈
            "take_profit_multiplier": "TAKE_PROFIT['tp1_multiplier_long']",  # 简化映射
            "stop_loss_buffer": "STOP_LOSS['long_buffer']",  # 简化映射
            
            # 风险参数
            "max_drawdown": "RISK_MANAGEMENT['daily_loss_limit']",  # 近似映射
            "max_consecutive_losses": "RISK_MANAGEMENT['max_consecutive_losses']",
            
            # 交易频率
            "min_trade_interval_minutes": "TRADE_FREQUENCY['min_trade_interval_minutes']",
            "max_daily_trades": "TRADE_FREQUENCY['max_daily_trades']"
        }
        
        # 集成历史
        self.integration_history = []
        
        print(f"参数集成器初始化完成: {config_file}")
        print(f"备份目录: {backup_dir}")
    
    def backup_config(self) -> str:
        """备份当前配置文件"""
        try:
            if not os.path.exists(self.config_file):
                return "配置文件不存在，无法备份"
            
            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"strategy_config_backup_{timestamp}.py")
            
            # 复制文件
            import shutil
            shutil.copy2(self.config_file, backup_file)
            
            return backup_file
            
        except Exception as e:
            return f"备份失败: {e}"
    
    def integrate_parameters(self, parameters: Dict, strategy_instance=None) -> Dict:
        """集成参数到配置文件
        
        Args:
            parameters: 优化后的参数字典
            strategy_instance: 策略实例（用于动态更新）
            
        Returns:
            集成结果
        """
        try:
            print(f"开始集成 {len(parameters)} 个参数...")
            
            # 备份当前配置
            backup_file = self.backup_config()
            print(f"配置文件已备份: {backup_file}")
            
            # 加载当前配置
            current_config = self._load_config_file()
            
            # 应用参数更新
            update_count = self._apply_parameter_updates(current_config, parameters)
            
            # 保存更新后的配置
            self._save_config_file(current_config)
            
            # 尝试动态更新策略实例（如果提供）
            dynamic_updates = 0
            if strategy_instance:
                dynamic_updates = self._update_strategy_instance(strategy_instance, parameters)
            
            # 记录集成历史
            integration_record = {
                "timestamp": datetime.now().isoformat(),
                "parameters_integrated": parameters,
                "parameters_updated": update_count,
                "dynamic_updates": dynamic_updates,
                "backup_file": backup_file,
                "success": True
            }
            self.integration_history.append(integration_record)
            
            result = {
                "success": True,
                "message": f"参数集成完成: {update_count} 个参数已更新",
                "backup_file": backup_file,
                "parameters_updated": update_count,
                "dynamic_updates": dynamic_updates,
                "integration_record": integration_record
            }
            
            print(f"✅ 参数集成成功: {update_count} 个参数已更新")
            if dynamic_updates > 0:
                print(f"   动态更新: {dynamic_updates} 个参数")
            
            return result
            
        except Exception as e:
            error_msg = f"参数集成失败: {e}"
            print(f"❌ {error_msg}")
            
            result = {
                "success": False,
                "error": error_msg,
                "parameters": parameters
            }
            
            return result
    
    def _load_config_file(self) -> Dict:
        """加载配置文件为字典"""
        config_dict = {}
        
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析Python配置文件
        # 使用ast模块安全地解析
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            try:
                                # 安全地评估值
                                value = ast.literal_eval(node.value)
                                config_dict[var_name] = value
                            except:
                                # 如果无法评估，保留原始字符串
                                value_str = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                                config_dict[var_name] = value_str
        except Exception as e:
            print(f"AST解析失败，使用简单解析: {e}")
            # 使用简单解析
            config_dict = self._simple_parse_config(content)
        
        return config_dict
    
    def _simple_parse_config(self, content: str) -> Dict:
        """简单解析配置文件（当AST解析失败时使用）"""
        config_dict = {}
        
        # 按行解析
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#') or line.startswith('"""'):
                continue
            
            # 尝试匹配变量赋值
            if '=' in line:
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                value_str = parts[1].strip()
                
                # 尝试解析值
                try:
                    # 尝试作为Python表达式解析
                    value = ast.literal_eval(value_str)
                    config_dict[var_name] = value
                except:
                    # 保留为字符串
                    config_dict[var_name] = value_str
        
        return config_dict
    
    def _apply_parameter_updates(self, config_dict: Dict, parameters: Dict) -> int:
        """应用参数更新到配置字典"""
        update_count = 0
        
        for param_name, param_value in parameters.items():
            # 跳过内部字段
            if param_name.startswith('_'):
                continue
            
            # 查找参数映射
            config_var = self._find_config_variable(param_name)
            
            if config_var:
                # 更新配置字典
                updated = self._update_config_value(config_dict, config_var, param_value)
                if updated:
                    update_count += 1
                    print(f"  更新: {param_name} -> {config_var} = {param_value}")
            else:
                print(f"  警告: 未找到参数 {param_name} 的映射")
        
        return update_count
    
    def _find_config_variable(self, parameter_name: str) -> Optional[str]:
        """查找参数对应的配置变量"""
        # 直接匹配
        if parameter_name in self.parameter_mapping:
            return self.parameter_mapping[parameter_name]
        
        # 大小写不敏感匹配
        for key in self.parameter_mapping.keys():
            if key.lower() == parameter_name.lower():
                return self.parameter_mapping[key]
        
        # 尝试猜测映射
        if 'threshold' in parameter_name.lower():
            return 'TREND_STRENGTH_THRESHOLD'
        elif 'position' in parameter_name.lower() and 'size' in parameter_name.lower():
            return 'RISK_MANAGEMENT[\'single_trade_risk\']'
        elif 'risk' in parameter_name.lower():
            return 'RISK_MANAGEMENT[\'single_trade_risk\']'
        elif 'profit' in parameter_name.lower():
            return 'TAKE_PROFIT[\'tp1_multiplier_long\']'
        elif 'loss' in parameter_name.lower():
            return 'STOP_LOSS[\'long_buffer\']'
        
        return None
    
    def _update_config_value(self, config_dict: Dict, config_var: str, value: Any) -> bool:
        """更新配置字典中的值"""
        try:
            # 处理嵌套字典访问（如 RISK_MANAGEMENT['single_trade_risk']）
            if '[' in config_var and ']' in config_var:
                # 解析嵌套访问
                match = re.match(r"(\w+)\['([^']+)'\]", config_var)
                if match:
                    dict_name = match.group(1)
                    key_name = match.group(2)
                    
                    if dict_name in config_dict and isinstance(config_dict[dict_name], dict):
                        config_dict[dict_name][key_name] = value
                        return True
            else:
                # 简单变量赋值
                config_dict[config_var] = value
                return True
                
        except Exception as e:
            print(f"更新配置值失败: {config_var} = {value}, 错误: {e}")
        
        return False
    
    def _save_config_file(self, config_dict: Dict):
        """保存配置字典到文件"""
        try:
            # 生成配置文件内容
            lines = []
            lines.append('"""策略配置文件 - 由参数集成器自动生成"""')
            lines.append('')
            lines.append('class StrategyConfig:')
            
            # 添加配置变量
            for var_name, var_value in config_dict.items():
                if var_name.isupper():  # 只处理大写常量
                    if isinstance(var_value, dict):
                        lines.append(f'    {var_name} = {{')
                        for key, val in var_value.items():
                            lines.append(f'        "{key}": {repr(val)},')
                        lines.append('    }')
                        lines.append('')
                    else:
                        lines.append(f'    {var_name} = {repr(var_value)}')
            
            # 写入文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"配置文件已保存: {self.config_file}")
            
        except Exception as e:
            raise Exception(f"保存配置文件失败: {e}")
    
    def _update_strategy_instance(self, strategy_instance, parameters: Dict) -> int:
        """动态更新策略实例参数"""
        update_count = 0
        
        try:
            # 尝试更新策略实例的参数
            # 这取决于策略类的具体实现
            
            # 检查是否有risk_params属性
            if hasattr(strategy_instance, 'risk_params'):
                risk_params = getattr(strategy_instance, 'risk_params')
                if isinstance(risk_params, dict):
                    # 更新风险参数
                    if 'single_trade_risk' in parameters:
                        risk_params['single_trade_risk'] = parameters['single_trade_risk']
                        update_count += 1
                    
                    if 'max_consecutive_losses' in parameters:
                        risk_params['max_consecutive_losses'] = parameters['max_consecutive_losses']
                        update_count += 1
            
            # 检查是否有TREND_STRENGTH_THRESHOLD属性
            if hasattr(strategy_instance, 'TREND_STRENGTH_THRESHOLD'):
                if 'TREND_STRENGTH_THRESHOLD' in parameters:
                    setattr(strategy_instance, 'TREND_STRENGTH_THRESHOLD', parameters['TREND_STRENGTH_THRESHOLD'])
                    update_count += 1
                elif 'trend_strength_threshold' in parameters:
                    setattr(strategy_instance, 'TREND_STRENGTH_THRESHOLD', parameters['trend_strength_threshold'])
                    update_count += 1
            
            # 尝试调用策略的更新方法（如果存在）
            if hasattr(strategy_instance, 'update_parameters'):
                try:
                    strategy_instance.update_parameters(parameters)
                    update_count += 1
                except:
                    pass
            
        except Exception as e:
            print(f"动态更新策略实例失败: {e}")
        
        return update_count
    
    def get_integration_history(self, limit: int = 10) -> List[Dict]:
        """获取集成历史"""
        return self.integration_history[-limit:] if self.integration_history else []
    
    def generate_integration_report(self, integration_result: Dict) -> str:
        """生成集成报告"""
        report = []
        report.append("=== 参数集成报告 ===")
        report.append("")
        
        if integration_result.get("success"):
            report.append("✅ 集成成功")
            report.append(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"参数更新数: {integration_result.get('parameters_updated', 0)}")
            report.append(f"动态更新数: {integration_result.get('dynamic_updates', 0)}")
            report.append(f"备份文件: {integration_result.get('backup_file', 'N/A')}")
            
            # 列出更新的参数
            params = integration_result.get("integration_record", {}).get("parameters_integrated", {})
            if params:
                report.append("")
                report.append("【更新的参数】")
                for param, value in params.items():
                    if not param.startswith('_'):
                        report.append(f"  • {param}: {value}")
        else:
            report.append("❌ 集成失败")
            report.append(f"错误: {integration_result.get('error', '未知错误')}")
        
        return "\n".join(report)


# 测试函数
if __name__ == "__main__":
    print("测试参数集成器...")
    
    try:
        # 创建集成器
        integrator = ParameterIntegrator(
            config_file="strategy_config.py",
            backup_dir="test_config_backups"
        )
        
        # 测试参数
        test_parameters = {
            "TREND_STRENGTH_THRESHOLD": 0.012,
            "position_size_pct": 0.08,
            "single_trade_risk": 0.006,
            "take_profit_multiplier": 1.02,
            "max_consecutive_losses": 3,
            "_internal_field": "跳过"
        }
        
        # 集成参数
        result = integrator.integrate_parameters(test_parameters)
        
        if result.get("success"):
            print("✅ 参数集成测试成功")
            print(f"更新参数数: {result.get('parameters_updated')}")
            
            # 生成报告
            report = integrator.generate_integration_report(result)
            print("\n" + report)
        else:
            print(f"❌ 参数集成测试失败: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("\n✅ 参数集成器测试完成")