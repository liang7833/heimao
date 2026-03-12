#!/usr/bin/env python
"""Qwen3.5策略优化器 - 使用本地Qwen3.5模型进行策略代码生成和参数优化"""

import os
import sys
import logging

# 配置日志 - 兼容 PyInstaller 打包环境（无终端模式）
log_file = None
try:
    # 如果是打包后的无终端模式，创建日志文件
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.path.dirname(sys.executable), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'qwen3_optimizer.log')
    
    # 配置日志
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a',
            encoding='utf-8'
        )
    else:
        # 开发模式，输出到控制台
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
except Exception:
    # 备用方案：不配置日志，避免崩溃
    pass

# 禁用 transformers 的详细日志
try:
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.ERROR)
except Exception:
    pass

# 设置HuggingFace环境变量
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 尝试加载.env文件以获取HF_TOKEN
try:
    from dotenv import load_dotenv
    
    # 兼容 PyInstaller 打包环境
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)
    
    # 加载.env文件
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # 如果.env文件中设置了HF_TOKEN，则设置环境变量
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.strip() and not hf_token.startswith("your_"):
        os.environ["HF_TOKEN"] = hf_token
except ImportError:
    pass
except Exception:
    pass

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 尝试导入Qwen3相关库
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("警告: transformers库未安装，请运行: pip install transformers accelerate")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch库未安装，请运行: pip install torch")


class Qwen3Optimizer:
    """使用Qwen3.5进行策略优化的优化器"""
    
    def __init__(self, 
                 model_path: str = None,  # Qwen3.5-0.8B参数，性能与大小的平衡
                 device: str = None,
                 max_length: int = 2048):
        """
        初始化Qwen3.5优化器
        
        Args:
            model_name: Qwen模型名称
            device: 设备类型 (cuda/cpu/auto)
            max_length: 最大文本长度
        """
        # 兼容 PyInstaller 打包环境
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(__file__)
        
        if model_path:
            self.model_path = model_path
        else:
            # 检查 _internal/models 目录（打包环境），如果不存在再检查 models 目录
            self.model_path = os.path.join(base_dir, "_internal", "models", "Qwen3.5-0.8B-Instruct")
            if not os.path.exists(self.model_path):
                self.model_path = os.path.join(base_dir, "models", "Qwen3.5-0.8B-Instruct")
        self.max_length = max_length
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 模型和tokenizer
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        print(f"初始化Qwen3.5优化器: {self.model_path}")
        print(f"设备: {self.device}")
        
        # 缓存优化结果
        self.optimization_cache = {}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载Qwen3模型"""
        if not QWEN_AVAILABLE or not TORCH_AVAILABLE:
            print("  ✗ 依赖库缺失，无法加载Qwen3模型")
            return
        
        try:
            print(f"  加载 Qwen3 模型...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在：{self.model_path}")
            print(f"   检测到本地模型")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if self.device == "cuda" else self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True
            )
            self.model.eval()
            self.is_loaded = True
            print(f"  ✓ Qwen3模型加载成功")
            
        except Exception as e:
            print(f"  ✗ Qwen3模型加载失败: {e}")
            print(f"  可能的原因:")
            print(f"    1. 模型未下载到本地缓存")
            print(f"    2. 磁盘空间不足")
            print(f"    3. 网络连接问题")
            print(f"    4. 权限问题")
            print(f"  解决方案:")
            print(f"    下载模型: huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B-Instruct")
            self.is_loaded = False
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """
        通用文本生成方法
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        if not self.is_loaded:
            print("Qwen3模型未加载，无法生成文本")
            return None
        
        try:
            # 构建输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成文本
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除输入部分，只保留生成的部分
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Qwen3生成文本失败: {e}")
            return None
    
    def generate_strategy_code(self, 
                               strategy_type: str = "moving_average",
                               market_conditions: Dict = None,
                               risk_profile: str = "balanced") -> Dict:
        """
        生成交易策略代码
        
        Args:
            strategy_type: 策略类型
            market_conditions: 市场条件
            risk_profile: 风险偏好
            
        Returns:
            策略代码和参数
        """
        if not self.is_loaded:
            return {"error": "Qwen3模型未加载"}
        
        # 构建提示
        prompt = self._build_strategy_prompt(
            strategy_type=strategy_type,
            market_conditions=market_conditions,
            risk_profile=risk_profile
        )
        
        try:
            # 生成代码
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的代码
            strategy_code = self._extract_strategy_code(response)
            
            # 解析参数
            parameters = self._extract_parameters(response)
            
            result = {
                "success": True,
                "strategy_type": strategy_type,
                "risk_profile": risk_profile,
                "generated_code": strategy_code,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
            
            # 缓存结果
            cache_key = f"{strategy_type}_{risk_profile}"
            self.optimization_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_parameters(self,
                            backtest_results: Dict,
                            strategy_type: str = "moving_average",
                            target_metric: str = "sharpe_ratio") -> Dict:
        """
        基于回测结果优化策略参数
        
        Args:
            backtest_results: 回测结果
            strategy_type: 策略类型
            target_metric: 目标指标
            
        Returns:
            优化后的参数
        """
        if not self.is_loaded:
            return {"error": "Qwen3模型未加载"}
        
        # 构建优化提示
        prompt = self._build_optimization_prompt(
            backtest_results=backtest_results,
            strategy_type=strategy_type,
            target_metric=target_metric
        )
        
        try:
            # 生成优化建议
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取优化建议
            optimization_suggestions = self._extract_optimization_suggestions(response)
            
            # 提取推荐参数
            recommended_params = self._extract_recommended_params(response)
            
            result = {
                "success": True,
                "strategy_type": strategy_type,
                "target_metric": target_metric,
                "original_results": backtest_results.get("summary", {}),
                "optimization_suggestions": optimization_suggestions,
                "recommended_parameters": recommended_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # 缓存结果
            cache_key = f"optimize_{strategy_type}_{target_metric}"
            self.optimization_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_backtest_results(self, backtest_results: Dict) -> Dict:
        """
        分析回测结果并提供优化建议
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            分析报告
        """
        if not self.is_loaded:
            return {"error": "Qwen3模型未加载"}
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(backtest_results)
        
        try:
            # 生成分析
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取分析结果
            analysis = self._extract_analysis_results(response)
            
            result = {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_strategy_prompt(self, 
                               strategy_type: str,
                               market_conditions: Dict,
                               risk_profile: str) -> str:
        """构建策略生成提示"""
        
        market_info = market_conditions or {
            "trend": "sideways",
            "volatility": "medium",
            "liquidity": "high",
            "risk_level": "medium"
        }
        
        prompt = f"""你是一个专业的量化交易策略专家。请根据以下市场条件生成一个完整的交易策略。

市场条件:
- 趋势: {market_info.get('trend', 'sideways')}
- 波动率: {market_info.get('volatility', 'medium')}
- 流动性: {market_info.get('liquidity', 'high')}
- 风险等级: {market_info.get('risk_level', 'medium')}

策略类型: {strategy_type}
风险偏好: {risk_profile}

请生成:
1. 策略名称和概述
2. 完整的Python策略代码（使用pandas和numpy）
3. 关键参数设置
4. 风险管理规则
5. 入场和出场条件

请以JSON格式返回:
{{
    "strategy_name": "策略名称",
    "description": "策略描述",
    "code": "完整的Python策略代码",
    "parameters": {{
        "param1": "值",
        "param2": "值"
    }},
    "risk_management": {{
        "stop_loss": "止损规则",
        "take_profit": "止盈规则",
        "position_size": "仓位大小"
    }}
}}"""

        return prompt
    
    def _build_optimization_prompt(self,
                                   backtest_results: Dict,
                                   strategy_type: str,
                                   target_metric: str) -> str:
        """构建参数优化提示"""
        
        summary = backtest_results.get("summary", {})
        metrics = backtest_results.get("performance_metrics", {})
        
        prompt = f"""你是一个专业的量化交易策略优化专家。请分析以下回测结果并提供优化建议。

当前回测结果:
- 初始资金: ${summary.get('initial_capital', 0):.2f}
- 最终权益: ${summary.get('final_equity', 0):.2f}
- 总收益率: {summary.get('total_return_pct', 0):.2f}%
- 最大回撤: {summary.get('max_drawdown_pct', 0):.2f}%
- 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
- 总交易次数: {summary.get('total_trades', 0)}
- 胜率: {summary.get('win_rate_pct', 0):.2f}%
- 盈亏比: {metrics.get('profit_factor', 0):.2f}

目标指标: {target_metric}
策略类型: {strategy_type}

请分析:
1. 当前策略的优势和劣势
2. 参数优化建议（如移动平均线周期、RSI阈值等）
3. 风险管理改进方案
4. 具体的参数调整值

请以JSON格式返回:
{{
    "analysis": {{
        "strengths": ["优势1", "优势2"],
        "weaknesses": ["劣势1", "劣势2"],
        "issues": ["问题1", "问题2"]
    }},
    "optimization_suggestions": [
        "建议1",
        "建议2"
    ],
    "recommended_parameters": {{
        "param1": 新值,
        "param2": 新值
    }},
    "expected_improvement": {{
        "sharpe_ratio": 预期值,
        "max_drawdown": 预期值,
        "win_rate": 预期值
    }}
}}"""

        return prompt
    
    def _build_analysis_prompt(self, backtest_results: Dict) -> str:
        """构建分析提示"""
        
        summary = backtest_results.get("summary", {})
        metrics = backtest_results.get("performance_metrics", {})
        
        prompt = f"""你是一个专业的量化交易策略分析师。请分析以下回测结果并提供详细的分析报告。

回测结果摘要:
- 初始资金: ${summary.get('initial_capital', 0):.2f}
- 最终权益: ${summary.get('final_equity', 0):.2f}
- 总收益率: {summary.get('total_return_pct', 0):.2f}%
- 年化收益率: {metrics.get('annual_return', 0)*100:.2f}%
- 波动率: {metrics.get('volatility', 0)*100:.2f}%
- 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%
- 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
- 总交易次数: {summary.get('total_trades', 0)}
- 胜率: {summary.get('win_rate_pct', 0):.2f}%
- 盈亏比: {metrics.get('profit_factor', 0):.2f}

请提供:
1. 整体表现评估
2. 风险分析
3. 策略稳定性评估
4. 详细优化建议
5. 可能的改进方向

请以JSON格式返回:
{{
    "overall_assessment": {{
        "rating": "优秀/良好/一般/需优化",
        "score": 0-100,
        "summary": "总体评价"
    }},
    "risk_analysis": {{
        "drawdown_risk": "回撤风险等级",
        "volatility_risk": "波动率风险等级",
        "liquidity_risk": "流动性风险等级"
    }},
    "strategy_stability": {{
        "consistency": "一致性评估",
        "robustness": "稳健性评估"
    }},
    "detailed_analysis": {{
        "strengths": ["优势1", "优势2"],
        "weaknesses": ["劣势1", "劣势2"],
        "concerns": ["担忧1", "担忧2"]
    }},
    "optimization_recommendations": [
        "建议1",
        "建议2",
        "建议3"
    ],
    "improvement_directions": [
        "方向1",
        "方向2"
    ]
}}"""

        return prompt
    
    def _extract_strategy_code(self, response: str) -> str:
        """提取生成的策略代码"""
        # 尝试从JSON中提取
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('code', response)
        except:
            pass
        
        return response
    
    def _extract_parameters(self, response: str) -> Dict:
        """提取参数"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('parameters', {})
        except:
            pass
        
        return {}
    
    def _extract_optimization_suggestions(self, response: str) -> List[str]:
        """提取优化建议"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('optimization_suggestions', [])
        except:
            pass
        
        return []
    
    def _extract_recommended_params(self, response: str) -> Dict:
        """提取推荐参数"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('recommended_parameters', {})
        except:
            pass
        
        return {}
    
    def _extract_analysis_results(self, response: str) -> Dict:
        """提取分析结果"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data
        except:
            pass
        
        return {"raw_response": response}
    
    def get_cached_results(self, key: str = None) -> Dict:
        """获取缓存结果"""
        if key:
            return self.optimization_cache.get(key)
        return self.optimization_cache
    
    def optimize_coordinator_parameters(self, 
                                       risk_profile: str = "balanced",
                                       strategy_type: str = "trend_following",
                                       market_conditions: Dict = None) -> Dict:
        """
        优化策略协调器参数
        
        Args:
            risk_profile: 风险偏好 (conservative/balanced/aggressive)
            strategy_type: 策略类型
            market_conditions: 市场条件
            
        Returns:
            协调器优化参数
        """
        if not self.is_loaded:
            return {"error": "Qwen3模型未加载"}
        
        # 构建提示
        prompt = self._build_coordinator_optimization_prompt(
            risk_profile=risk_profile,
            strategy_type=strategy_type,
            market_conditions=market_conditions
        )
        
        try:
            # 生成优化建议
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取协调器参数
            coordinator_params = self._extract_coordinator_parameters(response)
            
            result = {
                "success": True,
                "risk_profile": risk_profile,
                "strategy_type": strategy_type,
                "coordinator_parameters": coordinator_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # 缓存结果
            cache_key = f"coordinator_{risk_profile}_{strategy_type}"
            self.optimization_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_coordinator_optimization_prompt(self,
                                               risk_profile: str,
                                               strategy_type: str,
                                               market_conditions: Dict = None) -> str:
        """构建协调器参数优化提示"""
        
        market_info = market_conditions or {
            "trend": "sideways",
            "volatility": "medium",
            "liquidity": "high"
        }
        
        prompt = f"""你是一个专业的量化交易系统优化专家。请根据以下条件优化策略协调器的参数。

策略协调器可配置参数:
- min_signal_strength: 最小信号强度 (0.0-1.0, 默认0.15)
- max_position_size: 最大仓位比例 (0.0-1.0, 默认0.1)
- sentiment_weight: 舆情信号权重 (0.0-1.0, 默认0.3)
- technical_weight: 技术信号权重 (0.0-1.0, 默认0.7)
- black_swan_threshold: 黑天鹅风险阈值 (LOW/MEDIUM/HIGH, 默认HIGH)
- enable_adaptive_filtering: 是否启用自适应过滤 (True/False, 默认True)

输入条件:
- 风险偏好: {risk_profile} (conservative=保守, balanced=平衡, aggressive=激进)
- 策略类型: {strategy_type}
- 市场趋势: {market_info.get('trend', 'sideways')}
- 市场波动率: {market_info.get('volatility', 'medium')}
- 市场流动性: {market_info.get('liquidity', 'high')}

请根据以上条件，推荐协调器的最优参数配置。

请以JSON格式返回:
{{
    "reasoning": "参数优化理由",
    "coordinator_parameters": {{
        "min_signal_strength": 0.15,
        "max_position_size": 0.1,
        "sentiment_weight": 0.3,
        "technical_weight": 0.7,
        "black_swan_threshold": "HIGH",
        "enable_adaptive_filtering": true
    }}
}}"""

        return prompt
    
    def _extract_coordinator_parameters(self, response: str) -> Dict:
        """提取协调器参数"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('coordinator_parameters', {})
        except:
            pass
        
        return {}
    
    def analyze_and_optimize_all_parameters(self,
                                            market_conditions: Dict = None,
                                            performance_data: Dict = None,
                                            risk_profile: str = "balanced",
                                            optimization_reason: str = "regular",
                                            optimization_details: dict = None) -> Dict:
        """
        全面分析市场并优化所有交易参数
        
        Args:
            market_conditions: 市场条件数据
            performance_data: 历史性能数据
            risk_profile: 风险偏好
            optimization_reason: 优化原因
                - "regular": 定期优化
                - "loss_trigger": 亏损触发
                - "performance": 表现不佳
                - "market_change": 市场环境变化
            optimization_details: 优化详细信息
            
        Returns:
            完整的优化参数集
        """
        if not self.is_loaded:
            return {"error": "Qwen3模型未加载"}
        
        # 构建提示
        prompt = self._build_full_optimization_prompt(
            market_conditions=market_conditions,
            performance_data=performance_data,
            risk_profile=risk_profile,
            optimization_reason=optimization_reason,
            optimization_details=optimization_details
        )
        
        try:
            # 生成优化建议
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1536,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取完整参数
            full_params = self._extract_full_parameters(response)
            
            result = {
                "success": True,
                "risk_profile": risk_profile,
                "full_parameters": full_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # 缓存结果
            cache_key = f"full_optimization_{risk_profile}_{datetime.now().strftime('%Y%m%d_%H')}"
            self.optimization_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_full_optimization_prompt(self,
                                        market_conditions: Dict = None,
                                        performance_data: Dict = None,
                                        risk_profile: str = "balanced",
                                        optimization_reason: str = "regular",
                                        optimization_details: dict = None) -> str:
        """构建完整优化提示"""
        
        market_info = market_conditions or {
            "trend": "sideways",
            "volatility": "medium",
            "liquidity": "high",
            "current_price": 50000,
            "price_change_24h": 0.0
        }
        
        perf_info = performance_data or {
            "win_rate": 0.55,
            "profit_factor": 1.3,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
            "total_trades": 100
        }
        
        # 优化原因说明
        reason_descriptions = {
            "regular": "定期优化 - 保持策略持续适应市场变化",
            "loss_trigger": "亏损触发 - 因账户亏损达到阈值需要优化风险控制",
            "no_trade": "无交易触发 - 因12小时无交易需要优化信号灵敏度和入场条件",
            "performance": "表现不佳 - 策略表现持续低迷需要调整",
            "market_change": "市场环境变化 - 市场状态发生重大变化"
        }

        reason_text = reason_descriptions.get(optimization_reason, "定期优化")
        
        prompt = f"""你是一个专业的量化交易策略优化专家。请全面分析市场条件并优化所有交易参数。

【重要】本次优化原因:
{reason_text}
{'详细信息: ' + str(optimization_details) if optimization_details else ''}

【优化建议】:
- 如果是亏损触发，请重点优化止损参数、仓位大小和风险阈值
- 如果是无交易触发，请重点优化信号灵敏度、入场条件和确认次数，降低入场门槛
- 如果是定期优化，请微调参数，保持当前策略风格
- 如果是表现不佳，请全面评估并重新平衡所有参数
- 如果是市场环境变化，请重点调整趋势判断和波动率参数

输入信息:
市场条件:
- 趋势方向: {market_info.get('trend', 'sideways')}
- 波动率: {market_info.get('volatility', 'medium')}
- 流动性: {market_info.get('liquidity', 'high')}
- 当前价格: ${market_info.get('current_price', 50000):.2f}
- 24h涨跌幅: {market_info.get('price_change_24h', 0):.2%}

历史性能:
- 胜率: {perf_info.get('win_rate', 0.55):.1%}
- 盈亏比: {perf_info.get('profit_factor', 1.3):.2f}
- 夏普比率: {perf_info.get('sharpe_ratio', 1.2):.2f}
- 最大回撤: {perf_info.get('max_drawdown', 0.15):.1%}
- 总交易次数: {perf_info.get('total_trades', 100)}

风险偏好: {risk_profile} (conservative=保守, balanced=平衡, aggressive=激进)

请优化以下所有交易参数:

1. 协调器参数 (coordinator_parameters):
   - min_signal_strength: 最小信号强度 (0.0-1.0)
   - max_position_size: 最大仓位比例 (0.0-1.0)
   - sentiment_weight: 舆情权重 (0.0-1.0)
   - technical_weight: 技术权重 (0.0-1.0)
   - black_swan_threshold: 黑天鹅阈值 (LOW/MEDIUM/HIGH)
   - enable_adaptive_filtering: 自适应过滤 (True/False)

2. 基础参数 (basic_parameters):
   - LEVERAGE: 杠杆倍数 (1-125)
   - TREND_STRENGTH_THRESHOLD: 趋势强度阈值 (0.001-0.05)
   - LOOKBACK_PERIOD: 回看周期 (64-1024)
   - PREDICTION_LENGTH: 预测长度 (12-120)
   - CHECK_INTERVAL: 检查间隔秒数 (60-1800)

3. 入场过滤参数 (entry_filter):
   - max_kline_change: 最大K线变化 (0.001-0.05)
   - max_funding_rate_long: 多头最大资金费率 (-0.1-0.1)
   - min_funding_rate_short: 空头最小资金费率 (-0.1-0.1)
   - support_buffer: 支撑位缓冲 (1.001-1.01)
   - resistance_buffer: 阻力位缓冲 (0.99-0.999)

4. 止损参数 (stop_loss):
   - long_buffer: 多头止损缓冲 (0.98-0.999)
   - short_buffer: 空头止损缓冲 (1.001-1.02)

5. 止盈参数 (take_profit):
   - tp1_multiplier_long: 多头止盈1倍数 (1.005-1.05)
   - tp2_multiplier_long: 多头止盈2倍数 (1.01-1.1)
   - tp1_multiplier_short: 空头止盈1倍数 (0.95-0.995)
   - tp2_multiplier_short: 空头止盈2倍数 (0.9-0.99)
   - tp1_position_ratio: 止盈1仓位比例 (0.3-0.7)

6. 风险管理参数 (risk_management):
   - single_trade_risk: 单笔交易风险 (0.005-0.05)
   - daily_loss_limit: 日亏损限制 (0.01-0.1)
   - max_consecutive_losses: 最大连续亏损 (2-10)
   - pause_after_losses_minutes: 亏损后暂停分钟 (10-60)
   - max_single_position: 最大单笔仓位 (0.1-0.5)
   - max_daily_position: 最大日仓位 (0.2-0.8)
   - extreme_move_threshold: 极端波动阈值 (0.01-0.05)

7. 交易频率参数 (trade_frequency):
   - max_daily_trades: 最大日交易次数 (5-50)
   - min_trade_interval_minutes: 最小交易间隔分钟 (5-60)
   - active_hours_start: 活跃开始时间 (0-23)
   - active_hours_end: 活跃结束时间 (0-23)

8. 仓位管理参数 (position_management):
   - initial_entry_ratio: 初始入场比例 (0.3-0.8)
   - confirm_interval_kline: 确认K线数 (1-10)

9. 三套策略参数 (strategy_parameters):
   - TREND_BURST.min_trend_strength: 趋势爆发策略最小趋势强度 (0.005-0.02)
   - RANGE_ARBITRAGE.max_trend_strength: 震荡套利策略最大趋势强度 (0.003-0.01)
   - NEWS_BREAKOUT.min_trend_strength: 消息突破策略最小趋势强度 (0.008-0.025)

10. 确认次数参数 (confirmation_parameters):
    - ENTRY_CONFIRM_COUNT: 开仓确认次数 (1-5)
    - REVERSE_CONFIRM_COUNT: 趋势反转确认次数 (2-5)
    - STRONG_REVERSE_CONFIRM_COUNT: 强趋势反转确认次数 (2-4)

请以JSON格式返回:
{{
    "market_analysis": "市场分析总结",
    "optimization_reasoning": "优化理由说明",
    "coordinator_parameters": {{...}},
    "basic_parameters": {{
        "LEVERAGE": 5,
        "TREND_STRENGTH_THRESHOLD": 0.008,
        "ENTRY_CONFIRM_COUNT": 2,
        "REVERSE_CONFIRM_COUNT": 3,
        "STRONG_REVERSE_CONFIRM_COUNT": 2
    }},
    "entry_filter": {{...}},
    "stop_loss": {{...}},
    "take_profit": {{...}},
    "risk_management": {{...}},
    "trade_frequency": {{...}},
    "position_management": {{...}},
    "strategy_parameters": {{
        "TREND_BURST": {{
            "min_trend_strength": 0.010
        }},
        "RANGE_ARBITRAGE": {{
            "max_trend_strength": 0.008
        }},
        "NEWS_BREAKOUT": {{
            "min_trend_strength": 0.015
        }}
    }}
}}"""

        return prompt
    
    def _extract_full_parameters(self, response: str) -> Dict:
        """提取完整参数"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data
        except:
            pass
        
        return {}
    
    def clear_cache(self):
        """清空缓存"""
        self.optimization_cache = {}
        print("缓存已清空")


# 使用示例
if __name__ == "__main__":
    print("=== Qwen3策略优化器测试 ===")
    
    # 初始化优化器
    print("\n1. 初始化优化器...")
    try:
        optimizer = Qwen3Optimizer()
        
        if not optimizer.is_loaded:
            print("  ✗ 模型加载失败，跳过测试")
            sys.exit(1)
        
        print("  ✓ 优化器初始化成功")
        
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        sys.exit(1)
    
    # 测试策略代码生成
    print("\n2. 测试策略代码生成...")
    try:
        market_conditions = {
            "trend": "upward",
            "volatility": "medium",
            "liquidity": "high",
            "risk_level": "medium"
        }
        
        result = optimizer.generate_strategy_code(
            strategy_type="moving_average",
            market_conditions=market_conditions,
            risk_profile="balanced"
        )
        
        if result.get("success"):
            print("  ✓ 策略代码生成成功")
            print(f"    策略名称: {result.get('strategy_type')}")
            print(f"    参数数量: {len(result.get('parameters', {}))}")
            
            # 显示部分代码
            code = result.get("generated_code", "")
            if len(code) > 200:
                print(f"    代码长度: {len(code)}字符")
                print(f"    前200字符: {code[:200]}...")
            else:
                print(f"    代码: {code}")
        else:
            print(f"  ✗ 策略代码生成失败: {result.get('error')}")
            
    except Exception as e:
        print(f"  ✗ 策略代码生成测试失败: {e}")
    
    # 测试参数优化
    print("\n3. 测试参数优化...")
    try:
        # 模拟回测结果
        backtest_results = {
            "summary": {
                "initial_capital": 10000.0,
                "final_equity": 11500.0,
                "total_return_pct": 15.0,
                "max_drawdown_pct": 8.0,
                "total_trades": 25,
                "win_rate_pct": 60.0
            },
            "performance_metrics": {
                "sharpe_ratio": 1.2,
                "annual_return": 0.18,
                "volatility": 0.12,
                "max_drawdown": 0.08,
                "profit_factor": 1.5
            }
        }
        
        result = optimizer.optimize_parameters(
            backtest_results=backtest_results,
            strategy_type="moving_average",
            target_metric="sharpe_ratio"
        )
        
        if result.get("success"):
            print("  ✓ 参数优化成功")
            print(f"    目标指标: {result.get('target_metric')}")
            print(f"    优化建议数量: {len(result.get('optimization_suggestions', []))}")
            
            if result.get('optimization_suggestions'):
                print("    建议:")
                for i, suggestion in enumerate(result['optimization_suggestions'][:3], 1):
                    print(f"      {i}. {suggestion}")
            
            if result.get('recommended_parameters'):
                print("    推荐参数:")
                for param, value in result['recommended_parameters'].items():
                    print(f"      {param}: {value}")
        else:
            print(f"  ✗ 参数优化失败: {result.get('error')}")
            
    except Exception as e:
        print(f"  ✗ 参数优化测试失败: {e}")
    
    # 测试回测结果分析
    print("\n4. 测试回测结果分析...")
    try:
        result = optimizer.analyze_backtest_results(backtest_results)
        
        if result.get("success"):
            print("  ✓ 回测分析成功")
            analysis = result.get("analysis", {})
            
            if "overall_assessment" in analysis:
                assessment = analysis["overall_assessment"]
                print(f"    总体评分: {assessment.get('rating', 'N/A')}")
                print(f"    评分: {assessment.get('score', 'N/A')}/100")
            
            if "detailed_analysis" in analysis:
                details = analysis["detailed_analysis"]
                if details.get("strengths"):
                    print(f"    优势: {', '.join(details['strengths'][:2])}")
                if details.get("weaknesses"):
                    print(f"    劣势: {', '.join(details['weaknesses'][:2])}")
            
            if "optimization_recommendations" in analysis:
                print(f"    优化建议数量: {len(analysis['optimization_recommendations'])}")
        else:
            print(f"  ✗ 回测分析失败: {result.get('error')}")
            
    except Exception as e:
        print(f"  ✗ 回测分析测试失败: {e}")
    
    print("\n=== Qwen3策略优化器测试完成 ===")