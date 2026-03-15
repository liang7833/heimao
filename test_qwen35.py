#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3.5-4B 独立测试文件
用于测试 Qwen3.5-4B 模型的加载和分析功能
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 兼容 PyInstaller 打包环境
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)

# 模型目录 - 使用项目内的 models/qwen35 目录
MODEL_DIR = os.path.join(base_dir, "models", "qwen35")
print(f"模型目录: {MODEL_DIR}")


class Qwen35Tester:
    """Qwen 3.5:4B 测试器"""
    
    def __init__(self, model_dir: str = MODEL_DIR):
        """
        初始化 Qwen 3.5:4B 测试器
        
        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        
        print("="*80)
        print("Qwen 3.5:4B 测试器初始化")
        print("="*80)
        
        # 检查模型目录
        if not os.path.exists(model_dir):
            print(f"❌ 模型目录不存在: {model_dir}")
            print(f"请先运行模型下载脚本")
            return
        
        print(f"✓ 模型目录存在: {model_dir}")
        
        # 检查必要文件
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        missing_files = []
        for f in required_files:
            if not os.path.exists(os.path.join(model_dir, f)):
                missing_files.append(f)
        
        if missing_files:
            print(f"❌ 缺失必要文件: {missing_files}")
            return
        
        print(f"✓ 所有必要文件存在")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载 Qwen 3.5:4B 模型"""
        print("\n正在加载 Qwen 3.5:4B 模型...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import transformers
            import torch
            
            print(f"  Transformers 版本: {transformers.__version__}")
            
            # 检查模型配置
            config_file = os.path.join(self.model_dir, "config.json")
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_type = config.get("model_type", "unknown")
                print(f"  模型类型: {model_type}")
                
                # 特殊处理 Qwen3.5 模型
                if model_type == "qwen3_5":
                    print("  🎯 检测到 Qwen3.5 模型")
                    min_version = "4.46.0"
                    from packaging import version
                    if version.parse(transformers.__version__) < version.parse(min_version):
                        print(f"  ⚠️ Transformers 版本过旧，需要 {min_version}+")
                        print(f"  💡 请运行: pip install --upgrade transformers")
                        raise Exception(f"需要 transformers>={min_version}")
            
            print("  正在加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            print("  ✓ Tokenizer 加载成功")
            
            print("  正在加载模型（这可能需要几分钟，请耐心等待）...")
            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                **load_kwargs
            )
            print("  ✓ 模型加载成功")
            print(f"  ✓ 模型设备: {self.model.device}")
            
            print("\n" + "="*80)
            print("✓ Qwen 3.5:4B 模型加载完成！")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
    
    def _prepare_test_prompt(self, df: pd.DataFrame) -> str:
        """准备测试提示词
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            提示词字符串
        """
        # 获取最近的K线数据
        recent_df = df.tail(500).copy()
        
        # 计算指标（简化版）
        closes = recent_df['close'].values
        current_price = closes[-1]
        
        # 生成最近的K线数据（取最后50条）
        recent_klines = df.tail(50).copy()
        
        # 构建K线数据部分
        kline_data = []
        for idx, row in recent_klines.iterrows():
            kline_data.append(
                f"  - 开盘:${row['open']:.2f}, 最高:${row['high']:.2f}, "
                f"最低:${row['low']:.2f}, 收盘:${row['close']:.2f}, "
                f"成交量:{row.get('vol', row.get('volume', 0)):.0f}"
            )
        
        # 构建完善的提示词
        prompt = f"""你是一位专业的加密货币技术分析师。请仔细分析以下BTC数据，做出全面的技术分析。

【分析步骤】
1. 仔细观察最近50条K线的价格走势和成交量变化
2. 分析技术指标的含义和相互关系
3. 判断当前趋势方向和强度
4. 识别关键支撑位和阻力位
5. 检查是否有拐点信号
6. 给出明确的交易建议

【数据输入】
当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

最近50条K线数据（从旧到新排列）:
{chr(10).join(kline_data)}

【输出要求】
请在思考完成后，用以下格式返回JSON。重要：在JSON前面必须写上"FINAL_JSON:"这几个字！

【JSON字段说明】
- trend_direction: 趋势方向，必须是"LONG"（看涨）、"SHORT"（看跌）或"NEUTRAL"（观望）
- trend_strength: 趋势强度，范围0.002-0.01之间，数值越大表示趋势越强
  * 0.002-0.004: 弱趋势
  * 0.004-0.007: 中等趋势
  * 0.007-0.01: 强趋势
- pred_support: 预测支撑位，根据最近的价格低点和技术分析确定
- pred_resistance: 预测阻力位，根据最近的价格高点和技术分析确定
- has_turning_point: 是否检测到拐点，true=是，false=否
- recent_turn_type: 拐点类型，"PEAK"（峰值/顶部）、"VALLEY"（谷值/底部）或null（无）
- confidence: 分析置信度，范围0.0-1.0，数值越大越确信
- reasoning: 详细的分析理由，说明你是如何得出结论的

FINAL_JSON:
{{
    "trend_direction": "LONG",
    "trend_strength": 0.003,
    "pred_support": {current_price * 0.995:.2f},
    "pred_resistance": {current_price * 1.005:.2f},
    "has_turning_point": false,
    "recent_turn_type": null,
    "confidence": 0.7,
    "reasoning": "价格在支撑位附近，趋势向上"
}}"""
        
        return prompt
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """使用 Qwen 3.5:4B 分析数据
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            分析结果字典
        """
        if not self.model or not self.tokenizer:
            print("❌ 模型未加载，无法分析")
            return {"error": "模型未加载"}
        
        print("\n" + "="*80)
        print("开始分析...")
        print("="*80)
        
        try:
            import torch
            
            prompt = self._prepare_test_prompt(df)
            
            # 简化消息格式 - 只有用户消息
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 使用模型生成
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            print("  正在生成分析结果...")
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False,
                    top_p=0.9,
                    top_k=50
                )
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elapsed_time = time.time() - start_time
            
            print(f"  ✓ 分析完成，耗时: {elapsed_time:.2f}秒")
            print(f"\n  原始响应完整长度: {len(response)}")
            
            # 解析JSON响应 - 先找FINAL_JSON标记
            try:
                final_json_marker = "FINAL_JSON:"
                marker_pos = response.find(final_json_marker)
                
                if marker_pos >= 0:
                    # 找到标记，从标记后面开始找JSON
                    json_part = response[marker_pos + len(final_json_marker):]
                    json_start = json_part.find('{')
                    json_end = json_part.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = json_part[json_start:json_end]
                        print(f"\n  从FINAL_JSON标记后提取的JSON: {repr(json_str)}")
                        
                        # 验证JSON是否有效
                        if len(json_str) > 10:  # 确保不是空的{}
                            result = json.loads(json_str)
                            
                            print("\n" + "="*80)
                            print("分析结果:")
                            print("="*80)
                            print(json.dumps(result, indent=2, ensure_ascii=False))
                            print("="*80)
                            
                            return result
                        else:
                            print(f"  ⚠️ JSON内容太短，可能不是完整的结果")
                else:
                    # 没找到标记，尝试找最后一个JSON
                    print(f"  未找到FINAL_JSON标记，尝试找最后一个JSON...")
                
                # 备用方案：找最后一个JSON
                json_start = response.rfind('{')
                json_end = response.rfind('}') + 1
                print(f"\n  最后一个JSON起始位置: {json_start}, 结束位置: {json_end}")
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    print(f"  提取的JSON: {repr(json_str)}")
                    
                    if len(json_str) > 10:
                        result = json.loads(json_str)
                        
                        print("\n" + "="*80)
                        print("分析结果:")
                        print("="*80)
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                        print("="*80)
                        
                        return result
                
                print(f"  ⚠️ 无法找到有效的JSON")
                print(f"  完整响应: {repr(response)}")
                return {"error": "无法找到JSON格式的响应", "raw_response": response}
            except Exception as e:
                print(f"  ⚠️ JSON解析失败: {e}")
                print(f"  原始响应: {repr(response)}")
                return {"error": "JSON解析失败", "raw_response": response}
            
        except Exception as e:
            print(f"\n❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def generate_test_data(n_candles: int = 100) -> pd.DataFrame:
    """生成测试用的K线数据
    
    Args:
        n_candles: K线数量
        
    Returns:
        测试用K线DataFrame
    """
    print(f"\n生成测试数据: {n_candles}条K线")
    
    # 基础价格参数
    base_price = 50000.0
    volatility = 0.01
    
    # 生成时间序列
    end_time = datetime.now()
    timestamps = pd.date_range(
        end=end_time,
        periods=n_candles,
        freq="5T"
    )[::-1]
    
    # 生成随机价格序列
    np.random.seed(42)
    returns = np.random.normal(0, volatility, n_candles)
    cum_returns = np.exp(np.cumsum(returns))
    prices = base_price * cum_returns
    
    # 生成OHLC数据
    opens = prices * (1 + np.random.normal(0, 0.001, n_candles))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.003, n_candles)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.003, n_candles)))
    closes = prices
    
    # 生成成交量
    avg_volume = 1000
    volumes = avg_volume * (1 + np.random.normal(0, 0.3, n_candles))
    volumes = np.maximum(volumes, 10)
    
    # 创建DataFrame
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "amount": volumes,
        "volume": volumes
    })
    
    print(f"✓ 测试数据生成完成")
    print(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
    
    return df


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("Qwen 3.5:4B 独立测试")
    print("="*80)
    
    # 1. 生成测试数据
    df = generate_test_data(100)
    
    # 2. 初始化测试器
    tester = Qwen35Tester()
    
    if not tester.model or not tester.tokenizer:
        print("\n❌ 测试器初始化失败，无法继续测试")
        return
    
    # 3. 执行分析
    result = tester.analyze(df)
    
    # 4. 总结
    print("\n" + "="*80)
    print("测试总结:")
    print("="*80)
    if result is None:
        print(f"❌ 测试失败: 返回结果为空")
    elif "error" in result:
        print(f"❌ 测试失败: {result['error']}")
    else:
        print("✓ 测试成功！")
        print(f"  趋势方向: {result.get('trend_direction', 'N/A')}")
        print(f"  趋势强度: {result.get('trend_strength', 'N/A')}")
        print(f"  支撑位: ${result.get('pred_support', 'N/A')}")
        print(f"  阻力位: ${result.get('pred_resistance', 'N/A')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
