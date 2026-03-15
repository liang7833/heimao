#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen分析模块 - 使用Qwen 3.5:4B进行K线数据分析

================================================================================
Qwen分析模块功能说明
================================================================================

1. 功能：
   - 接收币安K线数据
   - 使用Qwen模型进行技术分析
   - 生成与Kronos相同格式的交易信号
   - 返回趋势方向、强度、支撑位、阻力位等

2. 输入数据格式（与Kronos相同）：
   - DataFrame包含：open, high, low, close, volume等
   - 支持5m, 15m, 1h等时间周期

3. 输出格式（与Kronos兼容）：
   - trend_direction: LONG/SHORT/NEUTRAL
   - trend_strength: 0.002-0.01范围
   - pred_support: 预测支撑位
   - pred_resistance: 预测阻力位
   - has_turning_point: 是否检测到拐点
   - signal_valid: 信号是否有效

================================================================================
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 兼容 PyInstaller 打包环境
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)


class QwenAnalyzer:
    """Qwen分析器 - 使用Qwen模型进行K线数据分析"""
    
    def __init__(self, 
                 use_local_model: bool = True,
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 symbol: str = "BTC"):
        """
        初始化Qwen分析器
        
        Args:
            use_local_model: 是否使用本地模型
            model_name: Qwen模型名称
            symbol: 交易品种符号
        """
        self.use_local_model = use_local_model
        self.model_name = model_name
        self.symbol = symbol
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        
        # 缓存最近的分析结果
        self.cache = {}
        self.cache_ttl = 60  # 1分钟缓存
        
        # 初始化模型
        self._initialize_models()
        
        print(f"Qwen分析器初始化完成 (本地模型: {use_local_model})")
    
    def _initialize_models(self):
        """初始化Qwen模型"""
        if self.use_local_model:
            try:
                print("正在加载Qwen模型...")
                
                # 检查多个可能的本地模型目录（只使用 Qwen3.5）
                possible_model_paths = [
                    os.path.join(base_dir, "_internal", "models", "qwen35"),
                    os.path.join(base_dir, "models", "qwen35"),
                ]
                
                model_path = None
                for path in possible_model_paths:
                    if os.path.exists(path):
                        # 检查是否有tokenizer和model文件
                        has_tokenizer = os.path.exists(os.path.join(path, "tokenizer_config.json"))
                        has_model = os.path.exists(os.path.join(path, "config.json"))
                        if has_tokenizer and has_model:
                            model_path = path
                            print(f"  找到模型目录: {path}")
                            break
                
                if not model_path:
                    print(f"  ⚠️ 未找到本地Qwen模型目录")
                    print(f"  ⚠️ 尝试的路径:")
                    for path in possible_model_paths:
                        print(f"    - {path}")
                    print(f"  ⚠️ 将使用基于规则的分析")
                    print(f"  💡 提示: 运行 'python download_qwen_model.py' 下载模型")
                    self.use_local_model = False
                else:
                    # 尝试加载模型
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        import transformers
                        import torch
                        
                        print(f"  Transformers版本: {transformers.__version__}")
                        
                        # 检查模型配置
                        config_file = os.path.join(model_path, "config.json")
                        if os.path.exists(config_file):
                            import json
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                model_type = config.get("model_type", "unknown")
                                print(f"  检测到模型类型: {model_type}")
                                
                                # 特殊处理Qwen3.5模型
                                if model_type == "qwen3_5":
                                    print("  🎯 检测到Qwen3.5模型")
                                    min_version = "4.46.0"
                                    from packaging import version
                                    if version.parse(transformers.__version__) < version.parse(min_version):
                                        print(f"  ⚠️ Transformers版本过旧，需要 {min_version}+")
                                        print(f"  💡 请运行: pip install --upgrade transformers")
                                        raise Exception(f"需要transformers>={min_version}")
                        
                        print("  正在加载tokenizer...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path, 
                            local_files_only=True,
                            trust_remote_code=True
                        )
                        
                        print("  正在加载模型（这可能需要几分钟）...")
                        
                        # 尝试多种加载方式
                        load_kwargs = {
                            "local_files_only": True,
                            "trust_remote_code": True,
                            "dtype": torch.float16,
                            "device_map": "auto",
                            "low_cpu_mem_usage": True
                        }
                        
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path, 
                                **load_kwargs
                            )
                        except Exception as first_e:
                            print(f"  第一次加载失败: {first_e}")
                            print("  尝试第二种加载方式...")
                            # 尝试不带trust_remote_code
                            load_kwargs["trust_remote_code"] = False
                            try:
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_path, 
                                    **load_kwargs
                                )
                            except Exception as second_e:
                                print(f"  第二次加载失败: {second_e}")
                                print("  尝试第三种加载方式（float32）...")
                                # 尝试float32
                                load_kwargs["dtype"] = torch.float32
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_path, 
                                    **load_kwargs
                                )
                        
                        print("  ✓ Qwen模型加载成功")
                        
                    except Exception as local_e:
                        print(f"  ⚠️ Qwen模型加载失败: {local_e}")
                        print(f"  ⚠️ 将使用基于规则的分析")
                        print(f"  💡 解决方案:")
                        print(f"    1. 立即升级transformers:")
                        print(f"       py -m pip install --upgrade transformers huggingface_hub")
                        print(f"    2. 或者下载Qwen2.5-7B-Instruct（兼容性更好）:")
                        print(f"       py download_qwen_model.py --model Qwen/Qwen2.5-7B-Instruct")
                        print(f"    3. 确保有足够的显存（6GB+）")
                        print(f"    4. 规则分析模式无需模型文件，立即可用！")
                        self.use_local_model = False
                        
            except Exception as e:
                print(f"初始化Qwen模型时出错: {e}")
                print("将使用基于规则的分析")
                self.use_local_model = False
        
        if not self.use_local_model:
            print("使用基于规则的Qwen分析（备用方法）")
    
    def _calculate_kronos_features(self, df):
        """计算Kronos预测所需的27个技术指标特征"""
        df = df.copy()
        
        # 基础价格数据
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # 安全获取amount列（训练时用的是 amt 和 vol）
        if "amt" in df.columns:
            amount = df["amt"].values
        elif "amount" in df.columns:
            amount = df["amount"].values
            df["amt"] = amount
        elif "volume" in df.columns:
            amount = df["volume"].values
            df["amt"] = amount
        else:
            amount = np.zeros(len(df))
            df["amt"] = amount
        
        # 添加训练时用的 vol 列
        if "vol" not in df.columns:
            if "volume" in df.columns:
                df["vol"] = df["volume"].values
            else:
                df["vol"] = amount
        
        # 1. MA5, MA10, MA20
        df["MA5"] = pd.Series(close).rolling(window=5).mean().values
        df["MA10"] = pd.Series(close).rolling(window=10).mean().values
        df["MA20"] = pd.Series(close).rolling(window=20).mean().values
        
        # 2. 收盘价/MA20 乖离率
        df["BIAS20"] = (close / df["MA20"] - 1) * 100
        
        # 3. ATR(14)
        tr1 = high - low
        tr2 = np.zeros_like(high)
        tr3 = np.zeros_like(high)
        if len(close) > 1:
            tr2[1:] = np.abs(high[1:] - close[:-1])
            tr3[1:] = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values
        
        # 4. 振幅
        df["AMPLITUDE"] = (high - low) / close * 100
        
        # 5. 成交额MA5, MA10
        df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
        df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values
        
        # 6. 量比
        df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]
        
        # 7. RSI(14) 和 RSI(7)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain14 = gain.rolling(window=14).mean()
        avg_loss14 = loss.rolling(window=14).mean()
        avg_loss14_safe = avg_loss14.replace(0, 0.00001)
        rs14 = avg_gain14 / avg_loss14_safe
        df["RSI14"] = (100 - (100 / (1 + rs14))).values
        
        avg_gain7 = gain.rolling(window=7).mean()
        avg_loss7 = loss.rolling(window=7).mean()
        avg_loss7_safe = avg_loss7.replace(0, 0.00001)
        rs7 = avg_gain7 / avg_loss7_safe
        df["RSI7"] = (100 - (100 / (1 + rs7))).values
        
        # 8. MACD线 和 MACD柱
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
        
        # 9. 价格斜率(5期) 和 (10期)
        def calculate_slope(values):
            if len(values) < 2:
                return 0.0
            try:
                x = np.arange(len(values))
                y = np.array(values)
                n = len(x)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_x2 = np.sum(x * x)
                denominator = n * sum_x2 - sum_x * sum_x
                if denominator == 0:
                    return 0.0
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                return slope
            except Exception:
                return 0.0
        
        df["PRICE_SLOPE5"] = (
            pd.Series(close)
            .rolling(window=5)
            .apply(
                lambda x: calculate_slope(x),
                raw=True,
            )
            .values
        )
        df["PRICE_SLOPE10"] = (
            pd.Series(close)
            .rolling(window=10)
            .apply(
                lambda x: calculate_slope(x),
                raw=True,
            )
            .values
        )
        
        # 10. 近5日最高/最低
        df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
        df["LOW5"] = pd.Series(low).rolling(window=5).min().values
        
        # 11. 近10日最高/最低
        df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
        df["LOW10"] = pd.Series(low).rolling(window=10).min().values
        
        # 12. 放量突破
        df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int).values
        
        # 13. 缩量下跌
        df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int).values
        
        return df
    
    def _prepare_prompt(self, df: pd.DataFrame) -> str:
        """准备Qwen分析提示词（包含27个Kronos指标）
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            提示词字符串
        """
        # 获取最近的K线数据
        recent_df = df.tail(500).copy()
        
        # 计算所有27个Kronos指标
        df_with_features = self._calculate_kronos_features(recent_df)
        
        # 获取最新值
        latest = df_with_features.iloc[-1]
        
        # 生成最近的K线数据（取最后50条）
        recent_klines = df_with_features.tail(50).copy()
        
        # 构建K线数据部分
        kline_data = []
        for idx, row in recent_klines.iterrows():
            kline_data.append(
                f"  - 开盘:${row['open']:.2f}, 最高:${row['high']:.2f}, "
                f"最低:${row['low']:.2f}, 收盘:${row['close']:.2f}, "
                f"成交量:{row.get('vol', row.get('volume', 0)):.0f}"
            )
        
        # 构建提示词 - 完善版，更精准的分析
        prompt = f"""你是一位专业的加密货币技术分析师。请仔细分析以下{self.symbol}数据，做出全面的技术分析。

【分析步骤】
1. 仔细观察最近50条K线的价格走势和成交量变化
2. 分析27个技术指标的含义和相互关系
3. 判断当前趋势方向和强度
4. 识别关键支撑位和阻力位
5. 检查是否有拐点信号
6. 给出明确的交易建议

【数据输入】
当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

最近50条K线数据（从旧到新排列）:
{chr(10).join(kline_data)}

最新技术指标数据（27个Kronos指标）:
- 基础价格数据:
  - 当前价格: ${latest['close']:.2f}
  - 最高价: ${latest['high']:.2f}
  - 最低价: ${latest['low']:.2f}
  - 开盘价: ${latest['open']:.2f}
  - 成交量: {latest.get('vol', latest.get('volume', 0)):.0f}
  - 成交额: {latest.get('amt', latest.get('amount', 0)):.0f}

- 移动平均线:
  - MA5: ${latest.get('MA5', 0):.2f}
  - MA10: ${latest.get('MA10', 0):.2f}
  - MA20: ${latest.get('MA20', 0):.2f}

- 乖离率:
  - BIAS20: {latest.get('BIAS20', 0):.2f}%

- 波动性指标:
  - ATR14: ${latest.get('ATR14', 0):.2f}
  - AMPLITUDE: {latest.get('AMPLITUDE', 0):.2f}%

- 成交量指标:
  - AMOUNT_MA5: {latest.get('AMOUNT_MA5', 0):.0f}
  - AMOUNT_MA10: {latest.get('AMOUNT_MA10', 0):.0f}
  - VOL_RATIO: {latest.get('VOL_RATIO', 0):.2f}

- 动量指标:
  - RSI14: {latest.get('RSI14', 0):.2f}
  - RSI7: {latest.get('RSI7', 0):.2f}

- MACD指标:
  - MACD: {latest.get('MACD', 0):.4f}
  - MACD_HIST: {latest.get('MACD_HIST', 0):.4f}

- 趋势指标:
  - PRICE_SLOPE5: {latest.get('PRICE_SLOPE5', 0):.4f}
  - PRICE_SLOPE10: {latest.get('PRICE_SLOPE10', 0):.4f}

- 极值指标:
  - HIGH5: ${latest.get('HIGH5', 0):.2f}
  - LOW5: ${latest.get('LOW5', 0):.2f}
  - HIGH10: ${latest.get('HIGH10', 0):.2f}
  - LOW10: ${latest.get('LOW10', 0):.2f}

- 成交量突破指标:
  - VOL_BREAKOUT: {latest.get('VOL_BREAKOUT', 0)} (0=否, 1=是)
  - VOL_SHRINK: {latest.get('VOL_SHRINK', 0)} (0=否, 1=是)

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
    "trend_direction": "LONG/SHORT/NEUTRAL",
    "trend_strength": 0.002-0.01,
    "pred_support": number,
    "pred_resistance": number,
    "has_turning_point": true/false,
    "recent_turn_type": "PEAK/VALLEY/null",
    "confidence": 0.0-1.0,
    "reasoning": "详细的分析理由"
}}"""
        

        
        return prompt
    
    def _analyze_with_qwen(self, df: pd.DataFrame) -> Dict:
        """使用Qwen模型进行分析
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            分析结果字典
        """
        if not self.model or not self.tokenizer:
            return self._rule_based_analysis(df)
        
        try:
            import torch
            
            prompt = self._prepare_prompt(df)
            
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
            
            # 解析JSON响应 - 先找FINAL_JSON标记
            final_json_marker = "FINAL_JSON:"
            marker_pos = response.find(final_json_marker)
            
            if marker_pos >= 0:
                # 找到标记，从标记后面开始找JSON
                json_part = response[marker_pos + len(final_json_marker):]
                json_start = json_part.find('{')
                json_end = json_part.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = json_part[json_start:json_end]
                    if len(json_str) > 10:
                        result = json.loads(json_str)
                        return self._convert_to_standard_format(result)
            
            # 备用方案：找最后一个JSON
            json_start = response.rfind('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                if len(json_str) > 10:
                    result = json.loads(json_str)
                    return self._convert_to_standard_format(result)
            
            # 如果解析失败，使用规则分析
            return self._rule_based_analysis(df)
            
        except Exception as e:
            print(f"Qwen模型分析失败: {e}")
            return self._rule_based_analysis(df)
    
    def _rule_based_analysis(self, df: pd.DataFrame) -> Dict:
        """基于规则的分析（备用方法）
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            分析结果字典
        """
        recent_df = df.tail(50).copy()
        
        closes = recent_df['close'].values
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        current_price = closes[-1]
        
        # 计算移动平均线
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
        ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        
        # 计算支撑阻力位
        recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
        recent_low = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
        
        # 确定趋势方向
        trend_direction = "NEUTRAL"
        trend_strength = 0.003
        
        # MA排列判断趋势
        if ma5 > ma10 and ma10 > ma20:
            trend_direction = "LONG"
            # 计算趋势强度
            ma_diff = (ma5 - ma20) / ma20
            trend_strength = 0.003 + min(ma_diff * 10, 0.007)
        elif ma5 < ma10 and ma10 < ma20:
            trend_direction = "SHORT"
            ma_diff = (ma20 - ma5) / ma20
            trend_strength = 0.003 + min(ma_diff * 10, 0.007)
        
        # 价格相对于MA的位置
        if trend_direction == "LONG" and current_price > ma5:
            trend_strength = min(trend_strength * 1.2, 0.01)
        elif trend_direction == "SHORT" and current_price < ma5:
            trend_strength = min(trend_strength * 1.2, 0.01)
        
        # 检测拐点
        has_turning_point = False
        recent_turn_type = None
        
        if len(closes) >= 10:
            # 检测最近是否有明显拐点
            recent_closes = closes[-10:]
            max_idx = np.argmax(recent_closes)
            min_idx = np.argmin(recent_closes)
            
            # 如果最大值或最小值在中间位置，可能是拐点
            if max_idx in [3, 4, 5, 6] and max_idx != 0 and max_idx != 9:
                has_turning_point = True
                recent_turn_type = "PEAK"
            elif min_idx in [3, 4, 5, 6] and min_idx != 0 and min_idx != 9:
                has_turning_point = True
                recent_turn_type = "VALLEY"
        
        # 计算支撑阻力位
        pred_support = recent_low * 0.998
        pred_resistance = recent_high * 1.002
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": max(0.002, min(trend_strength, 0.01)),
            "pred_support": pred_support,
            "pred_resistance": pred_resistance,
            "has_turning_point": has_turning_point,
            "recent_turn_type": recent_turn_type,
            "price_change_pct": (current_price - closes[-2]) / closes[-2],
            "current_price": current_price,
            "signal_valid": True,
            "confidence": 0.7,
            "reasoning": "基于技术指标规则分析",
            "analysis_method": "规则分析"
        }
    
    def _convert_to_standard_format(self, qwen_result: Dict) -> Dict:
        """将Qwen结果转换为标准格式
        
        Args:
            qwen_result: Qwen返回的结果
            
        Returns:
            标准格式的信号
        """
        # 确保趋势方向是标准格式
        trend_dir_val = qwen_result.get("trend_direction", "NEUTRAL")
        trend_dir = str(trend_dir_val).upper() if trend_dir_val is not None else "NEUTRAL"
        if trend_dir not in ["LONG", "SHORT", "NEUTRAL"]:
            if "BUY" in trend_dir or "LONG" in trend_dir:
                trend_dir = "LONG"
            elif "SELL" in trend_dir or "SHORT" in trend_dir:
                trend_dir = "SHORT"
            else:
                trend_dir = "NEUTRAL"
        
        # 确保趋势强度在正确范围内
        trend_strength_val = qwen_result.get("trend_strength", 0.003)
        try:
            trend_strength = float(trend_strength_val) if trend_strength_val is not None else 0.003
        except (ValueError, TypeError):
            trend_strength = 0.003
        trend_strength = max(0.002, min(trend_strength, 0.01))
        
        # 获取支撑阻力位
        pred_support_val = qwen_result.get("pred_support", 0)
        try:
            pred_support = float(pred_support_val) if pred_support_val is not None else 0
        except (ValueError, TypeError):
            pred_support = 0
        
        pred_resistance_val = qwen_result.get("pred_resistance", 0)
        try:
            pred_resistance = float(pred_resistance_val) if pred_resistance_val is not None else 0
        except (ValueError, TypeError):
            pred_resistance = 0
        
        # 获取当前价格
        current_price_val = qwen_result.get("current_price", 0)
        try:
            current_price = float(current_price_val) if current_price_val is not None else 0
        except (ValueError, TypeError):
            current_price = 0
        
        # 拐点信息
        has_turning_point_val = qwen_result.get("has_turning_point", False)
        has_turning_point = bool(has_turning_point_val) if has_turning_point_val is not None else False
        
        recent_turn_type = qwen_result.get("recent_turn_type")
        if recent_turn_type:
            recent_turn_type = str(recent_turn_type).upper()
            if recent_turn_type not in ["PEAK", "VALLEY"]:
                recent_turn_type = None
        
        # 置信度
        confidence_val = qwen_result.get("confidence", 0.7)
        try:
            confidence = float(confidence_val) if confidence_val is not None else 0.7
        except (ValueError, TypeError):
            confidence = 0.7
        
        return {
            "trend_direction": trend_dir,
            "trend_strength": trend_strength,
            "pred_support": pred_support,
            "pred_resistance": pred_resistance,
            "current_price": current_price,
            "has_turning_point": has_turning_point,
            "recent_turn_type": recent_turn_type,
            "signal_valid": True,
            "confidence": confidence,
            "reasoning": qwen_result.get("reasoning", ""),
            "analysis_method": "Qwen模型分析"
        }
    
    def get_enhanced_signal(self, df: pd.DataFrame) -> Dict:
        """获取增强信号（与Kronos接口兼容）
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            标准格式的交易信号
        """
        # 计算27个Kronos指标（不显示过程）
        df_with_features = None
        try:
            df_with_features = self._calculate_kronos_features(df)
        except Exception as e:
            print(f"  [Qwen] 计算指标时出错: {e}")
        
        if df.empty or len(df) < 20:
            print(f"  [Qwen] ⚠️ 数据不足，返回无效信号")
            return {
                "trend_direction": "NEUTRAL",
                "trend_strength": 0.0,
                "pred_support": 0,
                "pred_resistance": 0,
                "has_turning_point": False,
                "recent_turn_type": None,
                "signal_valid": False,
                "error": "数据不足"
            }
        
        # 检查缓存
        cache_key = f"qwen_{self.symbol}_{hash(str(df.tail(20).to_json()))}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                print(f"  [Qwen] 使用缓存的分析结果")
                print(f"\n{'='*80}")
                print(f"[Qwen] 分析结果（缓存）")
                print(f"{'='*80}")
                print(f"  趋势方向: {cached_data['trend_direction']}")
                print(f"  趋势强度: {cached_data['trend_strength']:.4f}")
                print(f"  支撑位: ${cached_data['pred_support']:.2f}")
                print(f"  阻力位: ${cached_data['pred_resistance']:.2f}")
                print(f"  拐点检测: {cached_data['has_turning_point']}")
                print(f"  分析方法: {cached_data.get('analysis_method', '未知')}")
                print(f"{'='*80}\n")
                return cached_data
        
        print(f"\n[Qwen] 开始分析 {self.symbol} K线数据...")
        
        # 执行分析
        if self.use_local_model and self.model:
            result = self._analyze_with_qwen(df)
        else:
            result = self._rule_based_analysis(df)
        
        print(f"\n{'='*80}")
        print(f"[Qwen] 分析结果")
        print(f"{'='*80}")
        print(f"  趋势方向: {result['trend_direction']}")
        print(f"  趋势强度: {result['trend_strength']:.4f}")
        print(f"  支撑位: ${result['pred_support']:.2f}")
        print(f"  阻力位: ${result['pred_resistance']:.2f}")
        print(f"  拐点检测: {result['has_turning_point']}")
        print(f"  分析方法: {result.get('analysis_method', '未知')}")
        print(f"{'='*80}\n")
        
        # 更新缓存
        self.cache[cache_key] = (time.time(), result)
        
        return result


# 全局实例
_qwen_analyzer_instance = None


def get_qwen_analyzer(symbol: str = "BTC", use_local_model: bool = True) -> QwenAnalyzer:
    """获取Qwen分析器单例
    
    Args:
        symbol: 交易品种符号
        use_local_model: 是否使用本地模型
        
    Returns:
        Qwen分析器实例
    """
    global _qwen_analyzer_instance
    
    if _qwen_analyzer_instance is None:
        _qwen_analyzer_instance = QwenAnalyzer(
            use_local_model=use_local_model,
            symbol=symbol
        )
    
    return _qwen_analyzer_instance


# 使用示例
if __name__ == "__main__":
    print("Qwen分析器测试")
    
    # 创建分析器
    analyzer = QwenAnalyzer(use_local_model=False, symbol="BTC")
    
    # 创建示例数据
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5T")
    np.random.seed(42)
    base_price = 50000
    prices = base_price + np.cumsum(np.random.normal(0, 100, 100))
    
    sample_data = pd.DataFrame({
        "open": prices,
        "high": prices + np.random.uniform(0, 50, 100),
        "low": prices - np.random.uniform(0, 50, 100),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, 100),
        "amt": np.random.uniform(50000000, 250000000, 100)
    }, index=dates)
    
    # 分析数据
    print("\n正在分析示例数据...")
    result = analyzer.get_enhanced_signal(sample_data)
    
    print(f"\n=== Qwen分析结果 ===")
    print(f"趋势方向: {result['trend_direction']}")
    print(f"趋势强度: {result['trend_strength']:.4f}")
    print(f"支撑位: ${result['pred_support']:.2f}")
    print(f"阻力位: ${result['pred_resistance']:.2f}")
    print(f"拐点检测: {result['has_turning_point']}")
    if result['has_turning_point']:
        print(f"拐点类型: {result['recent_turn_type']}")
    print(f"分析方法: {result.get('analysis_method', '未知')}")
