import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "Kronos"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Kronos", "model"))

import pandas as pd
import numpy as np
from datetime import datetime
from strategy_config import StrategyConfig
import warnings

warnings.filterwarnings("ignore")

from safetensors.torch import load_file
import json


def convert_timestamps(ts):
    if hasattr(ts, "dt"):
        return ts
    return pd.DatetimeIndex(ts)


pd.DatetimeIndex.dt = property(lambda self: self)


class DatetimeIndexAccessor:
    def __init__(self, index):
        self._index = index

    @property
    def minute(self):
        return self._index.minute

    @property
    def hour(self):
        return self._index.hour

    @property
    def weekday(self):
        return self._index.weekday

    @property
    def day(self):
        return self._index.day

    @property
    def month(self):
        return self._index.month


original_dt = getattr(pd.DatetimeIndex, "dt", None)
pd.DatetimeIndex.dt = property(lambda self: DatetimeIndexAccessor(self))

KronosTokenizer = None
KronosModel = None
KronosPredictor = None
KRONOS_AVAILABLE = False


def load_kronos_models(model_name: str):
    try:
        from Kronos.model.kronos import Kronos, KronosTokenizer, KronosPredictor
        
        if model_name.startswith("custom:"):
            custom_name = model_name[7:]
            
            # 尝试多个路径
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "custom_models", custom_name),
                os.path.join(os.path.dirname(__file__), "Kronos", "finetune_csv", "Kronos", "finetune_csv", "finetuned", custom_name),
                custom_name
            ]
            
            custom_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    custom_path = path
                    print(f"  找到自定义模型: {custom_path}")
                    break
            
            if custom_path:
                tokenizer_path = os.path.join(custom_path, "tokenizer", "best_model")
                model_path = os.path.join(custom_path, "basemodel", "best_model")
                
                if os.path.exists(tokenizer_path) and os.path.exists(model_path):
                    print(f"  从本地路径加载 tokenizer: {tokenizer_path}")
                    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
                    print(f"  从本地路径加载 model: {model_path}")
                    model = Kronos.from_pretrained(model_path)
                    predictor = KronosPredictor(model, tokenizer, max_context=512)
                    print(f"✓ 自定义模型全部加载成功!")
                    return tokenizer, model, predictor
                else:
                    print(f"  自定义模型子目录不存在: {tokenizer_path} 或 {model_path}")
            else:
                print(f"  未找到自定义模型: {custom_name}，尝试路径: {possible_paths}")
        
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        if model_name == "kronos-small":
            tokenizer_path = os.path.join(models_dir, "kronos-tokenizer-base")
            model_path = os.path.join(models_dir, "kronos-small")
            
            if os.path.exists(tokenizer_path) and os.path.exists(model_path):
                print(f"  尝试从本地目录加载...")
                try:
                    print(f"  从本地路径加载 tokenizer: {tokenizer_path}")
                    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
                    print(f"  ✓ Tokenizer 从本地加载成功")
                    print(f"  从本地路径加载 model: {model_path}")
                    model = Kronos.from_pretrained(model_path)
                    print(f"  ✓ Model 从本地加载成功 (n_layers={model.n_layers})")
                    predictor = KronosPredictor(model, tokenizer, max_context=512)
                    print(f"✓ kronos-small 模型全部加载成功!")
                    return tokenizer, model, predictor
                except Exception as e:
                    print(f"  本地加载失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"  从 HuggingFace Hub 加载...")
            try:
                tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
                predictor = KronosPredictor(model, tokenizer, max_context=512)
                return tokenizer, model, predictor
            except Exception as e:
                print(f"  Hub 加载失败: {e}")
        
    except Exception as e:
        print(f"Kronos 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
    
    return None, None, None


class TechnicalAnalyzer:
    def __init__(self):
        print("使用技术分析作为备选方案")
        self.use_kronos = False

    def calculate_indicators(self, df):
        df = df.copy()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        return df

    def analyze(self, df):
        try:
            if df is None or len(df) < 20:
                print(f"技术分析失败：数据不足，需要至少20条数据，当前只有 {len(df) if df is not None else 0} 条")
                current_price = float(df.iloc[-1]['close']) if df is not None and len(df) > 0 else 0
                return {
                    'trend_direction': 'NEUTRAL', 
                    'trend_strength': 0.0, 
                    'price_change_pct': 0.0,
                    'current_price': current_price,
                    'predicted_price': current_price,
                    'pred_support': current_price * 0.98,
                    'pred_resistance': current_price * 1.02,
                    'pred_low': current_price * 0.98,
                    'pred_high': current_price * 1.02,
                    'historical_df': df,
                    'signal_valid': False, 
                    'analysis_method': 'Fallback (Insufficient Data)'
                }
            
            df_copy = df.copy()
            df_copy = df_copy.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df_copy) < 20:
                print(f"技术分析失败：清理NaN后数据不足")
                current_price = float(df_copy.iloc[-1]['close']) if len(df_copy) > 0 else 0
                return {
                    'trend_direction': 'NEUTRAL', 
                    'trend_strength': 0.0, 
                    'price_change_pct': 0.0,
                    'current_price': current_price,
                    'predicted_price': current_price,
                    'pred_support': current_price * 0.98,
                    'pred_resistance': current_price * 1.02,
                    'pred_low': current_price * 0.98,
                    'pred_high': current_price * 1.02,
                    'historical_df': df,
                    'signal_valid': False, 
                    'analysis_method': 'Fallback (Bad Data)'
                }
            
            df = self.calculate_indicators(df_copy)
            current_price = df["close"].iloc[-1]
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1] if not pd.isna(df["sma_50"].iloc[-1]) else sma_20
            rsi = df["rsi"].iloc[-1]
            macd = df["macd"].iloc[-1]
            macd_signal = df["macd_signal"].iloc[-1]

            long_signals = 0
            short_signals = 0

            if current_price > sma_20:
                long_signals += 1
            else:
                short_signals += 1
            if sma_20 > sma_50:
                long_signals += 1
            else:
                short_signals += 1
            if rsi < 30:
                long_signals += 1
            elif rsi > 70:
                short_signals += 1
            if macd > macd_signal:
                long_signals += 1
            else:
                short_signals += 1

            recent = df["close"].iloc[-10:]
            price_change = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]

            if long_signals > short_signals:
                trend_direction = "LONG"
                trend_strength = (long_signals / 5) * abs(price_change) * 10
            else:
                trend_direction = "SHORT"
                trend_strength = (short_signals / 5) * abs(price_change) * 10

            volatility = df["close"].std() / df["close"].mean()

            return {
                "trend_direction": trend_direction,
                "trend_strength": min(trend_strength, 1.0),
                "price_change_pct": price_change,
                "current_price": current_price,
                "predicted_price": current_price * (1 + price_change * 0.5),
                "pred_support": current_price * (1 - volatility * 2),
                "pred_resistance": current_price * (1 + volatility * 2),
                "pred_low": current_price * (1 - volatility * 2),
                "pred_high": current_price * (1 + volatility * 2),
                "historical_df": df,
                "signal_valid": trend_strength >= StrategyConfig.TREND_STRENGTH_THRESHOLD,
                "analysis_method": "Technical Analysis",
            }
        except Exception as e:
            print(f"技术分析失败：{e}")
            import traceback
            traceback.print_exc()
            current_price = float(df.iloc[-1]['close']) if df is not None and len(df) > 0 else 0
            return {
                'trend_direction': 'NEUTRAL', 
                'trend_strength': 0.0, 
                'price_change_pct': 0.0,
                'current_price': current_price,
                'predicted_price': current_price,
                'pred_support': current_price * 0.98,
                'pred_resistance': current_price * 1.02,
                'pred_low': current_price * 0.98,
                'pred_high': current_price * 1.02,
                'historical_df': df,
                'signal_valid': False, 
                'analysis_method': 'Fallback'
            }


class AlphaSignalProcessor:
    """Alpha信号处理器 - 将原始Kronos信号转换为标准化Alpha信号"""

    def __init__(self, lookback_period=100):
        self.lookback_period = lookback_period
        self.signal_history = []
        self.market_state = "neutral"
        self.volatility_state = "normal"

    def process(self, raw_signal, historical_df=None):
        """将原始信号归一化到标准化Alpha值

        Args:
            raw_signal: 原始Kronos信号字典
            historical_df: 历史K线数据，用于市场状态分析

        Returns:
            处理后的Alpha信号字典
        """
        if raw_signal is None:
            return None

        # 提取原始信号的关键指标
        trend_strength = raw_signal.get("trend_strength", 0)
        price_change_pct = raw_signal.get("price_change_pct", 0)
        current_price = raw_signal.get("current_price", 0)
        pred_support = raw_signal.get("pred_support", current_price)
        pred_resistance = raw_signal.get("pred_resistance", current_price)
        trend_direction = raw_signal.get("trend_direction", "NEUTRAL")

        # 1. 趋势强度归一化 (0-1范围)
        # Kronos趋势强度通常在0-0.1之间，我们将其归一化到0-1
        normalized_trend_strength = min(
            trend_strength * 10, 1.0
        )  # 假设趋势强度最大0.1对应1.0

        # 2. 价格偏离度标准化 (基于历史波动性)
        normalized_price_deviation = 0.0
        if historical_df is not None and len(historical_df) > 20:
            # 计算历史波动性
            returns = historical_df["close"].pct_change().dropna()
            historical_volatility = returns.std() * np.sqrt(
                252 * 288
            )  # 5分钟数据，一年约252天*288个5分钟
            current_deviation = abs(price_change_pct)
            if historical_volatility > 0:
                # 将当前偏离度与历史波动性比较
                normalized_price_deviation = min(
                    current_deviation / (historical_volatility * 2), 1.0
                )

        # 3. 支撑阻力位距离分析
        sr_distance_pct = 0.0
        if current_price > 0 and pred_resistance > pred_support:
            # 计算价格相对于支撑阻力位的相对位置
            if trend_direction == "LONG":
                # 做多信号：计算距离阻力位的空间
                distance_to_resistance = (
                    pred_resistance - current_price
                ) / current_price
                sr_distance_pct = max(
                    min(distance_to_resistance / 0.05, 1.0), 0
                )  # 假设5%为最大预期空间
            elif trend_direction == "SHORT":
                # 做空信号：计算距离支撑位的空间
                distance_to_support = (current_price - pred_support) / current_price
                sr_distance_pct = max(min(distance_to_support / 0.05, 1.0), 0)

        # 4. 市场状态识别
        market_state_info = (
            self._analyze_market_state(historical_df)
            if historical_df is not None
            else {"state": "neutral", "volatility": "normal", "trendiness": 0.5}
        )

        # 5. 信号质量评估
        signal_quality = self._evaluate_signal_quality(raw_signal, historical_df)

        # 6. 综合Alpha计算
        # 权重分配：趋势强度40%，价格偏离度30%，支撑阻力空间20%，信号质量10%
        alpha_score = (
            normalized_trend_strength * 0.4
            + normalized_price_deviation * 0.3
            + sr_distance_pct * 0.2
            + signal_quality * 0.1
        )

        # 7. 根据市场状态调整Alpha
        adjusted_alpha = self._adjust_alpha_by_market_state(
            alpha_score, market_state_info
        )

        # 构建Alpha信号
        alpha_signal = {
            # 原始信号保留
            **raw_signal,
            # Alpha处理信息
            "alpha_score": adjusted_alpha,
            "normalized_trend_strength": normalized_trend_strength,
            "normalized_price_deviation": normalized_price_deviation,
            "sr_distance_pct": sr_distance_pct,
            "signal_quality": signal_quality,
            "market_state": market_state_info["state"],
            "market_volatility": market_state_info["volatility"],
            "market_trendiness": market_state_info.get("trendiness", 0.5),
            # 信号分类
            "signal_category": self._classify_signal(adjusted_alpha, trend_direction),
            "confidence_level": self._calculate_confidence_level(
                adjusted_alpha, signal_quality
            ),
            # 时间戳
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_alpha_signal": True,
        }

        # 更新信号历史
        self._update_signal_history(alpha_signal)

        return alpha_signal

    def _analyze_market_state(self, df):
        """分析市场状态：趋势/震荡，波动性高低"""
        if df is None or len(df) < 50:
            return {"state": "neutral", "volatility": "normal", "trendiness": 0.5}

        # 计算趋势性
        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]

        # 1. 波动性分析
        volatility = np.std(returns) * np.sqrt(252 * 288)  # 年化波动率
        volatility_percentile = np.percentile(
            np.abs(returns) * 100, 75
        )  # 75分位数的绝对回报

        if volatility_percentile > 0.3:  # 高波动
            volatility_state = "high"
        elif volatility_percentile < 0.1:  # 低波动
            volatility_state = "low"
        else:
            volatility_state = "normal"

        # 2. 趋势性分析
        # 使用ADX-like指标判断趋势强度
        close = df["close"].values

        # 简化趋势性计算
        sma_short = np.convolve(close, np.ones(20) / 20, mode="valid")
        sma_long = np.convolve(close, np.ones(50) / 50, mode="valid")

        if len(sma_short) > 0 and len(sma_long) > 0:
            trend_strength = abs(sma_short[-1] - sma_long[-1]) / close[-1]
        else:
            trend_strength = 0

        if trend_strength > 0.02:  # 强趋势
            market_state = "trending"
            trendiness = min(trend_strength / 0.05, 1.0)  # 归一化到0-1
        elif trend_strength < 0.005:  # 震荡
            market_state = "ranging"
            trendiness = 0.0
        else:  # 中性
            market_state = "neutral"
            trendiness = 0.5

        return {
            "state": market_state,
            "volatility": volatility_state,
            "trendiness": trendiness,
            "volatility_value": volatility,
            "trend_strength": trend_strength,
        }

    def _evaluate_signal_quality(self, signal, historical_df):
        """评估信号质量：一致性、稳定性、可靠性"""
        if signal is None:
            return 0.5  # 默认中等质量

        quality_factors = []

        # 1. 趋势方向一致性
        trend_direction = signal.get("trend_direction", "NEUTRAL")
        price_change_pct = signal.get("price_change_pct", 0)

        if trend_direction == "LONG" and price_change_pct > 0:
            quality_factors.append(0.8)  # 方向一致
        elif trend_direction == "SHORT" and price_change_pct < 0:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.3)  # 方向不一致

        # 2. 预测置信度（基于预测区间）
        pred_low = signal.get("pred_low", 0)
        pred_high = signal.get("pred_high", 0)
        current_price = signal.get("current_price", 0)

        if current_price > 0 and pred_high > pred_low:
            prediction_range = (pred_high - pred_low) / current_price
            # 预测区间越小，置信度越高（但也不能太小）
            if prediction_range < 0.02:  # 2%以内
                confidence = 0.9
            elif prediction_range < 0.05:  # 5%以内
                confidence = 0.7
            elif prediction_range < 0.1:  # 10%以内
                confidence = 0.5
            else:
                confidence = 0.3
            quality_factors.append(confidence)

        # 3. 历史信号表现（如果有历史数据）
        if len(self.signal_history) > 10:
            recent_signals = self.signal_history[-10:]
            successful_signals = sum(
                1 for s in recent_signals if s.get("alpha_score", 0) > 0.6
            )
            success_rate = successful_signals / len(recent_signals)
            quality_factors.append(success_rate)
        else:
            quality_factors.append(0.5)  # 默认中等成功率

        # 计算平均质量
        if quality_factors:
            return np.mean(quality_factors)
        else:
            return 0.5

    def _adjust_alpha_by_market_state(self, alpha_score, market_state_info):
        """根据市场状态调整Alpha分数"""
        adjusted_alpha = alpha_score

        market_state = market_state_info.get("state", "neutral")
        volatility_state = market_state_info.get("volatility", "normal")
        trendiness = market_state_info.get("trendiness", 0.5)

        # 趋势市场：增强趋势信号的Alpha
        if market_state == "trending":
            adjusted_alpha *= 1.0 + trendiness * 0.2  # 最多增强20%

        # 震荡市场：降低所有信号的Alpha（更难盈利）
        elif market_state == "ranging":
            adjusted_alpha *= 0.8  # 降低20%

        # 高波动市场：更加谨慎
        if volatility_state == "high":
            adjusted_alpha *= 0.9  # 降低10%
        elif volatility_state == "low":
            adjusted_alpha *= 1.1  # 低波动市场可以稍微激进

        return max(0, min(adjusted_alpha, 1.0))  # 确保在0-1范围内

    def _classify_signal(self, alpha_score, trend_direction):
        """根据Alpha分数对信号进行分类"""
        if alpha_score >= 0.8:
            strength = "STRONG"
        elif alpha_score >= 0.6:
            strength = "MODERATE"
        elif alpha_score >= 0.4:
            strength = "WEAK"
        else:
            strength = "NOISE"

        return f"{strength}_{trend_direction}"

    def _calculate_confidence_level(self, alpha_score, signal_quality):
        """计算信号置信度"""
        confidence = alpha_score * 0.7 + signal_quality * 0.3

        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"

    def _update_signal_history(self, signal):
        """更新信号历史记录"""
        self.signal_history.append(signal)

        # 保持历史记录长度
        if len(self.signal_history) > self.lookback_period:
            self.signal_history = self.signal_history[-self.lookback_period :]

    def get_signal_statistics(self):
        """获取信号统计信息"""
        if not self.signal_history:
            return {}

        alphas = [s.get("alpha_score", 0) for s in self.signal_history]
        qualities = [s.get("signal_quality", 0.5) for s in self.signal_history]

        return {
            "total_signals": len(self.signal_history),
            "avg_alpha_score": np.mean(alphas) if alphas else 0,
            "std_alpha_score": np.std(alphas) if len(alphas) > 1 else 0,
            "avg_signal_quality": np.mean(qualities) if qualities else 0,
            "strong_signals": sum(1 for a in alphas if a >= 0.7),
            "weak_signals": sum(1 for a in alphas if a < 0.3),
        }


class EnhancedKronosAnalyzer:
    def __init__(self, model_name="kronos-small"):
        print("正在加载分析器...")
        self.model_name = model_name
        self.use_kronos = False
        self.tokenizer = None
        self.model = None
        self.predictor = None

        self.tech_analyzer = TechnicalAnalyzer()
        self.alpha_processor = AlphaSignalProcessor()

        # 根据model_name动态加载模型
        try:
            print(f"使用Kronos {model_name}模型...")
            tokenizer, model, predictor = load_kronos_models(model_name)

            if tokenizer and model and predictor:
                self.tokenizer = tokenizer
                self.model = model
                self.predictor = predictor
                self.use_kronos = True
                
                # 将模型移到 GPU（如果可用）
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model = self.model.to("cuda")
                        print(f"✓ Kronos模型已移至 GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        print("  Kronos模型将使用 CPU")
                except Exception as e:
                    print(f"  警告: 无法将Kronos模型移至GPU: {e}")
                
                print("✓ Kronos模型就绪!")
            else:
                print(f"警告: Kronos模型加载返回空值，使用技术分析...")
                self.use_kronos = False
        except Exception as e:
            print(f"Kronos加载失败: {e}")
            import traceback

            traceback.print_exc()

        # 确保tech_analyzer始终存在
        if not hasattr(self, "tech_analyzer") or self.tech_analyzer is None:
            self.tech_analyzer = TechnicalAnalyzer()

    def calculate_kronos_features(self, df):
        """计算Kronos预测所需的技术指标特征"""
        df = df.copy()

        # 基础价格数据
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        df["open"].values
        
        # 安全获取amount列，如果不存在则使用volume列
        if "amount" in df.columns:
            amount = df["amount"].values
        elif "volume" in df.columns:
            amount = df["volume"].values
        else:
            amount = np.zeros(len(df))  # 如果都没有，使用零数组

        # 1. MA5, MA10, MA20
        df["MA5"] = pd.Series(close).rolling(window=5).mean().values
        df["MA10"] = pd.Series(close).rolling(window=10).mean().values
        df["MA20"] = pd.Series(close).rolling(window=20).mean().values

        # 2. 收盘价/MA20 乖离率
        df["BIAS20"] = (close / df["MA20"] - 1) * 100

        # 3. ATR(14)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
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
        rs14 = avg_gain14 / avg_loss14
        df["RSI14"] = (100 - (100 / (1 + rs14))).values

        avg_gain7 = gain.rolling(window=7).mean()
        avg_loss7 = loss.rolling(window=7).mean()
        rs7 = avg_gain7 / avg_loss7
        df["RSI7"] = (100 - (100 / (1 + rs7))).values

        # 8. MACD线 和 MACD柱
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        # 9. 价格斜率(5期) 和 (10期)
        df["PRICE_SLOPE5"] = (
            pd.Series(close)
            .rolling(window=5)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True,
            )
            .values
        )
        df["PRICE_SLOPE10"] = (
            pd.Series(close)
            .rolling(window=10)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
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

        # 12. 放量突破(0/1) - 放量突破5日高点
        df["VOL_BREAKOUT"] = (
            ((amount > df["AMOUNT_MA5"] * 1.5) & (close > df["HIGH5"]))
            .astype(int)
            .values
        )

        # 13. 缩量下跌(0/1) - 缩量跌破5日低点
        df["VOL_SHRINK"] = (
            ((amount < df["AMOUNT_MA5"] * 0.5) & (close < df["LOW5"]))
            .astype(int)
            .values
        )

        return df

    def get_enhanced_signal(self, df):
        # 确保tech_analyzer存在
        if not hasattr(self, "tech_analyzer") or self.tech_analyzer is None:
            print("警告: tech_analyzer未初始化，创建新的TechnicalAnalyzer")
            self.tech_analyzer = TechnicalAnalyzer()

        if self.use_kronos and self.predictor:
            try:
                print("使用Kronos AI进行预测...")

                # 计算技术指标特征
                df_with_features = self.calculate_kronos_features(df)

                # 选择Kronos需要的列（使用完整的技术指标特征集）
                # 注意：重新训练后的模型将见过所有技术指标
                kronos_columns = [
                    "open", "high", "low", "close", "amount",  # 基础价格数据
                    "MA5", "MA10", "MA20", "BIAS20",            # 移动平均线和乖离率
                    "ATR14", "AMPLITUDE",                       # 波动性指标
                    "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",   # 成交量指标
                    "RSI14", "RSI7",                            # 动量指标
                    "MACD", "MACD_HIST",                        # MACD指标
                    "PRICE_SLOPE5", "PRICE_SLOPE10",            # 趋势指标
                    "HIGH5", "LOW5", "HIGH10", "LOW10",         # 极值指标
                    "VOL_BREAKOUT", "VOL_SHRINK"                 # 成交量突破
                ]

                timestamps = pd.DatetimeIndex(df["timestamps"])
                y_timestamp = pd.date_range(
                    start=timestamps[-1] + pd.Timedelta(minutes=5),
                    periods=StrategyConfig.PREDICTION_LENGTH,
                    freq="5T",
                )

                pred_df = self.predictor.predict(
                    df=df_with_features[kronos_columns],
                    x_timestamp=timestamps,
                    y_timestamp=y_timestamp,
                    pred_len=StrategyConfig.PREDICTION_LENGTH,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                )

                current_price = df["close"].iloc[-1]
                pred_prices = pred_df["close"].values
                price_changes = np.diff(pred_prices)
                trend_up = np.mean(price_changes) > 0
                final_pred_price = pred_prices[-1]
                price_change_pct = (final_pred_price - current_price) / current_price
                volatility = np.std(pred_prices) / np.mean(pred_prices)
                trend_strength = abs(price_change_pct) * (1 + volatility * 0.5)
                pred_low = pred_df["low"].min()
                pred_high = pred_df["high"].max()

                # 创建原始信号
                raw_signal = {
                    "trend_direction": "LONG" if trend_up else "SHORT",
                    "trend_strength": trend_strength,
                    "price_change_pct": price_change_pct,
                    "current_price": current_price,
                    "predicted_price": final_pred_price,
                    "pred_support": pred_low * 0.998,
                    "pred_resistance": pred_high * 1.002,
                    "pred_low": pred_low,
                    "pred_high": pred_high,
                    "prediction_df": pred_df,
                    "historical_df": df,
                    "signal_valid": trend_strength
                    >= StrategyConfig.TREND_STRENGTH_THRESHOLD,
                    "analysis_method": "Kronos AI",
                }

                # 使用Alpha处理器转换为标准化Alpha信号
                alpha_signal = self.alpha_processor.process(raw_signal, df)

                # 打印Alpha信号信息
                if alpha_signal.get("is_alpha_signal", False):
                    print(f"Alpha信号生成完成!")
                    print(f"  Alpha分数: {alpha_signal.get('alpha_score', 0):.3f}")
                    print(
                        f"  信号类别: {alpha_signal.get('signal_category', 'UNKNOWN')}"
                    )
                    print(
                        f"  置信度: {alpha_signal.get('confidence_level', 'UNKNOWN')}"
                    )
                    print(f"  市场状态: {alpha_signal.get('market_state', 'unknown')}")
                    print(f"  信号质量: {alpha_signal.get('signal_quality', 0.5):.3f}")

                return alpha_signal
            except Exception as e:
                print(f"Kronos预测失败: {e}")
                import traceback

                traceback.print_exc()

        # 如果Kronos不可用，返回技术分析信号
        tech_signal = self.tech_analyzer.analyze(df)
        # 尝试将技术分析信号也转换为Alpha信号
        try:
            alpha_tech_signal = self.alpha_processor.process(tech_signal, df)
            if alpha_tech_signal:
                return alpha_tech_signal
        except:
            pass

        return tech_signal

    def should_trade(self, signal):
        if not signal.get("signal_valid", False):
            return False, "趋势强度不足"
        if signal["trend_strength"] < StrategyConfig.TREND_STRENGTH_THRESHOLD:
            return False, "趋势强度低于阈值"
        return True, "信号有效"

    def analyze_market_state(self, df):
        try:
            returns = df['close'].pct_change()
            volatility = returns.std()
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            trendiness = abs(ma5 - ma20) / ma20 if ma20 > 0 else 0
            vol_state = 'high' if volatility > 0.02 else ('low' if volatility < 0.01 else 'normal')
            trend_state = 'trending' if trendiness > 0.02 else 'ranging'
            return {'market_volatility': vol_state, 'market_trendiness': trendiness, 'market_state': trend_state, 'volatility': float(volatility)}
        except Exception as e:
            return {'market_volatility': 'unknown', 'market_trendiness': 0.0, 'market_state': 'unknown', 'volatility': 0.0}


def create_analyzer(model_name="kronos-small"):
    return EnhancedKronosAnalyzer(model_name)


def get_signal(analyzer, df):
    return analyzer.get_enhanced_signal(df)
