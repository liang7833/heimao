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


# 训练用的特征列表（和 Kronos/finetune_csv/config.py 一致）
CUSTOM_MODEL_FEATURES = [
    "open", "high", "low", "close", "vol", "amt",  # 基础数据
    "MA5", "MA10", "MA20",                           # 移动平均线
    "BIAS20",                                          # 乖离率
    "ATR14", "AMPLITUDE",                             # 波动性指标
    "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",         # 成交量指标
    "RSI14", "RSI7",                                  # 动量指标
    "MACD", "MACD_HIST",                              # MACD指标
    "PRICE_SLOPE5", "PRICE_SLOPE10",                  # 趋势指标
    "HIGH5", "LOW5", "HIGH10", "LOW10",              # 极值指标
    "VOL_BREAKOUT", "VOL_SHRINK"                       # 成交量突破
]

def load_kronos_models(model_name: str):
    try:
        from Kronos.model.kronos import Kronos, KronosTokenizer, KronosPredictor
        
        if model_name.startswith("custom:"):
            custom_name = model_name[7:]
            
            # 尝试多个路径
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "Kronos", "finetune_csv", "Kronos", "finetune_csv", "finetuned", custom_name),
                os.path.join(os.path.dirname(__file__), "Kronos", "finetune_csv", "finetuned", custom_name),
                os.path.join(os.path.dirname(__file__), "custom_models", custom_name),
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
                    predictor = KronosPredictor(model, tokenizer, max_context=512, feature_list=CUSTOM_MODEL_FEATURES)
                    print(f"✓ 自定义模型全部加载成功! (特征数: {len(CUSTOM_MODEL_FEATURES)})")
                    return tokenizer, model, predictor, CUSTOM_MODEL_FEATURES
                else:
                    print(f"  自定义模型子目录不存在: {tokenizer_path} 或 {model_path}")
            else:
                print(f"  未找到自定义模型: {custom_name}，尝试路径: {possible_paths}")
        
        # 官方模型使用默认特征列表
        OFFICIAL_MODEL_FEATURES = ["open", "high", "low", "close", "volume", "amount"]
        
        # 兼容 PyInstaller 打包环境
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(__file__)
        
        # 检查 _internal/models 目录（打包环境），如果不存在再检查 models 目录
        models_dir = os.path.join(base_dir, "_internal", "models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(base_dir, "models")
        
        # 支持的官方模型列表
        official_models = {
            "kronos-small": "NeoQuasar/Kronos-small",
            "kronos-base": "NeoQuasar/Kronos-base",
            "kronos-mini": "NeoQuasar/Kronos-mini"
        }
        
        if model_name in official_models:
            tokenizer_path = os.path.join(models_dir, "kronos-tokenizer-base")
            model_path = os.path.join(models_dir, model_name)
            
            if os.path.exists(tokenizer_path) and os.path.exists(model_path):
                print(f"  尝试从本地目录加载...")
                try:
                    print(f"  从本地路径加载 tokenizer: {tokenizer_path}")
                    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
                    print(f"  ✓ Tokenizer 从本地加载成功")
                    print(f"  从本地路径加载 model: {model_path}")
                    model = Kronos.from_pretrained(model_path)
                    print(f"  ✓ Model 从本地加载成功 (n_layers={model.n_layers})")
                    predictor = KronosPredictor(model, tokenizer, max_context=512, feature_list=OFFICIAL_MODEL_FEATURES)
                    print(f"✓ {model_name} 模型全部加载成功! (特征数: {len(OFFICIAL_MODEL_FEATURES)})")
                    return tokenizer, model, predictor, OFFICIAL_MODEL_FEATURES
                except Exception as e:
                    print(f"  本地加载失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"  从 HuggingFace Hub 加载...")
            try:
                tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                model = Kronos.from_pretrained(official_models[model_name])
                predictor = KronosPredictor(model, tokenizer, max_context=512, feature_list=OFFICIAL_MODEL_FEATURES)
                return tokenizer, model, predictor, OFFICIAL_MODEL_FEATURES
            except Exception as e:
                print(f"  Hub 加载失败: {e}")
        
    except Exception as e:
        print(f"Kronos 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
    
    return None, None, None, None


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
    def __init__(self, model_name="custom:custom_model"):
        print("正在加载分析器...")
        self.model_name = model_name
        self.use_kronos = False
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.feature_list = None

        self.alpha_processor = AlphaSignalProcessor()

        # 加载Kronos模型
        try:
            print(f"使用Kronos {model_name}模型...")
            tokenizer, model, predictor, feature_list = load_kronos_models(model_name)

            if tokenizer and model and predictor:
                self.tokenizer = tokenizer
                self.model = model
                self.predictor = predictor
                self.feature_list = feature_list
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
                raise RuntimeError(f"Kronos模型加载失败: tokenizer, model, predictor返回空值")
        except Exception as e:
            print(f"❌ Kronos加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Kronos模型加载失败，交易停止: {e}")
        
        print(f"✓ 分析器初始化完成: use_kronos={self.use_kronos}")

    def _calculate_slope(self, values):
        """
        计算线性回归斜率，用于判断趋势
        
        参数:
            values: 价格数组
            
        返回:
            slope: 斜率值（正数表示上涨，负数表示下跌）
        """
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # 简单线性回归
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

    def calculate_kronos_features(self, df):
        """计算Kronos预测所需的技术指标特征"""
        df = df.copy()

        # 处理时间戳列兼容性
        if "timestamp" in df.columns:
            df["timestamps"] = df["timestamp"]  # 确保有 timestamps 列名
        
        # 基础价格数据
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        df["open"].values
        
        # 安全获取amount列（训练时用的是 amt 和 vol）
        if "amt" in df.columns:
            amount = df["amt"].values
        elif "amount" in df.columns:
            amount = df["amount"].values
            df["amt"] = amount  # 添加训练时用的列名
        elif "volume" in df.columns:
            amount = df["volume"].values
            df["amt"] = amount  # 添加训练时用的列名
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

        # 12. 放量突破(0/1) - 和训练时保持一致（仅成交量条件）
        df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int).values

        # 13. 缩量下跌(0/1) - 和训练时保持一致（仅成交量条件）
        df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int).values

        return df

    def get_enhanced_signal(self, df, analysis_callback=None):
        if not self.use_kronos or not self.predictor:
            raise RuntimeError("Kronos模型不可用，无法进行分析")
        
        try:
            print("使用Kronos AI进行预测...")

            # 计算技术指标特征（始终计算所有27个指标，用于后续分析）
            df_with_features = self.calculate_kronos_features(df)

            # 使用模型实际需要的特征列表
            if self.feature_list is not None:
                kronos_columns = self.feature_list
                print(f"  使用模型特征列表: {len(kronos_columns)}个特征")
            else:
                # 如果没有特征列表，回退到默认
                kronos_columns = CUSTOM_MODEL_FEATURES
                print(f"  回退到默认特征列表: {len(kronos_columns)}个特征")
            
            # 去除 NaN 值（移动平均线等指标会产生前几个 NaN）
            df_with_features = df_with_features.dropna().copy()
            print(f"  去除 NaN 后数据行数: {len(df_with_features)}")

            # 使用去除 NaN 后的时间戳（兼容不同列名）
            if "timestamp" in df_with_features.columns:
                timestamps = pd.DatetimeIndex(df_with_features["timestamp"])
            elif "timestamps" in df_with_features.columns:
                timestamps = pd.DatetimeIndex(df_with_features["timestamps"])
            elif df_with_features.index.name == "timestamp":
                timestamps = df_with_features.index
            else:
                # 使用默认时间戳
                timestamps = pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(minutes=len(df_with_features)*5),
                    periods=len(df_with_features),
                    freq="5T"
                )
            
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
            
            # ============================================
            # Kronos AI 趋势判断（基于 Kronos 的预测价格）
            # ============================================
            # 我们已经把所有技术指标传给 Kronos 了
            # 现在信任 Kronos 基于这些指标的综合判断
            # ============================================
            
            # 1. 看预测的整体变化
            pred_change = (pred_prices[-1] - pred_prices[0]) / pred_prices[0] if len(pred_prices) > 1 else 0
            trend_up = pred_change > 0
            
            # 2. 趋势强度 = 预测变化的绝对值
            final_pred_price = pred_prices[-1]
            price_change_pct = (final_pred_price - current_price) / current_price
            volatility = np.std(pred_prices) / np.mean(pred_prices)
            trend_strength = abs(pred_change) * (1 + volatility * 0.2)
            
            pred_low = pred_df["low"].min()
            pred_high = pred_df["high"].max()

            # 打印趋势判断信息
            print(f"  [Kronos趋势分析]")
            print(f"    预测整体变化: {'UP' if trend_up else 'DOWN'} (变化: {pred_change*100:+.3f}%)")
            print(f"    最终方向: {'LONG' if trend_up else 'SHORT'}")
            print(f"    趋势强度: {trend_strength:.4f} (阈值: {StrategyConfig.TREND_STRENGTH_THRESHOLD})")
            
            # 调用回调函数更新图表
            if analysis_callback is not None:
                try:
                    # 准备历史K线和预测K线数据
                    # 最近30根历史K线
                    historical_df = df.tail(30).copy()
                    if "timestamp" in historical_df.columns:
                        history_timestamps = pd.DatetimeIndex(historical_df["timestamp"])
                    elif "timestamps" in historical_df.columns:
                        history_timestamps = pd.DatetimeIndex(historical_df["timestamps"])
                    elif historical_df.index.name == "timestamp":
                        history_timestamps = historical_df.index
                    else:
                        history_timestamps = pd.date_range(
                            start=pd.Timestamp.now() - pd.Timedelta(minutes=len(historical_df)*5),
                            periods=len(historical_df),
                            freq="5T"
                        )
                    history_prices = historical_df["close"].values
                    
                    # 预测K线
                    if "timestamp" in pred_df.columns:
                        pred_timestamps = pd.DatetimeIndex(pred_df["timestamp"])
                    else:
                        pred_timestamps = y_timestamp
                    pred_prices = pred_df["close"].values
                    
                    analysis_callback(
                        history_timestamps=history_timestamps,
                        history_prices=history_prices,
                        pred_timestamps=pred_timestamps,
                        pred_prices=pred_prices,
                        trend_direction='LONG' if trend_up else 'SHORT',
                        trend_strength=trend_strength,
                        pred_change=pred_change,
                        threshold=StrategyConfig.TREND_STRENGTH_THRESHOLD
                    )
                except Exception as e:
                    print(f"  [警告] 回调函数执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ============================================
            # Kronos AI 自主计算止盈止损（基于预测波动率和ATR）
            # ============================================
            atr_value = df_with_features["ATR14"].iloc[-1] if "ATR14" in df_with_features.columns else (current_price * 0.008)
            atr_pct = atr_value / current_price
            
            # 基于预测波动率动态计算止盈止损
            pred_volatility = np.std(pred_prices) / np.mean(pred_prices) if len(pred_prices) > 1 else 0.008
            
            # 止损 = 1.5 * ATR 或 0.8% 取较大者（保护本金）
            sl_pct = max(1.5 * atr_pct, 0.008)
            
            # 止盈1 = 1.2 * ATR（保守）
            tp1_pct = max(1.2 * atr_pct, 0.006)
            
            # 止盈2 = 2.5 * ATR（激进）
            tp2_pct = max(2.5 * atr_pct, 0.015)
            
            print(f"  [Kronos止盈止损计算]")
            print(f"    ATR: {atr_value:.2f} ({atr_pct*100:.2f}%)")
            print(f"    预测波动率: {pred_volatility*100:.2f}%")
            print(f"    止损: {sl_pct*100:.2f}%, 止盈1: {tp1_pct*100:.2f}%, 止盈2: {tp2_pct*100:.2f}%")
            
            # 计算具体价格
            if trend_up:
                ai_stop_loss = current_price * (1 - sl_pct)
                ai_take_profit_1 = current_price * (1 + tp1_pct)
                ai_take_profit_2 = current_price * (1 + tp2_pct)
            else:
                ai_stop_loss = current_price * (1 + sl_pct)
                ai_take_profit_1 = current_price * (1 - tp1_pct)
                ai_take_profit_2 = current_price * (1 - tp2_pct)
            
            print(f"    AI推荐止损: ${ai_stop_loss:.2f}")
            print(f"    AI推荐止盈1: ${ai_take_profit_1:.2f}, 止盈2: ${ai_take_profit_2:.2f}")
            
            # 创建原始信号（包含AI自主计算的止盈止损）
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
                "analysis_method": "Kronos AI (信任27个技术指标)",
                "ai_stop_loss": ai_stop_loss,
                "ai_take_profit_1": ai_take_profit_1,
                "ai_take_profit_2": ai_take_profit_2,
                "atr_value": atr_value,
                "atr_pct": atr_pct,
                "pred_volatility": pred_volatility,
                "sl_pct": sl_pct,
                "tp1_pct": tp1_pct,
                "tp2_pct": tp2_pct
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
            print(f"❌ Kronos预测失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Kronos预测失败: {e}")

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
