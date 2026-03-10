import numpy as np
import pandas as pd


class MarketStateAnalyzer:
    """市场状态分析器 - 自动判断当前市场适合趋势还是震荡策略"""

    def __init__(self, lookback_candles=100):
        self.lookback_candles = lookback_candles
        self.state_history = []

    def analyze(self, df):
        """分析市场状态

        Returns:
            dict: {
                'state': 'trend' | 'range' | 'breakout',
                'strength': 0.0-1.0,
                'confidence': 0.0-1.0,
                'indicators': {...}
            }
        """
        if len(df) < 50:
            return {
                "state": "unknown",
                "strength": 0,
                "confidence": 0,
                "indicators": {},
            }

        df = df.tail(self.lookback_candles).copy()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        indicators = self._calculate_indicators(close, high, low)
        state, strength, confidence = self._determine_state(indicators)

        result = {
            "state": state,
            "strength": strength,
            "confidence": confidence,
            "indicators": indicators,
        }

        self.state_history.append(result)
        if len(self.state_history) > 100:
            self.state_history.pop(0)

        return result

    def _calculate_indicators(self, close, high, low):
        """计算市场状态指标"""
        indicators = {}

        returns = np.diff(close) / close[:-1]

        # 1. 趋势强度指标
        ma20 = pd.Series(close).rolling(20).mean().values
        ma50 = pd.Series(close).rolling(50).mean().values

        if not np.isnan(ma20[-1]) and not np.isnan(ma50[-1]):
            ma_trend = (ma20[-1] - ma50[-1]) / ma50[-1]
        else:
            ma_trend = 0
        indicators["ma_trend"] = ma_trend

        # 2. 波动率指标
        volatility = np.std(returns) * np.sqrt(288)
        indicators["volatility"] = volatility

        # 3. 趋势性指标 (ADX类似)
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )

        plus_di = np.mean(plus_dm[-14:]) / (np.mean(tr[-14:]) + 1e-10) * 100
        minus_di = np.mean(minus_dm[-14:]) / (np.mean(tr[-14:]) + 1e-10) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = dx

        indicators["adx"] = adx
        indicators["plus_di"] = plus_di
        indicators["minus_di"] = minus_di
        indicators["trend_strength"] = adx / 100

        # 4. 区间指标 (价格在区间内波动的程度)
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        range_pct = (high_20 - low_20) / close[-1]
        indicators["range_pct"] = range_pct

        # 5. 动量指标
        rsi_period = 14
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi"] = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50

        # 6. 趋势持续性 (连续同方向K线)
        trend_candles = 0
        for i in range(-1, -min(10, len(returns)) - 1, -1):
            if returns[i] > 0:
                trend_candles += 1
            else:
                break
        for i in range(-1, -min(10, len(returns)) - 1, -1):
            if returns[i] < 0:
                trend_candles += 1
            else:
                break
        indicators["consecutive_trend"] = trend_candles / 10

        # 7. 突破次数
        breakouts = 0
        for i in range(1, len(close)):
            if close[i] > high[i - 1] or close[i] < low[i - 1]:
                breakouts += 1
        indicators["breakout_count"] = breakouts / len(close)

        return indicators

    def _determine_state(self, indicators):
        """根据指标判断市场状态"""
        adx = indicators["adx"]
        range_pct = indicators["range_pct"]
        breakout_count = indicators["breakout_count"]
        consecutive_trend = indicators["consecutive_trend"]
        indicators["volatility"]

        # 趋势判断条件
        trend_score = 0
        if adx > 25:
            trend_score += 0.4
        if consecutive_trend > 0.6:
            trend_score += 0.3
        if breakout_count > 0.15:
            trend_score += 0.3

        # 震荡判断条件
        range_score = 0
        if adx < 20:
            range_score += 0.4
        if range_pct < 0.03:
            range_score += 0.3
        if breakout_count < 0.1:
            range_score += 0.3

        # 突破判断条件
        breakout_score = 0
        if adx > 30:
            breakout_score += 0.5
        if breakout_count > 0.2:
            breakout_score += 0.5

        # 确定状态
        max_score = max(trend_score, range_score, breakout_score)

        if max_score < 0.3:
            state = "range"
            confidence = 0.5
        elif trend_score >= range_score and trend_score >= breakout_score:
            state = "trend"
            confidence = min(trend_score, 1.0)
        elif breakout_score >= trend_score and breakout_score >= range_score:
            state = "breakout"
            confidence = min(breakout_score, 1.0)
        else:
            state = "range"
            confidence = min(range_score, 1.0)

        strength = max_score

        return state, strength, confidence

    def get_recommended_strategy(self):
        """根据历史分析推荐最佳策略"""
        if len(self.state_history) < 10:
            return "trend"

        recent = self.state_history[-10:]
        trend_count = sum(1 for s in recent if s["state"] == "trend")
        range_count = sum(1 for s in recent if s["state"] == "range")
        breakout_count = sum(1 for s in recent if s["state"] == "breakout")

        max_count = max(trend_count, range_count, breakout_count)

        if max_count == trend_count:
            return "trend"
        elif max_count == breakout_count:
            return "breakout"
        else:
            return "range"


class TimeStrategyAnalyzer:
    """时间策略分析器 - 分析哪些时间段适合什么策略"""

    def __init__(self):
        self.time_analysis = {}
        self.hourly_stats = {
            i: {"trend": 0, "range": 0, "breakout": 0, "total": 0} for i in range(24)
        }
        self.weekday_stats = {
            i: {"trend": 0, "range": 0, "breakout": 0, "total": 0} for i in range(7)
        }

    def record_market_state(self, df, market_state):
        """记录市场状态与时间的关系"""
        if len(df) < 1:
            return

        timestamps = (
            df["timestamps"] if "timestamps" in df.columns else df.get("timestamp", [])
        )
        if len(timestamps) == 0:
            return

        latest_time = pd.Timestamp(timestamps.iloc[-1])
        hour = latest_time.hour
        weekday = latest_time.weekday()

        self.hourly_stats[hour][market_state["state"]] += 1
        self.hourly_stats[hour]["total"] += 1

        self.weekday_stats[weekday][market_state["state"]] += 1
        self.weekday_stats[weekday]["total"] += 1

    def get_hourly_recommendation(self):
        """获取当前小时推荐策略"""
        current_hour = pd.Timestamp.now().hour

        stats = self.hourly_stats[current_hour]
        if stats["total"] < 10:
            return self._get_default_recommendation(current_hour)

        trend_pct = stats["trend"] / stats["total"]
        range_pct = stats["range"] / stats["total"]
        breakout_pct = stats["breakout"] / stats["total"]

        # 只比较三个策略类型，排除total键
        strategy_keys = ["trend", "range", "breakout"]
        recommended = max(strategy_keys, key=lambda k: stats[k])
        
        return {
            "trend": trend_pct,
            "range": range_pct,
            "breakout": breakout_pct,
            "recommended": recommended,
            "sample_size": stats["total"],
        }

    def get_weekday_recommendation(self):
        """获取当前星期推荐策略"""
        current_weekday = pd.Timestamp.now().weekday()

        stats = self.weekday_stats[current_weekday]
        if stats["total"] < 10:
            return {"recommended": "trend", "sample_size": 0}

        trend_pct = stats["trend"] / stats["total"]
        range_pct = stats["range"] / stats["total"]
        breakout_pct = stats["breakout"] / stats["total"]

        # 只比较三个策略类型，排除total键
        strategy_keys = ["trend", "range", "breakout"]
        recommended = max(strategy_keys, key=lambda k: stats[k])
        
        return {
            "trend": trend_pct,
            "range": range_pct,
            "breakout": breakout_pct,
            "recommended": recommended,
            "sample_size": stats["total"],
        }

    def _get_default_recommendation(self, hour):
        """根据经验获取默认推荐 (UTC时间)"""
        if 0 <= hour < 8:
            return {"recommended": "range", "reason": "亚洲盘震荡"}
        elif 8 <= hour < 12:
            return {"recommended": "trend", "reason": "欧美早盘"}
        elif 12 <= hour < 16:
            return {"recommended": "trend", "reason": "欧美重叠"}
        elif 16 <= hour < 20:
            return {"recommended": "breakout", "reason": "美国早盘"}
        else:
            return {"recommended": "range", "reason": "美国午盘"}

    def get_best_hours_for_strategy(self, strategy):
        """获取适合指定策略的最佳时间段"""
        best_hours = []
        for hour, stats in self.hourly_stats.items():
            if stats["total"] > 0:
                pct = stats[strategy] / stats["total"]
                best_hours.append((hour, pct))

        best_hours.sort(key=lambda x: x[1], reverse=True)
        return best_hours[:5]

    def get_comprehensive_recommendation(self, market_state_analyzer):
        """综合时间和市场状态给出推荐"""
        time_rec = self.get_hourly_recommendation()
        weekday_rec = self.get_weekday_recommendation()

        pd.Timestamp.now().hour
        pd.Timestamp.now().weekday()

        if market_state_analyzer and len(market_state_analyzer.state_history) > 0:
            market_state = market_state_analyzer.state_history[-1]
            market_rec = market_state["state"]
            market_confidence = market_state["confidence"]
        else:
            market_rec = "trend"
            market_confidence = 0.5

        scores = {"trend": 0, "range": 0, "breakout": 0}

        if "recommended" in time_rec:
            scores[time_rec["recommended"]] += 0.3

        if "recommended" in weekday_rec:
            scores[weekday_rec["recommended"]] += 0.2

        scores[market_rec] += market_confidence * 0.5

        best_strategy = max(scores, key=scores.get)

        return {
            "recommended": best_strategy,
            "scores": scores,
            "time_factor": time_rec,
            "weekday_factor": weekday_rec,
            "market_factor": {"state": market_rec, "confidence": market_confidence},
        }
