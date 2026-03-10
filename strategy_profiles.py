from enum import Enum


class StrategyType(Enum):
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    AUTO = "auto"
    TIME = "time"


class StrategyProfiles:

    TREND_BURST = {
        "name": "Kronos_趋势爆发_5min",
        "description": "捕捉Kronos预测的高强度趋势，顺势追单",
        "ai_filter": {
            "min_trend_strength": 0.010,
            "require_strong_direction": False,
            "min_price_deviation": 0.008,
            "pred_lookahead": 12,
        },
        "entry_rules": {
            "long_condition": "price > pred_resistance * 0.998",
            "short_condition": "price < pred_support * 1.002",
            "position_type": "full",  # 一次性满仓
        },
        "exit_rules": {
            "stop_loss": {
                "type": "ai_predicted",
                "long_pct": 0.010,
                "short_pct": 0.010,
            },
            "take_profit": {
                "type": "trailing",
                "tp1_pct": 0.008,
                "tp1_move_sl_to_breakeven": True,
                "tp2_pct": 0.015,
                "tp2_sl_offset_pct": 0.008,
                "reverse_signal_exit": True,
            },
        },
        "special_filters": {
            "forbid_in_range": False,
            "max_funding_rate": 0.01,
            "min_funding_rate": -0.01,
        },
    }

    RANGE_ARBITRAGE = {
        "name": "Kronos_震荡套利_5min",
        "description": "依托Kronos预测的高低点，在区间上沿空、下沿多",
        "ai_filter": {
            "max_trend_strength": 0.008,
            "max_support_resistance_distance": 0.040,
            "price_at_extreme_pct": 0.20,
        },
        "entry_rules": {
            "long_condition": "price < pred_support * 1.000",
            "short_condition": "price > pred_resistance * 1.000",
            "position_type": "full",  # 一次性建仓
            "first_position_pct": 0.50,
            "confirm_position_pct": 0.50,
        },
        "exit_rules": {
            "stop_loss": {
                "type": "tight",
                "long_pct": 0.008,
                "short_pct": 0.008,
            },
            "take_profit": {
                "type": "midpoint",
                "target_pct_range": [0.005, 0.010],
                "no_trailing": True,
            },
        },
        "special_filters": {
            "breakout_exit": True,
            "switch_to_trend_on_breakout": True,
            "profit_target_range": [0.005, 0.010],
        },
    }

    NEWS_BREAKOUT = {
        "name": "Kronos_消息突破_5min",
        "description": "等待消息确认第一波真实突破方向后，跟随入场",
        "ai_filter": {
            "min_trend_strength": 0.015,
            "require_consecutive_prediction": 3,
            "activation_after_news_minutes": [5, 30],
        },
        "entry_rules": {
            "wait_candle_complete": False,
            "long_condition": "bullish_candle OR preds_up",
            "short_condition": "bearish_candle OR preds_down",
            "position_type": "full",
        },
        "exit_rules": {
            "stop_loss": {
                "type": "candle_extreme",
                "long_pct": "candle_low_offset",
                "short_pct": "candle_high_offset",
            },
            "take_profit": {
                "type": "fixed",
                "target_pct": 0.015,
                "mode": "lightning",
                "max_hold_minutes": 30,
                "exit_on_weakness": False,
            },
        },
        "special_filters": {
            "min_volatility_after_news": 0.001,
            "max_trades_per_day": 5,
        },
    }

    @staticmethod
    def get_profile(strategy_type: str):
        profiles = {
            "trend": StrategyProfiles.TREND_BURST,
            "range": StrategyProfiles.RANGE_ARBITRAGE,
            "breakout": StrategyProfiles.NEWS_BREAKOUT,
            "auto": StrategyProfiles.TREND_BURST,
            "time": StrategyProfiles.TREND_BURST,
        }
        return profiles.get(strategy_type, StrategyProfiles.TREND_BURST)

    @staticmethod
    def get_default_params(strategy_type: str):
        profile = StrategyProfiles.get_profile(strategy_type)
        return {
            "ai_filter": profile["ai_filter"].copy(),
            "entry_rules": profile["entry_rules"].copy(),
            "exit_rules": {
                "stop_loss": profile["exit_rules"]["stop_loss"].copy(),
                "take_profit": profile["exit_rules"]["take_profit"].copy(),
            },
            "special_filters": profile["special_filters"].copy(),
        }
