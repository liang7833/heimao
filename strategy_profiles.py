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
        "basic": {
            "LEVERAGE": 10,
            "TREND_STRENGTH_THRESHOLD": 0.010,
            "LOOKBACK_PERIOD": 128,
            "PREDICTION_LENGTH": 36,
            "CHECK_INTERVAL": 300
        },
        "ai_filter": {
            "min_trend_strength": 0.010,
            "require_strong_direction": False,
            "min_price_deviation": 0.008,
            "pred_lookahead": 12,
        },
        "entry": {
            "max_kline_change": 0.01,
            "max_funding_rate_long": 0.02,
            "min_funding_rate_short": -0.02,
            "support_buffer": 1.002,
            "resistance_buffer": 0.998
        },
        "stop_loss": {
            "long_buffer": 0.97,
            "short_buffer": 1.03
        },
        "take_profit": {
            "tp1_multiplier_long": 1.04,
            "tp2_multiplier_long": 1.08,
            "tp3_multiplier_long": 1.2,
            "tp1_multiplier_short": 0.96,
            "tp2_multiplier_short": 0.92,
            "tp3_multiplier_short": 0.8,
            "tp1_position_ratio": 0.25,
            "tp2_position_ratio": 0.35,
            "tp3_position_ratio": 0.4
        },
        "risk": {
            "single_trade_risk": 0.035,
            "daily_loss_limit": 0.15,
            "max_consecutive_losses": 5,
            "max_single_position": 0.35,
            "max_daily_position": 0.9
        },
        "frequency": {
            "max_daily_trades": 15,
            "min_trade_interval_minutes": 30,
            "active_hours_start": 0,
            "active_hours_end": 24
        },
        "position": {
            "initial_entry_ratio": 0.3,
            "confirm_interval_kline": 4,
            "add_on_profit": True,
            "add_ratio": 0.35,
            "max_add_times": 3
        },
        "strategy": {
            "entry_confirm_count": 2,
            "reverse_confirm_count": 3,
            "require_consecutive_prediction": 3,
            "post_entry_hours": 12.0,
            "take_profit_min_pct": 0.8
        }
    }

    RANGE_ARBITRAGE = {
        "name": "Kronos_震荡套利_5min",
        "description": "依托Kronos预测的高低点，在区间上沿空、下沿多",
        "basic": {
            "LEVERAGE": 10,
            "TREND_STRENGTH_THRESHOLD": 0.0025,
            "LOOKBACK_PERIOD": 64,
            "PREDICTION_LENGTH": 12,
            "CHECK_INTERVAL": 120
        },
        "ai_filter": {
            "max_trend_strength": 0.008,
            "max_support_resistance_distance": 0.040,
            "price_at_extreme_pct": 0.20,
            "min_price_deviation": 0.003,
        },
        "entry": {
            "max_kline_change": 0.008,
            "max_funding_rate_long": 0.015,
            "min_funding_rate_short": -0.015,
            "support_buffer": 1.0015,
            "resistance_buffer": 0.9985
        },
        "stop_loss": {
            "long_buffer": 0.993,
            "short_buffer": 1.007
        },
        "take_profit": {
            "tp1_multiplier_long": 1.012,
            "tp2_multiplier_long": 1.022,
            "tp3_multiplier_long": 1.035,
            "tp1_multiplier_short": 0.988,
            "tp2_multiplier_short": 0.978,
            "tp3_multiplier_short": 0.965,
            "tp1_position_ratio": 0.4,
            "tp2_position_ratio": 0.4,
            "tp3_position_ratio": 0.2
        },
        "risk": {
            "single_trade_risk": 0.015,
            "daily_loss_limit": 0.08,
            "max_consecutive_losses": 8,
            "max_single_position": 0.25,
            "max_daily_position": 0.7
        },
        "frequency": {
            "max_daily_trades": 35,
            "min_trade_interval_minutes": 8,
            "active_hours_start": 0,
            "active_hours_end": 24
        },
        "position": {
            "initial_entry_ratio": 0.4,
            "confirm_interval_kline": 2,
            "add_on_profit": False,
            "add_ratio": 0.0,
            "max_add_times": 0
        },
        "strategy": {
            "entry_confirm_count": 2,
            "reverse_confirm_count": 1,
            "require_consecutive_prediction": 2,
            "post_entry_hours": 4.0,
            "take_profit_min_pct": 0.25
        }
    }

    NEWS_BREAKOUT = {
        "name": "Kronos_消息突破_5min",
        "description": "配合FinGPT舆情，等待消息确认第一波真实突破方向后跟随入场",
        "basic": {
            "LEVERAGE": 10,
            "TREND_STRENGTH_THRESHOLD": 0.015,
            "LOOKBACK_PERIOD": 80,
            "PREDICTION_LENGTH": 18,
            "CHECK_INTERVAL": 90
        },
        "ai_filter": {
            "min_trend_strength": 0.015,
            "require_consecutive_prediction": 2,
            "min_price_deviation": 0.01,
            "require_fingpt_signal": True,
        },
        "entry": {
            "max_kline_change": 0.02,
            "max_funding_rate_long": 0.04,
            "min_funding_rate_short": -0.04,
            "support_buffer": 1.0008,
            "resistance_buffer": 0.9992
        },
        "stop_loss": {
            "long_buffer": 0.982,
            "short_buffer": 1.018
        },
        "take_profit": {
            "tp1_multiplier_long": 1.02,
            "tp2_multiplier_long": 1.045,
            "tp3_multiplier_long": 1.09,
            "tp1_multiplier_short": 0.98,
            "tp2_multiplier_short": 0.955,
            "tp3_multiplier_short": 0.91,
            "tp1_position_ratio": 0.4,
            "tp2_position_ratio": 0.35,
            "tp3_position_ratio": 0.25
        },
        "risk": {
            "single_trade_risk": 0.035,
            "daily_loss_limit": 0.15,
            "max_consecutive_losses": 5,
            "max_single_position": 0.32,
            "max_daily_position": 0.88
        },
        "frequency": {
            "max_daily_trades": 8,
            "min_trade_interval_minutes": 45,
            "active_hours_start": 0,
            "active_hours_end": 24
        },
        "position": {
            "initial_entry_ratio": 0.4,
            "confirm_interval_kline": 2,
            "add_on_profit": True,
            "add_ratio": 0.4,
            "max_add_times": 2
        },
        "strategy": {
            "entry_confirm_count": 2,
            "reverse_confirm_count": 2,
            "require_consecutive_prediction": 2,
            "post_entry_hours": 5.0,
            "take_profit_min_pct": 0.4
        }
    }

    AUTO_STRATEGY = {
        "name": "Kronos_自动策略_5min",
        "description": "自动判断市场状态，动态切换趋势/震荡策略",
        "basic": {
            "LEVERAGE": 10,
            "TREND_STRENGTH_THRESHOLD": 0.0047,
            "LOOKBACK_PERIOD": 96,
            "PREDICTION_LENGTH": 24,
            "CHECK_INTERVAL": 180
        },
        "ai_filter": {
            "min_trend_strength": 0.0047,
            "min_price_deviation": 0.005,
        },
        "entry": {
            "max_kline_change": 0.015,
            "max_funding_rate_long": 0.03,
            "min_funding_rate_short": -0.03,
            "support_buffer": 1.001,
            "resistance_buffer": 0.999
        },
        "stop_loss": {
            "long_buffer": 0.99,
            "short_buffer": 1.01
        },
        "take_profit": {
            "tp1_multiplier_long": 1.025,
            "tp2_multiplier_long": 1.05,
            "tp3_multiplier_long": 1.14,
            "tp1_multiplier_short": 0.975,
            "tp2_multiplier_short": 0.95,
            "tp3_multiplier_short": 0.86,
            "tp1_position_ratio": 0.35,
            "tp2_position_ratio": 0.35,
            "tp3_position_ratio": 0.3
        },
        "risk": {
            "single_trade_risk": 0.029,
            "daily_loss_limit": 0.12,
            "max_consecutive_losses": 6,
            "max_single_position": 0.29,
            "max_daily_position": 0.85
        },
        "frequency": {
            "max_daily_trades": 20,
            "min_trade_interval_minutes": 10,
            "active_hours_start": 0,
            "active_hours_end": 24
        },
        "position": {
            "initial_entry_ratio": 0.35,
            "confirm_interval_kline": 3,
            "add_on_profit": True,
            "add_ratio": 0.25,
            "max_add_times": 3
        },
        "strategy": {
            "entry_confirm_count": 3,
            "reverse_confirm_count": 2,
            "require_consecutive_prediction": 2,
            "post_entry_hours": 6.0,
            "take_profit_min_pct": 0.5
        }
    }

    TIME_STRATEGY = {
        "name": "Kronos_时间策略_5min",
        "description": "根据BTC交易活跃性时段特点，亚洲盘震荡策略，欧美盘趋势策略",
        "basic": {
            "LEVERAGE": 10,
            "TREND_STRENGTH_THRESHOLD": 0.006,
            "LOOKBACK_PERIOD": 96,
            "PREDICTION_LENGTH": 24,
            "CHECK_INTERVAL": 180
        },
        "ai_filter": {
            "min_trend_strength": 0.006,
            "min_price_deviation": 0.006,
        },
        "entry": {
            "max_kline_change": 0.018,
            "max_funding_rate_long": 0.035,
            "min_funding_rate_short": -0.035,
            "support_buffer": 1.0012,
            "resistance_buffer": 0.9988
        },
        "stop_loss": {
            "long_buffer": 0.985,
            "short_buffer": 1.015
        },
        "take_profit": {
            "tp1_multiplier_long": 1.03,
            "tp2_multiplier_long": 1.06,
            "tp3_multiplier_long": 1.12,
            "tp1_multiplier_short": 0.97,
            "tp2_multiplier_short": 0.94,
            "tp3_multiplier_short": 0.88,
            "tp1_position_ratio": 0.3,
            "tp2_position_ratio": 0.35,
            "tp3_position_ratio": 0.35
        },
        "risk": {
            "single_trade_risk": 0.032,
            "daily_loss_limit": 0.14,
            "max_consecutive_losses": 5,
            "max_single_position": 0.32,
            "max_daily_position": 0.88
        },
        "frequency": {
            "max_daily_trades": 18,
            "min_trade_interval_minutes": 15,
            "active_hours_start": 0,
            "active_hours_end": 24
        },
        "position": {
            "initial_entry_ratio": 0.32,
            "confirm_interval_kline": 3,
            "add_on_profit": True,
            "add_ratio": 0.3,
            "max_add_times": 2
        },
        "strategy": {
            "entry_confirm_count": 2,
            "reverse_confirm_count": 2,
            "require_consecutive_prediction": 2,
            "post_entry_hours": 8.0,
            "take_profit_min_pct": 0.6
        },
        "time_slots": {
            "asia": {
                "hours": [0, 1, 2, 3, 4, 5, 6, 7],
                "description": "亚洲盘 - 震荡套利",
                "use_range_strategy": True
            },
            "europe_morning": {
                "hours": [8, 9, 10, 11],
                "description": "欧美早盘 - 趋势追踪",
                "use_range_strategy": False
            },
            "overlap": {
                "hours": [12, 13, 14, 15],
                "description": "欧美重叠 - 趋势追踪",
                "use_range_strategy": False
            },
            "us_morning": {
                "hours": [16, 17, 18, 19],
                "description": "美国早盘 - 消息突破",
                "use_range_strategy": False
            },
            "us_afternoon": {
                "hours": [20, 21, 22, 23],
                "description": "美国午盘 - 震荡套利",
                "use_range_strategy": True
            }
        }
    }

    @staticmethod
    def get_profile(strategy_type: str):
        profiles = {
            "trend": StrategyProfiles.TREND_BURST,
            "range": StrategyProfiles.RANGE_ARBITRAGE,
            "breakout": StrategyProfiles.NEWS_BREAKOUT,
            "auto": StrategyProfiles.AUTO_STRATEGY,
            "time": StrategyProfiles.TIME_STRATEGY,
        }
        return profiles.get(strategy_type, StrategyProfiles.AUTO_STRATEGY)

    @staticmethod
    def get_default_params(strategy_type: str):
        profile = StrategyProfiles.get_profile(strategy_type)
        return {
            "basic": profile["basic"].copy(),
            "entry": profile["entry"].copy(),
            "stop_loss": profile["stop_loss"].copy(),
            "take_profit": profile["take_profit"].copy(),
            "risk": profile["risk"].copy(),
            "frequency": profile["frequency"].copy(),
            "position": profile["position"].copy(),
            "strategy": profile["strategy"].copy(),
            "ai_filter": profile.get("ai_filter", {}).copy(),
        }
