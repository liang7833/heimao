"""策略配置文件 - 由参数集成器自动生成"""

class StrategyConfig:
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '5m'
    LEVERAGE = 10
    TREND_STRENGTH_THRESHOLD = 0.0025
    LOOKBACK_PERIOD = 64
    PREDICTION_LENGTH = 12
    CHECK_INTERVAL = 120
    ENTRY_FILTER = {
        "max_kline_change": 0.008,
        "max_funding_rate_long": 0.015,
        "min_funding_rate_short": -0.015,
        "support_buffer": 1.0015,
        "resistance_buffer": 0.9985,
    }

    STOP_LOSS = {
        "long_buffer": 0.993,
        "short_buffer": 1.007,
    }

    TAKE_PROFIT = {
        "tp1_multiplier_long": 0.962,
        "tp2_multiplier_long": 1.022,
        "tp3_multiplier_long": 1.035,
        "tp1_multiplier_short": 1.038,
        "tp2_multiplier_short": 0.978,
        "tp3_multiplier_short": 0.965,
        "tp1_position_ratio": 0.4,
        "tp2_position_ratio": 0.4,
        "tp3_position_ratio": 0.2,
    }

    RISK_MANAGEMENT = {
        "single_trade_risk": 0.015,
        "daily_loss_limit": 0.08,
        "max_consecutive_losses": 8,
        "max_single_position": 0.25,
        "max_daily_position": 0.7,
    }

    TRADE_FREQUENCY = {
        "max_daily_trades": 35,
        "min_trade_interval_minutes": 8,
        "active_hours_start": 0,
        "active_hours_end": 24,
    }

    POSITION_MANAGEMENT = {
        "initial_entry_ratio": 0.4,
        "confirm_interval_kline": 2,
        "add_on_profit": False,
        "add_ratio": 0.0,
        "max_add_times": 0,
    }

    STRATEGY_CONFIG = {
        "entry_confirm_count": 2,
        "reverse_confirm_count": 2,
        "require_consecutive_prediction": 2,
        "post_entry_hours": 4.0,
        "take_profit_min_pct": 0.25,
    }
