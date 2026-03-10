"""策略配置文件 - 由参数集成器自动生成"""

class StrategyConfig:
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '5m'
    LEVERAGE = 10
    TREND_STRENGTH_THRESHOLD = 0.0047
    LOOKBACK_PERIOD = 64
    PREDICTION_LENGTH = 10
    CHECK_INTERVAL = 180
    ENTRY_FILTER = {
        "max_kline_change": 0.015,
        "max_funding_rate_long": 0.03,
        "min_funding_rate_short": -0.03,
        "support_buffer": 1.001,
        "resistance_buffer": 0.999,
    }

    STOP_LOSS = {
        "long_buffer": 0.996,
        "short_buffer": 1.004,
    }

    TAKE_PROFIT = {
        "tp1_multiplier_long": 1.025,
        "tp2_multiplier_long": 1.05,
        "tp3_multiplier_long": 1.14,
        "tp1_multiplier_short": 0.975,
        "tp2_multiplier_short": 0.95,
        "tp3_multiplier_short": 0.86,
        "tp1_position_ratio": 0.35,
        "tp2_position_ratio": 0.35,
        "tp3_position_ratio": 0.3,
    }

    RISK_MANAGEMENT = {
        "single_trade_risk": 0.029,
        "daily_loss_limit": 0.12,
        "max_consecutive_losses": 6,
        "max_single_position": 0.29,
        "max_daily_position": 0.85,
    }

    TRADE_FREQUENCY = {
        "max_daily_trades": 30,
        "min_trade_interval_minutes": 5,
        "active_hours_start": 0,
        "active_hours_end": 0,
    }

    POSITION_MANAGEMENT = {
        "initial_entry_ratio": 0.35,
        "confirm_interval_kline": 1,
        "add_on_profit": True,
        "add_ratio": 1.0,
        "max_add_times": 3,
    }

    STRATEGY_CONFIG = {
        "entry_confirm_count": 1,
        "reverse_confirm_count": 2,
        "require_consecutive_prediction": 2,
        "post_entry_hours": 6.0,
        "take_profit_min_pct": 0.5,
    }
