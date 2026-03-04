"""策略配置文件 - 由参数集成器自动生成"""

class StrategyConfig:
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '5m'
    LEVERAGE = 10
    TREND_STRENGTH_THRESHOLD = 0.012
    LOOKBACK_PERIOD = 512
    PREDICTION_LENGTH = 120
    CHECK_INTERVAL = 300
    ENTRY_FILTER = {
        "max_kline_change": 0.003,
        "max_funding_rate_long": 0.03,
        "min_funding_rate_short": -0.03,
        "support_buffer": 1.001,
        "resistance_buffer": 0.999,
    }

    STOP_LOSS = {
        "long_buffer": 0.99,
        "short_buffer": 1.01,
    }

    TAKE_PROFIT = {
        "tp1_multiplier_long": 1.015,
        "tp2_multiplier_long": 1.018,
        "tp1_multiplier_short": 0.99,
        "tp2_multiplier_short": 0.982,
        "tp1_position_ratio": 0.5,
    }

    RISK_MANAGEMENT = {
        "single_trade_risk": 0.1,
        "daily_loss_limit": 0.03,
        "max_consecutive_losses": 2,
        "pause_after_losses_minutes": 30,
        "max_single_position": 0.15,
        "max_daily_position": 0.5,
        "extreme_move_threshold": 0.015,
    }

    TRADE_FREQUENCY = {
        "max_daily_trades": 8,
        "min_trade_interval_minutes": 10,
        "active_hours_start": 0,
        "active_hours_end": 24,
    }

    POSITION_MANAGEMENT = {
        "initial_entry_ratio": 0.5,
        "confirm_interval_kline": 3,
    }
