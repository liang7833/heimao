import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance_api import BinanceAPI
from enhanced_kronos import EnhancedKronosAnalyzer
from strategy_config import StrategyConfig
from strategy_profiles import StrategyProfiles
from market_state_analyzer import MarketStateAnalyzer, TimeStrategyAnalyzer
from binance.client import Client
from typing import Dict
from color_print import (
    print_open, 
    print_close, 
    print_success, 
    print_error, 
    print_warning, 
    print_info,
    print_signal_buy,
    print_signal_sell,
    print_signal_neutral,
    print_trend_up,
    print_trend_down,
    print_reverse_signal,
    print_ai_decision,
    print_highlight
)

# 多智能体系统组件将在初始化时动态导入
MULTI_AGENT_AVAILABLE = False
FinGPTSentimentAnalyzer = None
StrategyCoordinator = None


class ProfessionalTradingStrategy:

    def __init__(
        self,
        symbol=None,
        leverage=None,
        interval=None,
        model_name=None,
        timeframe=None,
        threshold=None,
        strategy_type="trend",
        min_position=None,
        ai_min_trend=None,
        ai_min_deviation=None,
        max_funding=None,
        min_funding=None,
        analysis_callback=None,
        strategy_config=None,
        binance=None,
        backtest_mode=False,
        log_callback=None,
    ):
        self.analysis_callback = analysis_callback
        self.log_callback = log_callback
        try:
            # 优先使用UTF-8编码加载
            load_dotenv(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试GBK（Windows默认）
                load_dotenv(encoding="gbk")
            except Exception:
                # 最后尝试默认方式
                load_dotenv()
        self.symbol = symbol or StrategyConfig.SYMBOL
        self.timeframe = timeframe or StrategyConfig.TIMEFRAME
        self.leverage = leverage if leverage is not None else StrategyConfig.LEVERAGE
        self.interval = (
            interval if interval is not None else StrategyConfig.CHECK_INTERVAL
        )
        self.backtest_mode = backtest_mode
        self.backtest_current_time = None  # 回测模式下的当前时间
        self.model_name = model_name or os.getenv("MODEL", "kronos-small")
        self.threshold = (
            threshold
            if threshold is not None
            else StrategyConfig.TREND_STRENGTH_THRESHOLD
        )
        self.strategy_type = strategy_type
        self.min_position = min_position or 100.0
        self.ai_min_trend = (
            ai_min_trend
            if ai_min_trend is not None
            else float(os.getenv("AI_MIN_TREND", "0.010"))
        )
        self.ai_min_deviation = (
            ai_min_deviation
            if ai_min_deviation is not None
            else float(os.getenv("AI_MIN_DEVIATION", "0.008"))
        )
        self.max_funding = (
            max_funding
            if max_funding is not None
            else float(os.getenv("MAX_FUNDING", "1.0")) / 100.0
        )
        self.min_funding = (
            min_funding
            if min_funding is not None
            else float(os.getenv("MIN_FUNDING", "-1.0")) / 100.0
        )

        # 从 strategy_config.py 加载策略配置
        strategy_config_dict = getattr(StrategyConfig, "STRATEGY_CONFIG", {})

        # 趋势反转连续确认次数
        self.reverse_confirm_count = strategy_config_dict.get("reverse_confirm_count", 2)
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None

        # 开仓连续确认次数
        self.entry_confirm_count = strategy_config_dict.get("entry_confirm_count", 2)
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        
        # 消息突破策略：连续预测确认次数
        self.require_consecutive_prediction = strategy_config_dict.get("require_consecutive_prediction", 1)
        
        # 开仓后计时参数
        self.post_entry_hours = strategy_config_dict.get("post_entry_hours", 2.0)
        self.take_profit_min_pct = strategy_config_dict.get("take_profit_min_pct", 0.6)
        self.post_entry_time = None  # 开仓时间
        self.post_entry_entry_count = 0  # 开仓后连续同向信号计数
        self.initial_position_size = 0  # 初始开仓仓位大小
        
        # 初始化所有策略配置变量（带默认值）
        self.lookback_period = 91
        self.prediction_length = 90
        self.entry_filter = {
            "max_kline_change": 0.015,
            "max_funding_rate_long": 0.03,
            "min_funding_rate_short": -0.03,
            "support_buffer": 1.001,
            "resistance_buffer": 0.999,
        }
        self.stop_loss_config = {
            "long_buffer": 0.996,
            "short_buffer": 1.004,
        }
        self.take_profit_config = {
            "tp1_multiplier_long": 1.025,
            "tp2_multiplier_long": 1.05,
            "tp3_multiplier_long": 1.14,
            "tp1_multiplier_short": 0.975,
            "tp2_multiplier_short": 0.95,
            "tp3_multiplier_short": 0.86,
            "tp1_position_ratio": 0.35,
            "tp2_position_ratio": 0.35,
            "tp3_position_ratio": 0.30,
        }
        self.risk_management_config = {
            "single_trade_risk": 0.029,
            "daily_loss_limit": 0.12,
            "max_consecutive_losses": 6,
            "max_single_position": 0.29,
            "max_daily_position": 0.85,
            "extreme_move_threshold": 0.02,
        }
        self.trade_frequency_config = {
            "max_daily_trades": 55,
            "min_trade_interval_minutes": 3,
            "active_hours_start": 0,
            "active_hours_end": 24,
        }
        self.position_management_config = {
            "initial_entry_ratio": 0.5,
            "confirm_interval_kline": 2,
            "add_on_profit": 0.01,
            "add_ratio": 0.25,
            "max_add_times": 2,
        }

        self.strategy_profile = StrategyProfiles.get_profile(strategy_type)
        
        # 从策略profile中读取杠杆参数
        basic_config = self.strategy_profile.get("basic", {})
        if "LEVERAGE" in basic_config:
            if leverage is None:
                self.leverage = basic_config["LEVERAGE"]
            elif leverage != basic_config["LEVERAGE"]:
                self._log(f"[警告] 传入的杠杆({leverage})与策略配置的杠杆({basic_config['LEVERAGE']})不一致，使用策略配置")
                self.leverage = basic_config["LEVERAGE"]
        
        # 先从strategy_profile加载默认配置
        self._load_strategy_config()
        
        # 如果提供了完整的策略配置（AI策略配置），则覆盖默认配置
        if strategy_config:
            self.update_config(strategy_config, is_initialization=True)

        # 自动/时间策略分析器
        self.market_analyzer = MarketStateAnalyzer(lookback_candles=100)
        self.time_analyzer = TimeStrategyAnalyzer()
        self.current_effective_strategy = strategy_type
        self.strategy_switch_count = 0
        
        # 策略切换确认计数器（需要2次连续确认）
        self.pending_strategy_switch = None
        self.strategy_switch_confirm_count = 0

        # 初始化币安API - 支持传入模拟API（用于回测）
        if binance is not None:
            self.binance = binance
        else:
            if not self.backtest_mode:
                self.binance = BinanceAPI()
                # 只有真实API才设置杠杆
                self.binance.set_leverage(self.symbol, self.leverage)
            else:
                # 回测模式下使用None的binance（不调用真实API）
                self.binance = None
        
        # 初始化Kronos分析器，失败时直接抛出异常停止交易
        try:
            self.analyzer = EnhancedKronosAnalyzer(model_name=self.model_name)
            if self.analyzer is None:
                raise RuntimeError("EnhancedKronosAnalyzer创建失败，交易停止")
            self._log("✓ Kronos分析器初始化成功")
        except Exception as e:
            self._log(f"❌ Kronos分析器初始化失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            raise RuntimeError(f"Kronos模型不可用，交易停止: {e}")

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trading_paused_until = None
        self.starting_balance = None

        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_3_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False

        self.trade_history = []
        self.risk_manager = None
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.max_drawdown = 0.0
        
        # 持仓状态追踪（用于检测从有持仓变为无持仓）
        self._last_had_position = False
        
        # 预测准确性跟踪
        self.prediction_history = []
        self.current_prediction = None
        
        # 自适应参数优化
        self.adaptive_params = {
            "threshold": self.threshold,
            "entry_confirm_count": self.entry_confirm_count,
            "reverse_confirm_count": self.reverse_confirm_count,
            "last_optimization": datetime.now()
        }

        # 保存默认参数（用于恢复）
        self._save_default_parameters()

        # 初始化多智能体系统
        self.fingpt_analyzer = None
        self.strategy_coordinator = None
        self._initialize_multi_agent_system()

        # 只有非回测模式才初始化真实币安API余额
        if not self.backtest_mode:
            self._initialize_balance()

    def _log(self, message):
        """统一的日志输出方法，根据backtest_mode选择输出方式"""
        if self.backtest_mode and self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def _load_strategy_config(self):
        """从strategy_profile加载完整策略配置"""
        try:
            self._log("[策略配置] 正在从strategy_profile加载完整配置...")
            
            profile = self.strategy_profile
            
            # 加载基础参数
            basic = profile.get("basic", {})
            if "LOOKBACK_PERIOD" in basic:
                self.lookback_period = basic["LOOKBACK_PERIOD"]
                self._log(f"  ✓ LOOKBACK_PERIOD: {self.lookback_period}")
            
            if "PREDICTION_LENGTH" in basic:
                self.prediction_length = basic["PREDICTION_LENGTH"]
                self._log(f"  ✓ PREDICTION_LENGTH: {self.prediction_length}")
            
            if "TREND_STRENGTH_THRESHOLD" in basic:
                self.threshold = basic["TREND_STRENGTH_THRESHOLD"]
                self._log(f"  ✓ TREND_STRENGTH_THRESHOLD: {self.threshold}")
            
            if "CHECK_INTERVAL" in basic:
                self.interval = basic["CHECK_INTERVAL"]
                self._log(f"  ✓ CHECK_INTERVAL: {self.interval}秒")
            
            # 加载入场过滤参数
            entry = profile.get("entry", {})
            if entry:
                self.entry_filter = entry.copy()
                self._log(f"  ✓ ENTRY_FILTER: {self.entry_filter}")
            
            # 加载止损参数
            stop_loss = profile.get("stop_loss", {})
            if stop_loss:
                self.stop_loss_config = stop_loss.copy()
                self._log(f"  ✓ STOP_LOSS: {self.stop_loss_config}")
            
            # 加载止盈参数
            take_profit = profile.get("take_profit", {})
            if take_profit:
                self.take_profit_config = take_profit.copy()
                self._log(f"  ✓ TAKE_PROFIT: {self.take_profit_config}")
            
            # 加载风险管理参数
            risk = profile.get("risk", {})
            if risk:
                self.risk_management_config = risk.copy()
                self._log(f"  ✓ RISK_MANAGEMENT: {self.risk_management_config}")
                
                # 同时更新风险管理器（如果存在）
                if hasattr(self, 'risk_manager') and self.risk_manager:
                    if "single_trade_risk" in risk:
                        self.risk_manager.single_trade_risk = risk["single_trade_risk"]
                    if "daily_loss_limit" in risk:
                        self.risk_manager.daily_loss_limit = risk["daily_loss_limit"]
                    if "max_consecutive_losses" in risk:
                        self.risk_manager.max_consecutive_losses = risk["max_consecutive_losses"]
                    if "max_single_position" in risk:
                        self.risk_manager.max_single_position = risk["max_single_position"]
                    if "max_daily_position" in risk:
                        self.risk_manager.max_daily_position = risk["max_daily_position"]
            
            # 加载交易频率参数
            frequency = profile.get("frequency", {})
            if frequency:
                self.trade_frequency_config = frequency.copy()
                self._log(f"  ✓ TRADE_FREQUENCY: {self.trade_frequency_config}")
            
            # 加载仓位管理参数
            position = profile.get("position", {})
            if position:
                self.position_management_config = position.copy()
                self._log(f"  ✓ POSITION_MANAGEMENT: {self.position_management_config}")
            
            # 加载策略参数
            strategy = profile.get("strategy", {})
            if strategy:
                if "entry_confirm_count" in strategy:
                    self.entry_confirm_count = strategy["entry_confirm_count"]
                    self._log(f"  ✓ entry_confirm_count: {self.entry_confirm_count}")
                
                if "reverse_confirm_count" in strategy:
                    self.reverse_confirm_count = strategy["reverse_confirm_count"]
                    self._log(f"  ✓ reverse_confirm_count: {self.reverse_confirm_count}")
                
                if "require_consecutive_prediction" in strategy:
                    self.require_consecutive_prediction = strategy["require_consecutive_prediction"]
                    self._log(f"  ✓ require_consecutive_prediction: {self.require_consecutive_prediction}")
                
                if "post_entry_hours" in strategy:
                    self.post_entry_hours = strategy["post_entry_hours"]
                    self._log(f"  ✓ post_entry_hours: {self.post_entry_hours}")
                
                if "take_profit_min_pct" in strategy:
                    self.take_profit_min_pct = strategy["take_profit_min_pct"]
                    self._log(f"  ✓ take_profit_min_pct: {self.take_profit_min_pct}")
            
            # 同步风险管理器配置
            self._sync_risk_manager_config()
            
            self._log("[策略配置] 完整配置加载完成！")
        except Exception as e:
            self._log(f"[策略配置] 加载配置失败: {e}")
            import traceback
            self._log(traceback.format_exc())
    
    def update_config(self, config, is_initialization=False):
        """动态更新策略配置
        
        Args:
            config: 配置字典，格式与_strategy_config_file中一致
            is_initialization: 是否是初始化时调用（不设置杠杆）
        """
        try:
            self._log("[策略配置] 正在动态更新配置...")
            
            basic = config.get("basic", {})
            entry = config.get("entry", {})
            stop_loss = config.get("stop_loss", {})
            take_profit = config.get("take_profit", {})
            risk = config.get("risk", {})
            frequency = config.get("frequency", {})
            position = config.get("position", {})
            strategy = config.get("strategy", {})
            
            # 更新基础参数
            if "LEVERAGE" in basic:
                new_leverage = basic["LEVERAGE"]
                if new_leverage != self.leverage:
                    self.leverage = new_leverage
                    if not is_initialization and hasattr(self, 'binance') and self.binance:
                        self.binance.set_leverage(self.symbol, self.leverage)
                    self._log(f"  ✓ LEVERAGE 更新为: {self.leverage}x")
            
            if "POSITION_MULTIPLIER" in basic:
                self._log(f"  ✓ POSITION_MULTIPLIER 更新为: {basic['POSITION_MULTIPLIER']}")
            
            if "TREND_STRENGTH_THRESHOLD" in basic:
                self.threshold = basic["TREND_STRENGTH_THRESHOLD"]
                self._log(f"  ✓ TREND_STRENGTH_THRESHOLD 更新为: {self.threshold}")
            
            if "LOOKBACK_PERIOD" in basic:
                self.lookback_period = basic["LOOKBACK_PERIOD"]
                self._log(f"  ✓ LOOKBACK_PERIOD 更新为: {self.lookback_period}")
            
            if "PREDICTION_LENGTH" in basic:
                self.prediction_length = basic["PREDICTION_LENGTH"]
                self._log(f"  ✓ PREDICTION_LENGTH 更新为: {self.prediction_length}")
            
            if "CHECK_INTERVAL" in basic:
                self.interval = basic["CHECK_INTERVAL"]
                self._log(f"  ✓ CHECK_INTERVAL 更新为: {self.interval}秒")
            
            # 更新入场过滤参数
            if entry:
                if not hasattr(self, 'entry_filter'):
                    self.entry_filter = {}
                self.entry_filter.update(entry)
                self._log(f"  ✓ ENTRY_FILTER 更新为: {self.entry_filter}")
            
            # 更新止损参数
            if stop_loss:
                if not hasattr(self, 'stop_loss_config'):
                    self.stop_loss_config = {}
                self.stop_loss_config.update(stop_loss)
                self._log(f"  ✓ STOP_LOSS 更新为: {self.stop_loss_config}")
            
            # 更新止盈参数
            if take_profit:
                if not hasattr(self, 'take_profit_config'):
                    self.take_profit_config = {}
                self.take_profit_config.update(take_profit)
                self._log(f"  ✓ TAKE_PROFIT 更新为: {self.take_profit_config}")
            
            # 更新风险管理参数
            if risk:
                if not hasattr(self, 'risk_management_config'):
                    self.risk_management_config = {}
                self.risk_management_config.update(risk)
                self._log(f"  ✓ RISK_MANAGEMENT 更新为: {self.risk_management_config}")
                
                # 同时更新风险管理器（如果存在）
                if hasattr(self, 'risk_manager') and self.risk_manager:
                    if "single_trade_risk" in risk:
                        self.risk_manager.single_trade_risk = risk["single_trade_risk"]
                    if "daily_loss_limit" in risk:
                        self.risk_manager.daily_loss_limit = risk["daily_loss_limit"]
                    if "max_consecutive_losses" in risk:
                        self.risk_manager.max_consecutive_losses = risk["max_consecutive_losses"]
                    if "max_single_position" in risk:
                        self.risk_manager.max_single_position = risk["max_single_position"]
                    if "max_daily_position" in risk:
                        self.risk_manager.max_daily_position = risk["max_daily_position"]
            
            # 更新交易频率参数
            if frequency:
                if not hasattr(self, 'trade_frequency_config'):
                    self.trade_frequency_config = {}
                self.trade_frequency_config.update(frequency)
                self._log(f"  ✓ TRADE_FREQUENCY 更新为: {self.trade_frequency_config}")
            
            # 更新仓位管理参数
            if position:
                if not hasattr(self, 'position_management_config'):
                    self.position_management_config = {}
                self.position_management_config.update(position)
                self._log(f"  ✓ POSITION_MANAGEMENT 更新为: {self.position_management_config}")
            
            # 更新策略参数（第9个标签页）
            if strategy:
                if "entry_confirm_count" in strategy:
                    self.entry_confirm_count = strategy["entry_confirm_count"]
                    self._log(f"  ✓ entry_confirm_count 更新为: {self.entry_confirm_count}")
                
                if "reverse_confirm_count" in strategy:
                    self.reverse_confirm_count = strategy["reverse_confirm_count"]
                    self._log(f"  ✓ reverse_confirm_count 更新为: {self.reverse_confirm_count}")
                
                if "require_consecutive_prediction" in strategy:
                    self.require_consecutive_prediction = strategy["require_consecutive_prediction"]
                    self._log(f"  ✓ require_consecutive_prediction 更新为: {self.require_consecutive_prediction}")
                
                if "post_entry_hours" in strategy:
                    self.post_entry_hours = strategy["post_entry_hours"]
                    self._log(f"  ✓ post_entry_hours 更新为: {self.post_entry_hours}")
                
                if "take_profit_min_pct" in strategy:
                    self.take_profit_min_pct = strategy["take_profit_min_pct"]
                    self._log(f"  ✓ take_profit_min_pct 更新为: {self.take_profit_min_pct}")
            
            self._log("[策略配置] 动态配置更新完成！")
            return True
        except Exception as e:
            self._log(f"[策略配置] 更新配置失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            return False

    def _initialize_multi_agent_system(self):
        """初始化多智能体量化交易系统"""
        global MULTI_AGENT_AVAILABLE, FinGPTSentimentAnalyzer, StrategyCoordinator
        
        # 动态导入多智能体系统组件
        if FinGPTSentimentAnalyzer is None or StrategyCoordinator is None:
            try:
                self._log("[多智能体系统] 正在导入模块...")
                from fingpt_analyzer import FinGPTSentimentAnalyzer as FGSA
                from strategy_coordinator import StrategyCoordinator as SC
                FinGPTSentimentAnalyzer = FGSA
                StrategyCoordinator = SC
                MULTI_AGENT_AVAILABLE = True
                self._log("[多智能体系统] 模块导入成功")
            except Exception as e:
                self._log(f"[多智能体系统] 模块导入失败: {e}")
                MULTI_AGENT_AVAILABLE = False
                return
        
        if not MULTI_AGENT_AVAILABLE:
            self._log("[多智能体系统] 模块不可用，跳过初始化")
            return

        try:
            self._log("[多智能体系统] 正在初始化...")

            # 初始化FinGPT舆情分析器
            self._log("  正在初始化FinGPT舆情分析器...")
            self.fingpt_analyzer = FinGPTSentimentAnalyzer(
                use_local_model=True
            )
            self._log("  ✓ FinGPT舆情分析器初始化完成")

            # 初始化策略协调器
            self._log("  正在初始化策略协调器...")
            coin_symbol = self.symbol.replace("USDT", "")
            self.strategy_coordinator = StrategyCoordinator(
                symbol=coin_symbol,
                use_fingpt=True,
                kronos_analyzer=self.analyzer,
                fingpt_analyzer=self.fingpt_analyzer
            )
            self._log("  ✓ 策略协调器初始化完成")
            
            # 将多智能体系统实例传递给Kronos分析器
            self.analyzer.fingpt_analyzer = self.fingpt_analyzer
            self.analyzer.strategy_coordinator = self.strategy_coordinator

            self._log("[多智能体系统] 初始化完成!")

        except Exception as e:
            self._log(f"[多智能体系统] 初始化失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.fingpt_analyzer = None
            self.strategy_coordinator = None

    def _save_default_parameters(self):
        """保存默认参数（用于恢复）"""
        import copy
        self.default_parameters = {
            "threshold": self.threshold,
            "leverage": self.leverage,
            "entry_confirm_count": self.entry_confirm_count,
            "reverse_confirm_count": self.reverse_confirm_count,
            "strategy_profile": copy.deepcopy(self.strategy_profile)
        }
        print("[参数管理] 默认参数已保存")

    def restore_default_parameters(self):
        """恢复默认参数（公共方法）"""
        self._restore_default_parameters()

    def _restore_default_parameters(self):
        """恢复默认参数（内部方法）"""
        import copy
        if hasattr(self, 'default_parameters'):
            self.threshold = self.default_parameters["threshold"]
            self.leverage = self.default_parameters["leverage"]
            self.entry_confirm_count = self.default_parameters["entry_confirm_count"]
            self.reverse_confirm_count = self.default_parameters["reverse_confirm_count"]
            self.strategy_profile = copy.deepcopy(self.default_parameters["strategy_profile"])
            print("[参数管理] 已恢复默认参数")
            print(f"  - 趋势强度阈值: {self.threshold}")
            print(f"  - 杠杆: {self.leverage}x")

    def _apply_strategy_profile_params(self, strategy_params):
        """应用策略配置参数 - 支持所有策略参数"""
        
        def _update_profile_section(profile_section, params_dict, section_name=""):
            """递归更新策略配置的各个部分"""
            for key, value in params_dict.items():
                if key in profile_section:
                    if isinstance(value, dict) and isinstance(profile_section[key], dict):
                        _update_profile_section(profile_section[key], value, f"{section_name}.{key}")
                    else:
                        profile_section[key] = value
                        print(f"    ✓ {section_name}.{key}: {value}")
        
        # 确定当前使用的策略类型
        current_profile_name = self.strategy_profile.get("name", "")
        
        if "TREND_BURST" in strategy_params and "趋势爆发" in current_profile_name:
            tb_params = strategy_params["TREND_BURST"]
            print("  应用趋势爆发策略参数:")
            _update_profile_section(self.strategy_profile, tb_params, "TREND_BURST")
        
        elif "RANGE_ARBITRAGE" in strategy_params and "震荡套利" in current_profile_name:
            ra_params = strategy_params["RANGE_ARBITRAGE"]
            print("  应用震荡套利策略参数:")
            _update_profile_section(self.strategy_profile, ra_params, "RANGE_ARBITRAGE")
        
        elif "NEWS_BREAKOUT" in strategy_params and "消息突破" in current_profile_name:
            nb_params = strategy_params["NEWS_BREAKOUT"]
            print("  应用消息突破策略参数:")
            _update_profile_section(self.strategy_profile, nb_params, "NEWS_BREAKOUT")
        
        else:
            # 如果没有匹配的策略类型，尝试应用所有可能的参数
            print("  应用通用策略参数:")
            for strategy_name, params in strategy_params.items():
                if isinstance(params, dict):
                    _update_profile_section(self.strategy_profile, params, strategy_name)

    def _initialize_balance(self):
        balance = self.binance.get_wallet_balance()
        if balance:
            self.starting_balance = balance
            print(f"初始余额(不含盈亏): ${self.starting_balance:.2f}")
            # 初始化风险管理器
            self.risk_manager = EnhancedRiskManager(self.starting_balance, self.symbol)
            # 从策略配置中加载风险参数
            self._sync_risk_manager_config()
            print(f"风险管理器已初始化")
        else:
            # 如果获取余额失败，使用默认值初始化
            self.starting_balance = 0.0
            self.risk_manager = EnhancedRiskManager(0.0, self.symbol)
            # 从策略配置中加载风险参数
            self._sync_risk_manager_config()
            print("警告: 获取余额失败，使用默认值初始化风险管理器")
        
        # 初始化余额缓存
        self._cached_balance = self.starting_balance
    
    def _get_current_time(self):
        """获取当前时间（支持回测模式）"""
        if self.backtest_mode and self.backtest_current_time is not None:
            return self.backtest_current_time
        return datetime.now()
    
    def _sync_risk_manager_config(self):
        """同步风险管理器配置"""
        if hasattr(self, 'risk_manager') and self.risk_manager is not None and hasattr(self, 'risk_management_config'):
            if "single_trade_risk" in self.risk_management_config:
                self.risk_manager.single_trade_risk = self.risk_management_config["single_trade_risk"]
            if "daily_loss_limit" in self.risk_management_config:
                self.risk_manager.daily_loss_limit = self.risk_management_config["daily_loss_limit"]
            if "max_consecutive_losses" in self.risk_management_config:
                self.risk_manager.max_consecutive_losses = self.risk_management_config["max_consecutive_losses"]
            if "max_single_position" in self.risk_management_config:
                self.risk_manager.max_single_position = self.risk_management_config["max_single_position"]
            if "max_daily_position" in self.risk_management_config:
                self.risk_manager.max_daily_position = self.risk_management_config["max_daily_position"]

    def _reset_position_only(self):
        """只重置持仓状态，不重置开仓确认计数器（用于无持仓时）"""
        # 持仓状态
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_3_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        
        # 预测记录
        self.current_prediction = None
        
        # 趋势反转确认计数器
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None
        
        # 开仓后计时变量
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        
        print("[状态重置] 持仓状态已重置（保留开仓确认计数器）")
    
    def _reset_full_state(self):
        """完全重置所有状态（包括开仓确认计数器，用于平仓后）"""
        # 持仓状态
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_3_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        
        # 开仓确认计数器
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        
        # 趋势反转确认计数器
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None
        
        # 开仓后计时变量
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        
        # 预测记录
        self.current_prediction = None
        
        # 重置最后交易时间（让下次开仓不受时间间隔限制）
        self.last_trade_time = None
        
        print("[状态重置] 所有持仓和开仓状态已完全重置")

    def get_total_balance(self):
        """获取合约账户总资金（包含可用+持仓盈亏）"""
        # 回测模式下直接返回 current_balance
        if self.backtest_mode:
            return self.current_balance
        
        try:
            total_balance = self.binance.get_total_balance()
            if total_balance:
                # 更新风险管理器的余额
                if self.risk_manager:
                    self.risk_manager.update_balance(total_balance)
                # 缓存成功的余额值
                self._cached_balance = total_balance
                return total_balance
            
            # 如果获取失败，返回缓存的余额
            if hasattr(self, '_cached_balance') and self._cached_balance:
                self._log(f"使用缓存的余额: {self._cached_balance}")
                return self._cached_balance
            return 0.0
            
        except Exception as e:
            self._log(f"获取总资金异常: {e}")
            # 网络错误时返回缓存的余额，不中断流程
            if hasattr(self, '_cached_balance') and self._cached_balance:
                self._log(f"使用缓存的余额: {self._cached_balance}")
                return self._cached_balance
            return 0.0

    def get_initial_balance(self):
        """获取初始资金（不含持仓盈亏）"""
        return self.starting_balance

    def check_risk_limits(self):
        now = datetime.now()

        # 即使设置了交易暂停，也允许分析继续，只阻止开仓
        # 所以这里不再检查 trading_paused_until
        # if self.trading_paused_until and now < self.trading_paused_until:
        #     return False, f"交易暂停至 {self.trading_paused_until}"

        current_balance = self.get_total_balance()
        if self.starting_balance and current_balance > 0:
            daily_loss = (
                self.starting_balance - current_balance
            ) / self.starting_balance
            daily_loss_limit = self.risk_management_config.get("daily_loss_limit", 0.12)
            if daily_loss >= daily_loss_limit:
                return False, f"触及单日亏损限制 {daily_loss*100:.1f}%"

        max_daily_trades = self.trade_frequency_config.get("max_daily_trades", 55)
        if self.daily_trades >= max_daily_trades:
            return False, f"触及单日最大交易次数 {self.daily_trades}"

        if self.last_trade_time:
            time_since_last = (now - self.last_trade_time).total_seconds() / 60
            self._log(f"[时间检查] 上次交易时间: {self.last_trade_time.strftime('%H:%M:%S')}, 当前时间: {now.strftime('%H:%M:%S')}, 已过 {time_since_last:.1f} 分钟")
            min_interval = self.trade_frequency_config.get("min_trade_interval_minutes", 3)
            if (
                time_since_last
                < min_interval
            ):
                return (
                    False,
                    f"距离上次交易不足 {min_interval} 分钟",
                )

        # 增强风险管理检查
        if self.risk_manager and not self.backtest_mode:
            # 检查回撤限制 - 如果当前余额为0，跳过这个检查（可能是网络错误）
            if hasattr(self.risk_manager, 'current_balance') and self.risk_manager.current_balance > 0:
                drawdown_ok, drawdown_msg = self.risk_manager.check_drawdown_limits()
                if not drawdown_ok:
                    return False, f"风险管理: {drawdown_msg}"

            # 检查黑天鹅事件
            if self.binance:
                df = self.binance.get_recent_klines(
                    self.symbol, self.timeframe, lookback=50
                )
                if df is not None and len(df) >= 20:
                    black_swan, black_swan_msg = self.risk_manager.check_black_swan_event(
                        df
                    )
                    if black_swan:
                        self._log(f"警告: {black_swan_msg}")
                        # 黑天鹅事件不一定要停止交易，但可以记录警告

        return True, "风控检查通过"

    def _adaptive_parameter_optimization(self, df):
        """自适应参数优化"""
        from datetime import datetime, timedelta
        
        # 回测模式下不进行自适应优化（避免影响回测结果）
        if self.backtest_mode:
            return
        
        # 每30分钟优化一次参数
        if datetime.now() - self.adaptive_params["last_optimization"] < timedelta(minutes=30):
            return
            
        self._log("\n[自适应优化] 开始参数优化...")
        
        # 分析市场波动性
        volatility = df["close"].pct_change().std()
        
        # 根据波动性调整阈值
        if volatility > 0.02:  # 高波动市场
            new_threshold = min(self.threshold * 1.5, 0.03)
            self._log(f"  高波动市场，阈值调整: {self.threshold:.4f} -> {new_threshold:.4f}")
            self.threshold = new_threshold
        elif volatility < 0.005:  # 低波动市场
            new_threshold = max(self.threshold * 0.7, 0.005)
            self._log(f"  低波动市场，阈值调整: {self.threshold:.4f} -> {new_threshold:.4f}")
            self.threshold = new_threshold
            
        # 根据预测准确性调整确认次数
        if len(self.prediction_history) >= 10:
            recent_predictions = self.prediction_history[-10:]
            accuracy_rate = sum(1 for p in recent_predictions if p.get('direction_correct', False)) / 10
            
            if accuracy_rate > 0.7:  # 高准确率
                new_entry_count = max(1, self.entry_confirm_count - 1)
                new_reverse_count = max(1, self.reverse_confirm_count - 1)
                self._log(f"  高准确率({accuracy_rate:.1%})，确认次数减少: {self.entry_confirm_count}->{new_entry_count}")
                self.entry_confirm_count = new_entry_count
                self.reverse_confirm_count = new_reverse_count
            elif accuracy_rate < 0.3:  # 低准确率
                new_entry_count = min(5, self.entry_confirm_count + 1)
                new_reverse_count = min(5, self.reverse_confirm_count + 1)
                self._log(f"  低准确率({accuracy_rate:.1%})，确认次数增加: {self.entry_confirm_count}->{new_entry_count}")
                self.entry_confirm_count = new_entry_count
                self.reverse_confirm_count = new_reverse_count
        
        self.adaptive_params.update({
            "threshold": self.threshold,
            "entry_confirm_count": self.entry_confirm_count,
            "reverse_confirm_count": self.reverse_confirm_count,
            "last_optimization": datetime.now()
        })
        
        # 性能反馈循环：记录优化历史
        if len(self.prediction_history) >= 5:
            recent_accuracy = sum(1 for p in self.prediction_history[-5:] if p.get('direction_correct', False)) / 5
            self._log(f"[性能反馈] 最近5次预测准确率: {recent_accuracy:.1%}")
            
            # 如果连续表现不佳，考虑策略切换
            if recent_accuracy < 0.2 and len(self.prediction_history) >= 20:
                last_20_accuracy = sum(1 for p in self.prediction_history[-20:] if p.get('direction_correct', False)) / 20
                if last_20_accuracy < 0.3:
                    self._log("⚠️ 警告: 预测准确率持续偏低，建议检查策略有效性")
        
        self._log("[自适应优化] 参数优化完成")

    def _determine_effective_strategy(self, df, signal):
        """根据市场状态和时间确定有效策略 - 需要2次连续确认"""
        if self.strategy_type not in ["auto", "time"]:
            return self.strategy_type

        if len(df) < 50:
            return "trend"

        if self.strategy_type == "auto":
            market_state = self.market_analyzer.analyze(df)
            state_name = market_state['state']
            self._log(f"[市场状态] {state_name}, 强度: {market_state['strength']:.2f}, 置信度: {market_state['confidence']:.2f}")
            self.time_analyzer.record_market_state(df, market_state)
            recommended = market_state["state"]
            
        elif self.strategy_type == "time":
            now = datetime.now()
            hour = now.hour
            time_slots = self.strategy_profile.get("time_slots", {})
            
            market_state = self.market_analyzer.analyze(df)
            self.time_analyzer.record_market_state(df, market_state)
            
            self._log(f"[时间策略] 当前时间: {now.strftime('%Y-%m-%d %H:%M')}, 小时: {hour}")
            
            recommended = "trend"
            for slot_name, slot_config in time_slots.items():
                if hour in slot_config.get("hours", []):
                    slot_desc = slot_config.get("description", slot_name)
                    use_range = slot_config.get("use_range_strategy", False)
                    
                    if use_range:
                        recommended = "range"
                    elif "消息" in slot_desc or "breakout" in slot_name.lower():
                        recommended = "breakout"
                    else:
                        recommended = "trend"
                    
                    self._log(f"[时间策略] 当前时段: {slot_desc}, 推荐策略: {recommended}")
                    break
        
        # 策略切换需要2次连续确认的逻辑
        if recommended == self.current_effective_strategy:
            # 推荐策略与当前策略一致，重置确认计数器
            self.pending_strategy_switch = None
            self.strategy_switch_confirm_count = 0
            return recommended
        else:
            # 推荐策略与当前策略不一致，检查确认计数
            if self.pending_strategy_switch != recommended:
                # 新的推荐策略，重置并开始计数
                self.pending_strategy_switch = recommended
                self.strategy_switch_confirm_count = 1
                self._log(f"[策略切换确认] 第1次确认: {self.current_effective_strategy} -> {recommended}")
                return self.current_effective_strategy
            else:
                # 同样的推荐策略，增加确认计数
                self.strategy_switch_confirm_count += 1
                if self.strategy_switch_confirm_count >= 2:
                    # 达到2次确认，执行切换
                    self._log(f"[策略切换确认] 第2次确认，执行切换: {self.current_effective_strategy} -> {recommended}")
                    self.pending_strategy_switch = None
                    self.strategy_switch_confirm_count = 0
                    return recommended
                else:
                    # 还未达到2次确认，继续等待
                    self._log(f"[策略切换确认] 第{self.strategy_switch_confirm_count}次确认: {self.current_effective_strategy} -> {recommended}")
                    return self.current_effective_strategy

    def check_trading_hours(self):
        now = datetime.now()
        hour = now.hour
        try:
            start = self.trade_frequency_config.get("active_hours_start", 0)
            end = self.trade_frequency_config.get("active_hours_end", 24)
        except (KeyError, AttributeError):
            # 如果配置有问题，默认全天交易
            return True

        if start <= end:
            in_active = start <= hour < end
        else:
            in_active = hour >= start or hour < end

        return in_active

    def check_extreme_move(self, df):
        if len(df) < 2:
            return False

        recent_change = abs(
            (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
        )
        try:
            threshold = self.risk_management_config.get("extreme_move_threshold", 0.02)
        except (KeyError, AttributeError):
            threshold = 0.02
        return recent_change >= threshold

    def calculate_kline_change(self, df):
        if len(df) < 2:
            return 0.0
        return (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1]

    def check_entry_conditions(self, signal, df, funding_rate):
        current_price = signal["current_price"]
        trend_direction = signal["trend_direction"]

        kline_change = self.calculate_kline_change(df)

        # 获取配置参数（带安全检查）
        support_buffer = self.entry_filter.get("support_buffer", 1.001)
        resistance_buffer = self.entry_filter.get("resistance_buffer", 0.999)
        max_kline_change = self.entry_filter.get("max_kline_change", 0.015)
        max_funding_rate_long = self.entry_filter.get("max_funding_rate_long", 0.03)
        min_funding_rate_short = self.entry_filter.get("min_funding_rate_short", -0.03)

        # 统一方向标识：BUY = LONG, SELL = SHORT
        if trend_direction in ["LONG", "BUY"]:
            if (
                current_price
                < signal["pred_support"] * support_buffer
            ):
                return False, "价格低于支撑位缓冲"
            if kline_change < -max_kline_change:
                return False, f"K线跌幅过大 {kline_change*100:.2f}%"
            if funding_rate > max_funding_rate_long:
                return False, f"资金费率过高 {funding_rate*100:.6f}%"
            return True, "做多条件满足"

        elif trend_direction in ["SHORT", "SELL"]:
            if (
                current_price
                > signal["pred_resistance"]
                * resistance_buffer
            ):
                return False, "价格高于阻力位缓冲"
            if kline_change > max_kline_change:
                return False, f"K线涨幅过大 {kline_change*100:.2f}%"
            if funding_rate < min_funding_rate_short:
                return False, f"资金费率过低 {funding_rate*100:.6f}%"
            return True, "做空条件满足"

        return False, f"无明确趋势方向: {trend_direction}"

    def calculate_position_size(self, entry_price, stop_loss_price):
        step_size, min_notional = self._get_symbol_filters()

        total_balance = self.get_total_balance()
        risk_config = self.strategy_profile.get("risk", {})
        basic_config = self.strategy_profile.get("basic", {})
        
        single_trade_risk = risk_config.get("single_trade_risk", 0.029)
        max_single_position = risk_config.get("max_single_position", 0.29)
        
        position_multiplier = basic_config.get("POSITION_MULTIPLIER", 1.0)
        
        risk_amount = (
            total_balance * single_trade_risk
        )
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk <= 0:
            size = (
                total_balance
                * max_single_position
                / entry_price
            )
        else:
            size = (risk_amount / price_risk) * self.leverage

        max_size = (
            total_balance
            * max_single_position
            * self.leverage
        ) / entry_price
        size = min(size, max_size)
        
        size = size * position_multiplier

        # 用户设置的最小仓位和交易所要求的最小值，取较大者
        effective_min_notional = max(min_notional, self.min_position)
        min_size = effective_min_notional / entry_price
        current_notional = size * entry_price

        if current_notional < effective_min_notional:
            size = min_size

        size = round(size / step_size) * step_size
        size = max(size, step_size)

        final_notional = size * entry_price
        if final_notional < effective_min_notional:
            additional_steps = (
                int((effective_min_notional - final_notional) / (step_size * entry_price)) + 1
            )
            size += additional_steps * step_size

        if self.risk_manager:
            df = self.binance.get_recent_klines(
                self.symbol, self.timeframe, lookback=50
            )
            if df is not None and len(df) >= 20:
                recent_volume = df["volume"].iloc[-20:].mean() * entry_price
                liquidity_ok, liquidity_msg = self.risk_manager.check_liquidity_risk(
                    recent_volume=recent_volume
                )
                if not liquidity_ok:
                    print(f"流动性警告: {liquidity_msg}")

        return size

    def _get_symbol_filters(self):
        """获取交易对的过滤器信息"""
        symbol_info = self.binance.get_symbol_info(self.symbol)
        step_size = 0.001
        min_notional = 100.0  # 默认最小名义金额 100 USDT

        if symbol_info:
            for filter in symbol_info["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    step_size = float(filter["stepSize"])
                elif filter["filterType"] == "MIN_NOTIONAL":
                    min_notional = float(filter.get("notional", 100))
                    if "minNotional" in filter:
                        min_notional = float(filter["minNotional"])

        return step_size, min_notional

    def open_position(self, signal):
        trend_direction = signal["trend_direction"]
        current_price = signal["current_price"]

        # 统一方向标识：BUY = LONG, SELL = SHORT
        if trend_direction in ["LONG", "BUY"]:
            side = Client.SIDE_BUY
        elif trend_direction in ["SHORT", "SELL"]:
            side = Client.SIDE_SELL
        else:
            print(f"❌ 无效的趋势方向: {trend_direction}")
            return False

        # ============================================
        # 优先使用Kronos AI自主计算的止盈止损
        # ============================================
        if "ai_stop_loss" in signal and "ai_take_profit_1" in signal:
            print("  [使用Kronos AI自主计算的止盈止损]")
            stop_loss = signal["ai_stop_loss"]
            tp1 = signal["ai_take_profit_1"]
            tp2 = signal.get("ai_take_profit_2", None)
            tp3 = None
            print(f"    止损: ${stop_loss:.2f} ({signal.get('sl_pct', 0)*100:.2f}%)")
            print(f"    止盈1: ${tp1:.2f} ({signal.get('tp1_pct', 0)*100:.2f}%)")
            if tp2:
                print(f"    止盈2: ${tp2:.2f} ({signal.get('tp2_pct', 0)*100:.2f}%)")
        else:
            # 备选方案：使用策略配置（新格式）
            print("  [备选：使用策略配置计算止盈止损]")
            
            # 从策略配置中获取参数
            stop_loss_config = self.strategy_profile.get("stop_loss", {})
            take_profit_config = self.strategy_profile.get("take_profit", {})
            
            # 止损配置 - 优先使用策略配置，如果没有则使用GUI配置
            long_buffer = stop_loss_config.get("long_buffer", self.stop_loss_config.get("long_buffer", 0.996))
            short_buffer = stop_loss_config.get("short_buffer", self.stop_loss_config.get("short_buffer", 1.004))
            
            # 止盈配置 - 优先使用策略配置，如果没有则使用GUI配置
            tp1_multiplier_long = take_profit_config.get("tp1_multiplier_long", self.take_profit_config.get("tp1_multiplier_long", 1.025))
            tp2_multiplier_long = take_profit_config.get("tp2_multiplier_long", self.take_profit_config.get("tp2_multiplier_long", 1.05))
            tp3_multiplier_long = take_profit_config.get("tp3_multiplier_long", self.take_profit_config.get("tp3_multiplier_long", 1.14))
            tp1_multiplier_short = take_profit_config.get("tp1_multiplier_short", self.take_profit_config.get("tp1_multiplier_short", 0.975))
            tp2_multiplier_short = take_profit_config.get("tp2_multiplier_short", self.take_profit_config.get("tp2_multiplier_short", 0.95))
            tp3_multiplier_short = take_profit_config.get("tp3_multiplier_short", self.take_profit_config.get("tp3_multiplier_short", 0.86))
            
            # 止盈仓位比例 - 优先使用策略配置
            tp1_position_ratio = take_profit_config.get("tp1_position_ratio", self.take_profit_config.get("tp1_position_ratio", 0.35))
            tp2_position_ratio = take_profit_config.get("tp2_position_ratio", self.take_profit_config.get("tp2_position_ratio", 0.35))
            tp3_position_ratio = take_profit_config.get("tp3_position_ratio", self.take_profit_config.get("tp3_position_ratio", 0.3))
            
            # 计算止损价 - 使用策略配置的缓冲
            if trend_direction in ["LONG", "BUY"]:
                stop_loss = signal["pred_support"] * long_buffer
            else:
                stop_loss = signal["pred_resistance"] * short_buffer
            
            # 计算止盈价 - 使用策略配置的乘数
            if trend_direction in ["LONG", "BUY"]:
                tp1 = current_price * tp1_multiplier_long
                tp2 = current_price * tp2_multiplier_long
                tp3 = current_price * tp3_multiplier_long
            else:
                tp1 = current_price * tp1_multiplier_short
                tp2 = current_price * tp2_multiplier_short
                tp3 = current_price * tp3_multiplier_short
            
            print(f"    止损: ${stop_loss:.2f} (缓冲: {long_buffer if trend_direction in ['LONG','BUY'] else short_buffer})")
            print(f"    止盈1: ${tp1:.2f}, 止盈2: ${tp2:.2f}, 止盈3: ${tp3:.2f}")

        position_size = self.calculate_position_size(current_price, stop_loss)

        # 风险管理器调整仓位大小
        if self.risk_manager:
            # 获取市场数据用于风险分析
            df = self.binance.get_recent_klines(
                self.symbol, self.timeframe, lookback=100
            )
            if df is not None and len(df) >= 50:
                # 应用风险调整
                adjusted_size = (
                    self.risk_manager.calculate_position_size_with_risk_adjustment(
                        position_size, df
                    )
                )
                adjustment_pct = (adjusted_size - position_size) / position_size * 100
                print(
                    f"风险调整仓位: {position_size:.6f} -> {adjusted_size:.6f} ({adjustment_pct:+.1f}%)"
                )
                position_size = adjusted_size

        # Alpha信号调整仓位大小
        is_alpha_signal = signal.get("is_alpha_signal", False)
        if is_alpha_signal:
            alpha_score = signal.get("alpha_score", 0)
            confidence_level = signal.get("confidence_level", "LOW")

            # 根据Alpha分数调整仓位
            if alpha_score >= 0.7 and confidence_level in ["HIGH", "MEDIUM"]:
                # 强Alpha信号增加仓位
                alpha_adjustment = 1.0 + (alpha_score - 0.7) * 0.5  # 最多增加15%
                position_size *= alpha_adjustment
                print(
                    f"强Alpha信号({alpha_score:.3f})，仓位增加{((alpha_adjustment-1)*100):.1f}%"
                )
            elif alpha_score <= 0.3 or confidence_level in ["VERY_LOW"]:
                # 弱Alpha信号减少仓位
                position_size *= 0.8  # 减少20%
                print(f"弱Alpha信号({alpha_score:.3f})，仓位减少20%")

        # 从策略配置获取仓位管理参数
        position_config = self.strategy_profile.get("position", {})
        initial_entry_ratio = position_config.get("initial_entry_ratio", 0.35)
        add_on_profit = position_config.get("add_on_profit", True)
        initial_size = position_size * initial_entry_ratio

        tp2_display = f"${tp2:.2f}" if tp2 is not None else "无"
        print_open(f"开仓: {trend_direction}")
        print(f"  入场价: ${current_price:.2f}")
        print(f"  止损: ${stop_loss:.2f}")
        print(f"  止盈1: ${tp1:.2f}, 止盈2: {tp2_display}")
        print(f"  初始仓位: {initial_size}")

        # 检查初始仓位是否满足最小名义金额要求（取交易所要求和用户设置的最大值）
        step_size, min_notional = self._get_symbol_filters()
        # 用户设置的最小仓位和交易所要求的最小值，取较大者
        effective_min_notional = max(min_notional, self.min_position)
        current_notional = initial_size * current_price
        if current_notional < effective_min_notional:
            print(
                f"⚠️ 初始仓位名义金额 ${current_notional:.2f} < ${effective_min_notional:.2f} (交易所:${min_notional:.2f}, 用户:${self.min_position:.2f}), 调整到最小要求"
            )
            min_size = effective_min_notional / current_price
            initial_size = max(min_size, initial_size)
            # 按照步长调整
            initial_size = round(initial_size / step_size) * step_size
            initial_size = max(initial_size, step_size)
            # 确保满足最小名义金额
            final_notional = initial_size * current_price
            if final_notional < effective_min_notional:
                additional_steps = (
                    int((effective_min_notional - final_notional) / (step_size * current_price))
                    + 1
                )
                initial_size += additional_steps * step_size
            print(
                f"⚠️ 调整后初始仓位: {initial_size}, 名义金额: ${initial_size * current_price:.2f}"
            )

        if trend_direction in ["LONG", "BUY"]:
            # 市场订单带重试
            market_order = None
            for retry in range(2):  # 重试一次
                market_order = self.binance.place_market_buy(self.symbol, initial_size)
                if market_order is not None:
                    break
                print(f"市场买入订单失败，重试 {retry+1}/2")
                time.sleep(1)  # 等待1秒后重试

            if market_order is None:
                print("❌ 市场买入订单失败，开仓中止")
                return False
            
            # 止损订单创建（带重试机制）
            stop_loss_order = None
            for retry in range(3):
                stop_loss_order = self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_SELL, initial_size, stop_loss
                )
                if stop_loss_order is not None:
                    break
                print(f"止损订单创建失败，重试 {retry+1}/3")
                time.sleep(1)
            
            if stop_loss_order is None:
                print("❌ 止损订单创建失败，取消开仓")
                self.binance.place_market_sell(self.symbol, initial_size)  # 立即平仓
                return False
            
            # 止盈订单创建（带重试机制）
            take_profit_order = None
            for retry in range(3):
                take_profit_order = self.binance.place_take_profit_order(
                    self.symbol, Client.SIDE_SELL, initial_size, tp1
                )
                if take_profit_order is not None:
                    break
                print(f"止盈订单创建失败，重试 {retry+1}/3")
                time.sleep(1)
            
            if take_profit_order is None:
                print("⚠️ 止盈订单创建失败，但止损订单已生效，继续持仓")
        elif trend_direction in ["SHORT", "SELL"]:
            # 市场订单带重试
            market_order = None
            for retry in range(2):  # 重试一次
                market_order = self.binance.place_market_sell(self.symbol, initial_size)
                if market_order is not None:
                    break
                print(f"市场卖出订单失败，重试 {retry+1}/2")
                time.sleep(1)  # 等待1秒后重试

            if market_order is None:
                print("❌ 市场卖出订单失败，开仓中止")
                return False
            
            # 止损订单创建（带重试机制）
            stop_loss_order = None
            for retry in range(3):
                stop_loss_order = self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_BUY, initial_size, stop_loss
                )
                if stop_loss_order is not None:
                    break
                print(f"止损订单创建失败，重试 {retry+1}/3")
                time.sleep(1)
            
            if stop_loss_order is None:
                print("❌ 止损订单创建失败，取消开仓")
                self.binance.place_market_buy(self.symbol, initial_size)  # 立即平仓
                return False
            
            # 止盈订单创建（带重试机制）
            take_profit_order = None
            for retry in range(3):
                take_profit_order = self.binance.place_take_profit_order(
                    self.symbol, Client.SIDE_BUY, initial_size, tp1
                )
                if take_profit_order is not None:
                    break
                print(f"止盈订单创建失败，重试 {retry+1}/3")
                time.sleep(1)
            
            if take_profit_order is None:
                print("⚠️ 止盈订单创建失败，但止损订单已生效，继续持仓")

        self.current_position = trend_direction
        self.position_entry_price = current_price
        self.position_size = initial_size
        self.initial_position_size = initial_size  # 保存初始仓位大小
        self.stop_loss_price = stop_loss
        self.take_profit_1_price = tp1
        self.take_profit_2_price = tp2
        self.take_profit_3_price = tp3
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.last_trade_time = self._get_current_time()
        self.daily_trades += 1
        
        # 开仓后计时逻辑
        self.post_entry_time = self._get_current_time()
        self.post_entry_entry_count = 0

        # 保存当前预测，用于后续验证准确性
        self.current_prediction = {
            "open_time": self._get_current_time(),
            "trend_direction": trend_direction,
            "open_price": current_price,
            "predicted_price": signal.get("predicted_price", current_price),
            "pred_support": signal.get("pred_support", current_price * 0.98),
            "pred_resistance": signal.get("pred_resistance", current_price * 1.02),
            "price_change_pct": signal.get("price_change_pct", 0),
            "trend_strength": signal.get("trend_strength", 0),
            "alpha_score": signal.get("alpha_score", 0),
            "analysis_method": signal.get("analysis_method", "unknown")
        }

        self.trade_history.append(
            {
                "time": datetime.now(),
                "action": f"OPEN_{trend_direction}",
                "price": current_price,
                "size": initial_size,
                "signal": {
                    "trend_strength": signal.get("trend_strength", 0),
                    "alpha_score": signal.get("alpha_score", 0),
                    "signal_quality": signal.get("signal_quality", 0),
                    "price_change_pct": signal.get("price_change_pct", 0),
                    "predicted_price": signal.get("predicted_price", current_price),
                    "pred_support": signal.get("pred_support", current_price * 0.98),
                    "pred_resistance": signal.get("pred_resistance", current_price * 1.02),
                    "analysis_method": signal.get("analysis_method", "unknown")
                },
                "stop_loss": stop_loss,
                "take_profit_1": tp1,
                "take_profit_2": tp2
            }
        )
        # 限制历史记录长度，避免内存无限增长
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        # 添加交易记录到风险管理器
        if self.risk_manager:
            self.risk_manager.add_trade_record(
                {
                    "action": f"OPEN_{trend_direction}",
                    "price": current_price,
                    "size": initial_size,
                    "stop_loss": stop_loss,
                    "take_profit_1": tp1,
                    "take_profit_2": tp2,
                    "position_size": position_size,
                    "initial_size": initial_size,
                    "trend_direction": trend_direction,
                }
            )
        
        # 更新最后交易时间
        self.last_trade_time = datetime.now()

        return True

    def close_position(self, reason=""):
        if not self.current_position:
            return False

        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt <= 0:
            self.current_position = None
            return False

        print_close(f"平仓: {self.current_position}, 原因: {reason}")

        self.binance.cancel_all_orders(self.symbol)
        self.binance.cancel_all_algo_orders(self.symbol)

        if self.current_position in ["LONG", "BUY"]:
            self.binance.place_market_sell(self.symbol, pos_amt)
        else:
            self.binance.place_market_buy(self.symbol, pos_amt)

        # 计算盈亏
        current_price = self.get_current_price()
        pnl_pct = 0.0
        if self.position_entry_price and current_price:
            if self.current_position in ["LONG", "BUY"]:
                pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price * 100
            else:  # SHORT/SELL
                pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price * 100
        
        # 验证并记录预测准确性
        prediction_accuracy = None
        if self.current_prediction:
            actual_price_change_pct = 0.0
            if self.current_position in ["LONG", "BUY"]:
                actual_price_change_pct = (current_price - self.current_prediction["open_price"]) / self.current_prediction["open_price"]
            else:  # SHORT/SELL
                actual_price_change_pct = (self.current_prediction["open_price"] - current_price) / self.current_prediction["open_price"]
            
            predicted_price_change_pct = self.current_prediction["price_change_pct"]
            
            # 计算预测准确性指标
            direction_correct = False
            if (predicted_price_change_pct > 0 and actual_price_change_pct > 0) or \
               (predicted_price_change_pct < 0 and actual_price_change_pct < 0):
                direction_correct = True
            
            price_error_pct = abs(predicted_price_change_pct - actual_price_change_pct) * 100
            
            prediction_accuracy = {
                "open_time": self.current_prediction["open_time"],
                "close_time": datetime.now(),
                "trend_direction": self.current_prediction["trend_direction"],
                "open_price": self.current_prediction["open_price"],
                "close_price": current_price,
                "predicted_price": self.current_prediction["predicted_price"],
                "predicted_change_pct": predicted_price_change_pct * 100,
                "actual_change_pct": actual_price_change_pct * 100,
                "direction_correct": direction_correct,
                "price_error_pct": price_error_pct,
                "trend_strength": self.current_prediction["trend_strength"],
                "alpha_score": self.current_prediction["alpha_score"],
                "analysis_method": self.current_prediction["analysis_method"],
                "pnl_pct": pnl_pct
            }
            
            self.prediction_history.append(prediction_accuracy)
            
            # 限制历史记录长度
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            # 打印预测准确性分析
            print(f"\n{'='*60}")
            print("预测准确性分析:")
            print(f"{'='*60}")
            print(f"  预测方向: {self.current_prediction['trend_direction']}")
            print(f"  开仓价: ${self.current_prediction['open_price']:.2f}")
            print(f"  平仓价: ${current_price:.2f}")
            print(f"  预测价: ${self.current_prediction['predicted_price']:.2f}")
            print(f"  预测变化: {predicted_price_change_pct*100:+.2f}%")
            print(f"  实际变化: {actual_price_change_pct*100:+.2f}%")
            print(f"  方向正确: {'✅ 是' if direction_correct else '❌ 否'}")
            print(f"  价格误差: {price_error_pct:.2f}%")
            print(f"  实际盈亏: {pnl_pct:+.2f}%")
            print(f"  分析方法: {self.current_prediction['analysis_method']}")
            print(f"{'='*60}\n")
        
        self.trade_history.append(
            {
                "time": datetime.now(),
                "action": f"CLOSE_{self.current_position}",
                "reason": reason,
                "price": current_price,
                "entry_price": self.position_entry_price,
                "pnl_pct": pnl_pct,
                "size": pos_amt,
                "prediction_accuracy": prediction_accuracy
            }
        )
        # 限制历史记录长度，避免内存无限增长
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        # 添加交易记录到风险管理器
        if self.risk_manager:
            self.risk_manager.add_trade_record(
                {
                    "action": f"CLOSE_{self.current_position}",
                    "reason": reason,
                    "price": self.get_current_price(),
                    "size": pos_amt,
                    "position_entry_price": self.position_entry_price,
                    "position_size": self.position_size,
                    "stop_loss_price": self.stop_loss_price,
                    "take_profit_1_price": self.take_profit_1_price,
                    "take_profit_2_price": self.take_profit_2_price,
                }
            )

        # 平仓后完全重置所有状态
        self._reset_full_state()

        return True

    def add_position(self, additional_ratio=0.5):
        if not self.current_position:
            return False, "无持仓"

        # 步骤1：获取加仓前的持仓信息
        pre_add_pos_type, pre_add_size = self.get_current_position_info()
        if pre_add_size <= 0:
            return False, "加仓前持仓为空"
        
        pre_add_entry_price = self.position_entry_price
        pre_add_size_internal = self.position_size
        
        current_price = self.get_current_price()
        if not current_price:
            return False, "获取价格失败"

        available_balance = self.get_total_balance()
        max_add_size = (
            available_balance * self.leverage * additional_ratio
        ) / current_price
        step_size = 0.001
        max_add_size = round(max_add_size / step_size) * step_size

        if max_add_size < 0.001:
            return False, "余额不足"

        print(f"加仓: {self.current_position}, 数量: {max_add_size}")
        print(f"  加仓前持仓: {pre_add_size}, 均价: ${pre_add_entry_price:.2f}")

        # 步骤2：清除所有止盈止损订单
        print("  清除所有止盈止损订单...")
        self.binance.cancel_all_orders(self.symbol)
        self.binance.cancel_all_algo_orders(self.symbol)

        # 步骤3：执行加仓
        if self.current_position in ["LONG", "BUY"]:
            add_order = self.binance.place_market_buy(self.symbol, max_add_size)
        else:
            add_order = self.binance.place_market_sell(self.symbol, max_add_size)
            
        if add_order is None:
            print("  ⚠️ 加仓订单执行失败")
            return False, "加仓订单执行失败"

        # 步骤4：等待交易所确认，获取实际加仓后的持仓
        time.sleep(0.5)
        pos_type, actual_total_size = self.get_current_position_info()
        if actual_total_size <= 0:
            return False, "加仓后持仓为空"
            
        actual_add_size = actual_total_size - pre_add_size
        print(f"  实际加仓数量: {actual_add_size}, 总持仓: {actual_total_size}")
        
        # 步骤5：计算新的持仓均价（基于实际加仓数量）
        avg_price = (
            pre_add_entry_price * pre_add_size
            + current_price * actual_add_size
        ) / actual_total_size
        
        # 步骤6：根据新持仓重新设置止盈止损
        print("  重新设置止盈止损...")
        
        # 使用策略配置的止损止盈规则
        exit_rules = self.strategy_profile["exit_rules"]
        stop_loss_rules = exit_rules["stop_loss"]
        take_profit_rules = exit_rules["take_profit"]
        
        # 获取安全的配置参数
        long_buffer = self.stop_loss_config.get("long_buffer", 0.996)
        short_buffer = self.stop_loss_config.get("short_buffer", 1.004)
        
        # 计算止损价
        stop_loss_type = stop_loss_rules["type"]
        if stop_loss_type in ["fixed", "tight"]:
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * (1 - stop_loss_rules["long_pct"])
            else:
                new_stop = avg_price * (1 + stop_loss_rules["short_pct"])
        elif stop_loss_type == "ai_predicted":
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * long_buffer
            else:
                new_stop = avg_price * short_buffer
        else:
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * long_buffer
            else:
                new_stop = avg_price * short_buffer
        
        # 计算止盈价
        take_profit_type = take_profit_rules["type"]
        if take_profit_type == "trailing":
            if self.current_position in ["LONG", "BUY"]:
                tp1 = avg_price * (1 + take_profit_rules.get("tp1_pct", 0.012))
                tp2 = avg_price * (1 + take_profit_rules.get("tp2_pct", 0.025))
            else:
                tp1 = avg_price * (1 - take_profit_rules.get("tp1_pct", 0.012))
                tp2 = avg_price * (1 - take_profit_rules.get("tp2_pct", 0.025))
        elif take_profit_type == "midpoint":
            target_range = take_profit_rules.get("target_pct_range", [0.008, 0.012])
            target_pct = (target_range[0] + target_range[1]) / 2
            if self.current_position in ["LONG", "BUY"]:
                tp1 = avg_price * (1 + target_pct)
                tp2 = None
            else:
                tp1 = avg_price * (1 - target_pct)
                tp2 = None
        elif take_profit_type == "fixed":
            target_pct = take_profit_rules.get("target_pct", 0.025)
            if self.current_position in ["LONG", "BUY"]:
                tp1 = avg_price * (1 + target_pct)
                tp2 = None
            else:
                tp1 = avg_price * (1 - target_pct)
                tp2 = None
        else:
            if self.current_position in ["LONG", "BUY"]:
                tp1 = avg_price * 1.012
                tp2 = avg_price * 1.025
            else:
                tp1 = avg_price * 0.988
                tp2 = avg_price * 0.975
        
        print(f"  新持仓: {actual_total_size}, 新均价: ${avg_price:.2f}")
        print(f"  新止损: ${new_stop:.2f}, 新止盈1: ${tp1:.2f}")
        
        # 更新内部变量（使用实际数据）
        self.position_entry_price = avg_price
        self.position_size = actual_total_size
        self.stop_loss_price = new_stop
        self.take_profit_1_price = tp1
        self.take_profit_2_price = tp2
        
        # 步骤7：重新设置止损订单
        from binance.client import Client
        stop_loss_side = Client.SIDE_SELL if self.current_position in ["LONG", "BUY"] else Client.SIDE_BUY
        
        stop_loss_order = None
        for retry in range(3):
            stop_loss_order = self.binance.place_stop_loss_order(
                self.symbol, stop_loss_side, actual_total_size, new_stop
            )
            if stop_loss_order is not None:
                break
            print(f"  止损订单创建失败，重试 {retry+1}/3")
            time.sleep(1)
        
        if stop_loss_order is None:
            print("  ⚠️ 止损订单创建失败")
        else:
            print("  ✓ 止损订单设置成功")
        
        # 步骤8：重新设置止盈订单
        take_profit_side = Client.SIDE_SELL if self.current_position in ["LONG", "BUY"] else Client.SIDE_BUY
        
        take_profit_order = None
        for retry in range(3):
            take_profit_order = self.binance.place_take_profit_order(
                self.symbol, take_profit_side, actual_total_size, tp1
            )
            if take_profit_order is not None:
                break
            print(f"  止盈订单创建失败，重试 {retry+1}/3")
            time.sleep(1)
        
        if take_profit_order is None:
            print("  ⚠️ 止盈订单创建失败")
        else:
            print("  ✓ 止盈订单设置成功")

        # 记录交易历史
        self.trade_history.append(
            {
                "time": datetime.now(),
                "action": f"ADD_{self.current_position}",
                "price": current_price,
                "size": actual_add_size,
                "new_avg_price": avg_price,
                "new_stop_loss": new_stop,
                "new_take_profit_1": tp1,
                "new_take_profit_2": tp2,
                "total_position": actual_total_size
            }
        )
        # 限制历史记录长度，避免内存无限增长
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        # 加仓后重置开仓后计时，相当于新开仓
        self.post_entry_time = datetime.now()
        self.post_entry_entry_count = 0
        self.last_trade_time = datetime.now()

        return True, f"加仓成功, 数量: {actual_add_size}, 新均价: ${avg_price:.2f}, 新止损: ${new_stop:.2f}"

    def reduce_position(self, reduce_ratio=0.5):
        if not self.current_position:
            return False, "无持仓"

        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt <= 0:
            return False, "无持仓"

        reduce_size = pos_amt * reduce_ratio
        reduce_size = round(reduce_size / 0.001) * 0.001

        current_price = self.get_current_price()

        print(f"减仓: {self.current_position}, 数量: {reduce_size}")

        if self.current_position in ["LONG", "BUY"]:
            self.binance.place_market_sell(self.symbol, reduce_size)
        else:
            self.binance.place_market_buy(self.symbol, reduce_size)

        self.position_size -= reduce_size

        if self.position_size <= 0.001:
            self.close_position("减仓至0")
            return True, "已全部平仓"

        self.trade_history.append(
            {
                "time": datetime.now(),
                "action": f"REDUCE_{self.current_position}",
                "price": current_price,
                "size": reduce_size,
            }
        )
        # 限制历史记录长度，避免内存无限增长
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        return True, f"减仓成功, 剩余: {self.position_size}"

    def get_current_position_info(self):
        pos = self.binance.get_position(self.symbol)
        if pos:
            amt = float(pos["positionAmt"])
            if amt > 0:
                return "LONG", amt
            elif amt < 0:
                return "SHORT", abs(amt)
        return None, 0

    def get_current_price(self):
        df = self.binance.get_recent_klines(self.symbol, self.timeframe, lookback=1)
        if df is not None and len(df) > 0:
            return df["close"].iloc[-1]
        return None

    def check_position_status(self, df=None, latest_signal=None):
        """
        检查持仓状态并执行保护机制
        新增：反向信号平仓、动态止损、趋势反转检测
        """
        if not self.current_position:
            return

        current_price = self.get_current_price()
        if not current_price:
            return

        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt <= 0:
            self.current_position = None
            return

        print(f"\n[持仓保护] 当前持仓: {self.current_position}, 价格: ${current_price:.2f}")

        # ============================================
        # 保护机制1：反向信号强制平仓
        # ============================================
        if latest_signal:
            signal_direction = latest_signal.get("trend_direction", "")
            signal_strength = latest_signal.get("trend_strength", 0)
            
            if (self.current_position == "LONG" and signal_direction in ["SHORT", "SELL"]) or \
               (self.current_position == "SHORT" and signal_direction in ["LONG", "BUY"]):
                
                if signal_strength >= self.threshold * 0.8:
                    print(f"  [反向信号平仓] 当前: {self.current_position}, 新信号: {signal_direction}, 强度: {signal_strength:.4f}")
                    self.close_position("Kronos反向信号平仓")
                    return

        # ============================================
        # 保护机制2：基于ATR的动态止损 + 移动止损
        # ============================================
        # 移动止损功能已暂时禁用
        # if df is not None and len(df) >= 20:
        #     # 计算ATR
        #     high = df["high"].values
        #     low = df["low"].values
        #     close = df["close"].values
        #     tr1 = high - low
        #     tr2 = np.abs(high - np.roll(close, 1))
        #     tr3 = np.abs(low - np.roll(close, 1))
        #     tr = np.maximum(tr1, np.maximum(tr2, tr3))
        #     tr[0] = 0
        #     atr = pd.Series(tr).rolling(window=14).mean().iloc[-1]
        #     atr_pct = atr / current_price
        #     
        #     print(f"  [动态止损] ATR: {atr:.2f} ({atr_pct*100:.2f}%)")
        #     
        #     # 动态止损 = 1.2 * ATR
        #     dynamic_sl_pct = max(1.2 * atr_pct, 0.006)
        #     
        #     if self.current_position == "LONG":
        #         new_dynamic_sl = current_price * (1 - dynamic_sl_pct)
        #         # 移动止损：只向上移动，不向下
        #         if self.stop_loss_price is None or new_dynamic_sl > self.stop_loss_price:
        #             if self.stop_loss_price is None:
        #                 print(f"  [移动止损] 设置初始止损: ${new_dynamic_sl:.2f}")
        #             else:
        #                 print(f"  [移动止损] 调整止损: ${self.stop_loss_price:.2f} -> ${new_dynamic_sl:.2f}")
        #             self.stop_loss_price = new_dynamic_sl
        #             # 更新交易所的止损订单
        #             self.binance.cancel_all_orders(self.symbol)
        #             self.binance.place_stop_loss_order(
        #                 self.symbol, Client.SIDE_SELL, pos_amt, self.stop_loss_price
        #             )
        #     else:  # SHORT
        #         new_dynamic_sl = current_price * (1 + dynamic_sl_pct)
        #         # 移动止损：只向下移动，不向上
        #         if self.stop_loss_price is None or new_dynamic_sl < self.stop_loss_price:
        #             if self.stop_loss_price is None:
        #                 print(f"  [移动止损] 设置初始止损: ${new_dynamic_sl:.2f}")
        #             else:
        #                 print(f"  [移动止损] 调整止损: ${self.stop_loss_price:.2f} -> ${new_dynamic_sl:.2f}")
        #             self.stop_loss_price = new_dynamic_sl
        #             # 更新交易所的止损订单
        #             self.binance.cancel_all_orders(self.symbol)
        #             self.binance.place_stop_loss_order(
        #                 self.symbol, Client.SIDE_BUY, pos_amt, self.stop_loss_price
        #             )

        # ============================================
        # 保护机制3：趋势反转检测（连续3次确认）
        # ============================================
        # MA均线反转检测已禁用，避免与Kronos信号反转检测冲突
        # if df is not None and len(df) >= 50:
        #     close_series = df["close"]
        #     ma5 = close_series.rolling(5).mean().iloc[-1]
        #     ma10 = close_series.rolling(10).mean().iloc[-1]
        #     ma20 = close_series.rolling(20).mean().iloc[-1]
        #     
        #     # 检测趋势反转信号
        #     reverse_signal = None
        #     if self.current_position == "LONG":
        #         # 多单：MA5下穿MA10 或 价格跌破MA20
        #         if ma5 < ma10 * 1.0005 and close_series.iloc[-1] < ma20:
        #             reverse_signal = "LONG"
        #     else:  # SHORT
        #         # 空单：MA5上穿MA10 或 价格突破MA20
        #         if ma5 > ma10 * 0.9995 and close_series.iloc[-1] > ma20:
        #             reverse_signal = "SHORT"
        #     
        #     # 处理反转信号
        #     if reverse_signal is not None:
        #         # 检查信号是否与上次一致
        #         if reverse_signal == self.last_reverse_signal:
        #             self.consecutive_reverse_count += 1
        #             print_reverse_signal(f"  [趋势反转检测] 检测到趋势反转信号 [{reverse_signal}] ({self.consecutive_reverse_count}/{self.reverse_confirm_count})")
        #             
        #             # 达到确认次数，执行平仓
        #             if self.consecutive_reverse_count >= self.reverse_confirm_count:
        #                 if self.current_position == "LONG":
        #                     print_highlight(f"  [趋势反转检测] 多单趋势反转确认: MA5(${ma5:.2f}) < MA10(${ma10:.2f}), 价格<MA20(${ma20:.2f})")
        #                     self.close_position("趋势反转-多单")
        #                 else:
        #                     print_highlight(f"  [趋势反转检测] 空单趋势反转确认: MA5(${ma5:.2f}) > MA10(${ma10:.2f}), 价格>MA20(${ma20:.2f})")
        #                     self.close_position("趋势反转-空单")
        #                 return
        #         else:
        #             # 新的反转信号，重置计数器
        #             self.consecutive_reverse_count = 1
        #             self.last_reverse_signal = reverse_signal
        #             print_reverse_signal(f"  [趋势反转检测] 检测到新趋势反转信号 [{reverse_signal}] (1/{self.reverse_confirm_count})")
        #     else:
        #         # 没有反转信号，重置计数器
        #         if self.consecutive_reverse_count > 0:
        #             print_info(f"  [趋势反转检测] 趋势反转信号中断，重置计数器")
        #         self.consecutive_reverse_count = 0
        #         self.last_reverse_signal = None

        # ============================================
        # 原有保护机制：固定止损止盈
        # ============================================
        # 获取安全的配置参数 - 优先使用策略配置
        take_profit_config = self.strategy_profile.get("take_profit", {})
        tp1_position_ratio = take_profit_config.get("tp1_position_ratio", self.take_profit_config.get("tp1_position_ratio", 0.35))
        tp2_position_ratio = take_profit_config.get("tp2_position_ratio", self.take_profit_config.get("tp2_position_ratio", 0.35))
        tp3_position_ratio = take_profit_config.get("tp3_position_ratio", self.take_profit_config.get("tp3_position_ratio", 0.30))
        if self.current_position == "LONG":
            if self.stop_loss_price and current_price <= self.stop_loss_price:
                self.close_position("止损")
                self.consecutive_losses += 1
                pos_type, pos_amt = self.get_current_position_info()
                if pos_amt <= 0:
                    self.current_position = None
                    self.position_size = 0
            elif (
                not self.tp1_hit
                and self.take_profit_1_price
                and current_price >= self.take_profit_1_price
            ):
                tp1_size = self.initial_position_size * tp1_position_ratio
                remaining_size = pos_amt - tp1_size
                print(f"触发止盈1，平仓 {tp1_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_sell(self.symbol, tp1_size)
                self.tp1_hit = True
                self.stop_loss_price = self.position_entry_price * 1.001
                self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_SELL, remaining_size, self.stop_loss_price
                )
                if self.take_profit_2_price is not None:
                    self.binance.place_take_profit_order(
                        self.symbol,
                        Client.SIDE_SELL,
                        remaining_size,
                        self.take_profit_2_price,
                    )
            elif (
                self.tp1_hit
                and not self.tp2_hit
                and self.take_profit_2_price is not None
                and current_price >= self.take_profit_2_price
            ):
                tp2_size = self.initial_position_size * tp2_position_ratio
                remaining_size = pos_amt - tp2_size
                print(f"触发止盈2，平仓 {tp2_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_sell(self.symbol, tp2_size)
                self.tp2_hit = True
                self.stop_loss_price = self.position_entry_price * 1.002
                self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_SELL, remaining_size, self.stop_loss_price
                )
                if self.take_profit_3_price is not None:
                    self.binance.place_take_profit_order(
                        self.symbol,
                        Client.SIDE_SELL,
                        remaining_size,
                        self.take_profit_3_price,
                    )
            elif (
                self.tp2_hit
                and self.take_profit_3_price is not None
                and current_price >= self.take_profit_3_price
            ):
                self.close_position("止盈3")
                self.consecutive_losses = 0

        elif self.current_position == "SHORT":
            if self.stop_loss_price and current_price >= self.stop_loss_price:
                self.close_position("止损")
                self.consecutive_losses += 1
                pos_type, pos_amt = self.get_current_position_info()
                if pos_amt <= 0:
                    self.current_position = None
                    self.position_size = 0
            elif (
                not self.tp1_hit
                and self.take_profit_1_price
                and current_price <= self.take_profit_1_price
            ):
                tp1_size = self.initial_position_size * tp1_position_ratio
                remaining_size = pos_amt - tp1_size
                print(f"触发止盈1，平仓 {tp1_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_buy(self.symbol, tp1_size)
                self.tp1_hit = True
                self.stop_loss_price = self.position_entry_price * 0.999
                self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_BUY, remaining_size, self.stop_loss_price
                )
                if self.take_profit_2_price is not None:
                    self.binance.place_take_profit_order(
                        self.symbol,
                        Client.SIDE_BUY,
                        remaining_size,
                        self.take_profit_2_price,
                    )
            elif (
                self.tp1_hit
                and not self.tp2_hit
                and self.take_profit_2_price is not None
                and current_price <= self.take_profit_2_price
            ):
                tp2_size = self.initial_position_size * tp2_position_ratio
                remaining_size = pos_amt - tp2_size
                print(f"触发止盈2，平仓 {tp2_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_buy(self.symbol, tp2_size)
                self.tp2_hit = True
                self.stop_loss_price = self.position_entry_price * 0.998
                self.binance.place_stop_loss_order(
                    self.symbol, Client.SIDE_BUY, remaining_size, self.stop_loss_price
                )
                if self.take_profit_3_price is not None:
                    self.binance.place_take_profit_order(
                        self.symbol,
                        Client.SIDE_BUY,
                        remaining_size,
                        self.take_profit_3_price,
                    )
            elif (
                self.tp2_hit
                and self.take_profit_3_price is not None
                and current_price <= self.take_profit_3_price
            ):
                self.close_position("止盈3")
                self.consecutive_losses = 0

    def run_once(self):
        import time

        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.strategy_type in ["auto", "time"]:
            strategy_display = (
                f"{self.strategy_profile['name']} (策略模式: {self.strategy_type})"
            )
        else:
            strategy_display = self.strategy_profile["name"]
        print(f"策略: {strategy_display}")
        print(f"{'='*80}")

        # 先检查持仓情况（不管风控限制，有持仓就要管理）
        df = self.binance.get_recent_klines(
            self.symbol, self.timeframe, lookback=self.lookback_period
        )
        if df is None or len(df) < min(50, self.lookback_period):
            print("获取K线数据失败")
            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return

        funding_rate = self.binance.get_funding_rate(self.symbol)
        current_price = df["close"].iloc[-1]

        print(f"当前价格: ${current_price:.2f}")
        print(f"上次结算资金费率: {funding_rate*100:.6f}%")

        # 检查真实持仓（从交易所获取）
        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt > 0:
            self.current_position = pos_type
            self.position_size = pos_amt
            print(f"当前持仓: {self.current_position}, 数量: {self.position_size}")
            
            # 标记当前有持仓
            self._last_had_position = True

            # AI分析当前持仓
            signal = self.analyzer.get_enhanced_signal(df, analysis_callback=self.analysis_callback)
            print(
                f"AI分析: 方向={signal['trend_direction']}, 强度={signal['trend_strength']:.4f}"
            )

            # 先计算当前盈亏（确保后面要用到）
            position_pnl_pct = 0
            if self.position_entry_price and current_price:
                if self.current_position == "LONG":
                    position_pnl_pct = (
                        (current_price - self.position_entry_price)
                        / self.position_entry_price
                        * 100
                    )
                else:
                    position_pnl_pct = (
                        (self.position_entry_price - current_price)
                        / self.position_entry_price
                        * 100
                    )

            # 运行FinGPT舆情分析和策略协调器
            if self.strategy_coordinator is not None:
                print("\n" + "="*60)
                print("[多智能体系统] 运行FinGPT舆情分析和信号过滤...")
                print("="*60)
                
                try:
                    coordinator_signal = self.strategy_coordinator.get_combined_signal(kronos_signal=signal)
                    
                    if coordinator_signal:
                        print(f"\n[FinGPT] 市场情绪分析:")
                        sentiment = coordinator_signal.get('sentiment', {})
                        print(f"  市场情绪: {sentiment.get('overall_sentiment', 'UNKNOWN')}")
                        print(f"  置信度: {sentiment.get('confidence', 0):.1%}")
                        print(f"  风险等级: {sentiment.get('risk_level', 'UNKNOWN')}")
                        
                        risk_factors = sentiment.get('risk_factors', [])
                        if risk_factors:
                            print(f"  风险因素: {', '.join(risk_factors[:3])}")
                        
                        print(f"\n[策略协调器] 信号过滤结果:")
                        print(f"  Kronos原始信号: {coordinator_signal.get('kronos_signal', 'UNKNOWN')}")
                        
                        final_signal = coordinator_signal.get('final_signal', 'NEUTRAL')
                        final_strength = coordinator_signal.get('signal_strength', 0)
                        
                        # 根据信号类型使用不同颜色显示
                        if final_signal in ['BUY', 'LONG']:
                            print_signal_buy(f"  过滤后信号: {final_signal}")
                        elif final_signal in ['SELL', 'SHORT']:
                            print_signal_sell(f"  过滤后信号: {final_signal}")
                        else:
                            print_signal_neutral(f"  过滤后信号: {final_signal}")
                        
                        print(f"  信号强度: {final_strength:.3f}")
                        print(f"  执行建议: {coordinator_signal.get('recommendation', 'UNKNOWN')}")
                        
                        if coordinator_signal.get('filtered', False):
                            print_warning(f"  ⚠️ 信号被过滤原因: {coordinator_signal.get('filter_reason', '未知')}")
                        
                        is_filtered = coordinator_signal.get('filtered', False)
                        
                        if is_filtered:
                            print_warning(f"\n[策略协调器] 信号被过滤，建议观望，不执行交易")
                            print("="*60 + "\n")
                            print(f"当前盈亏: {position_pnl_pct:.2f}%")
                            print_ai_decision(f"AI决策: 信号被过滤，持仓观望")
                            self.check_position_status(df=df, latest_signal=signal)
                            total_time = time.time() - start_time
                            print(f"*******总执行耗时: {total_time:.2f}秒*******")
                            return
                        elif final_signal in ['BUY', 'SELL', 'LONG', 'SHORT']:
                            if final_signal in ['BUY', 'LONG']:
                                print_highlight(f"\n[策略协调器] 采用综合信号: {final_signal} (强度: {final_strength:.3f})")
                            else:
                                print_highlight(f"\n[策略协调器] 采用综合信号: {final_signal} (强度: {final_strength:.3f})")
                            signal['trend_direction'] = final_signal
                            signal['trend_strength'] = final_strength
                        else:
                            print_info(f"\n[策略协调器] 建议观望，不执行交易")
                            print("="*60 + "\n")
                            print(f"当前盈亏: {position_pnl_pct:.2f}%")
                            print_ai_decision(f"AI决策: 观望")
                            self.check_position_status(df=df, latest_signal=signal)
                            total_time = time.time() - start_time
                            print(f"*******总执行耗时: {total_time:.2f}秒*******")
                            return
                    else:
                        print("[策略协调器] 未能生成综合信号，使用原始Kronos信号")
                        
                except Exception as e:
                    print(f"[多智能体系统] 分析失败: {e}")
                    print("[多智能体系统] 使用原始Kronos信号继续交易")
                
                print("="*60 + "\n")

            print(f"当前盈亏: {position_pnl_pct:.2f}%")

            # AI决策
            should_close = False
            should_add = False
            close_reason = ""

            # 1. 检查是否有强烈拐点（has_turning_point）或强烈信号
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            # 判断强烈信号条件：
            # - 有拐点检测 → 强烈信号
            # - 信号强度 > 阈值的1.5倍 → 强烈信号
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True

            # 2. 开仓后计时逻辑（新功能）
            if self.post_entry_time:
                hours_since_entry = (self._get_current_time() - self.post_entry_time).total_seconds() / 3600
                print(f"[开仓后计时] 已持仓 {hours_since_entry:.1f} 小时 (阈值: {self.post_entry_hours} 小时)")
                
                # 检查是否在a小时内
                if hours_since_entry < self.post_entry_hours:
                    # a小时内：检查同向信号
                    is_same_direction = (
                        (self.current_position == "LONG" and signal["trend_direction"] == "LONG") or
                        (self.current_position == "SHORT" and signal["trend_direction"] == "SHORT")
                    )
                    
                    if is_same_direction:
                        # 同向信号：增加计数
                        self.post_entry_entry_count += 1
                        print(f"  [加仓检查] 检测到同向开仓信号，连续第 {self.post_entry_entry_count}/2 次")
                        
                        # 连续2次同向信号 → 加仓
                        if self.post_entry_entry_count >= 2:
                            should_add = True
                            print_highlight(f"  ✅ 连续2次同向信号，准备加仓")
                    else:
                        # 非同向信号：重置计数
                        self.post_entry_entry_count = 0
                else:
                    # 超过a小时：检查盈利是否大于0.6%，如果是就平仓
                    if position_pnl_pct >= self.take_profit_min_pct:
                        should_close = True
                        close_reason = f"超过{self.post_entry_hours}小时且盈利{position_pnl_pct:.2f}% > {self.take_profit_min_pct}%"
                        print_highlight(f"  ✅ 超过{self.post_entry_hours}小时且盈利达标，执行止盈平仓")

            # 3. 反向信号检测和确认逻辑（强趋势和普通趋势共用一个计数器）
            if not should_close and self.current_position == "LONG" and signal["trend_direction"] == "SHORT":
                if "short" == self.last_reverse_signal:
                    self.consecutive_reverse_count += 1
                    if has_strong_signal:
                        if has_turning_point:
                            print_reverse_signal(f"  ⚡ 检测到强烈拐点做空信号（强度: {signal_strength:.4f}），连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                        else:
                            print_reverse_signal(f"  ⚡ 检测到强做空信号（强度: {signal_strength:.4f}），连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                    else:
                        print_warning(f"  检测到正常做空信号，连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                    
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        if has_strong_signal:
                            if has_turning_point:
                                close_reason = "强烈拐点检测做空（连续确认）"
                            else:
                                close_reason = "强趋势反转做空（连续确认）"
                        else:
                            close_reason = "趋势反转做空（连续确认）"
                        print_highlight(f"  ✅ 确认{self.reverse_confirm_count}次，执行平仓")
                else:
                    # 新的信号，重置计数器
                    self.consecutive_reverse_count = 1
                    self.last_reverse_signal = "short"
                    if has_strong_signal:
                        if has_turning_point:
                            print_reverse_signal(f"  ⚡ 检测到新强烈拐点做空信号（强度: {signal_strength:.4f}），连续第1/{self.reverse_confirm_count}次确认")
                        else:
                            print_reverse_signal(f"  ⚡ 检测到新强做空信号（强度: {signal_strength:.4f}），连续第1/{self.reverse_confirm_count}次确认")
                    else:
                        print_warning(f"  检测到新正常做空信号，连续第1/{self.reverse_confirm_count}次确认")
            elif not should_close and (
                self.current_position == "SHORT" and signal["trend_direction"] == "LONG"
            ):
                if "long" == self.last_reverse_signal:
                    self.consecutive_reverse_count += 1
                    if has_strong_signal:
                        if has_turning_point:
                            print_reverse_signal(f"  ⚡ 检测到强烈拐点做多信号（强度: {signal_strength:.4f}），连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                        else:
                            print_reverse_signal(f"  ⚡ 检测到强做多信号（强度: {signal_strength:.4f}），连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                    else:
                        print_warning(f"  检测到正常做多信号，连续第{self.consecutive_reverse_count}/{self.reverse_confirm_count}次确认")
                    
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        if has_strong_signal:
                            if has_turning_point:
                                close_reason = "强烈拐点检测做多（连续确认）"
                            else:
                                close_reason = "强趋势反转做多（连续确认）"
                        else:
                            close_reason = "趋势反转做多（连续确认）"
                        print_highlight(f"  ✅ 确认{self.reverse_confirm_count}次，执行平仓")
                else:
                    # 新的信号，重置计数器
                    self.consecutive_reverse_count = 1
                    self.last_reverse_signal = "long"
                    if has_strong_signal:
                        if has_turning_point:
                            print_reverse_signal(f"  ⚡ 检测到新强烈拐点做多信号（强度: {signal_strength:.4f}），连续第1/{self.reverse_confirm_count}次确认")
                        else:
                            print_reverse_signal(f"  ⚡ 检测到新强做多信号（强度: {signal_strength:.4f}），连续第1/{self.reverse_confirm_count}次确认")
                    else:
                        print_warning(f"  检测到新正常做多信号，连续第1/{self.reverse_confirm_count}次确认")
            else:
                # 趋势未反转，但保留计数器，避免频繁重置
                pass

            # 4. 亏损过大强制平仓
            if not should_close and position_pnl_pct < -3:
                should_close = True
                close_reason = f"亏损过大 ({position_pnl_pct:.2f}%)"
                print(f"  ⚡ 亏损过大，强制平仓")

            # 5. 原有加仓逻辑保留（与新加仓逻辑共存）
            if not should_close and not should_add and (
                position_pnl_pct > 1
                and signal["trend_strength"] > self.threshold
            ):
                if (
                    self.current_position == "LONG"
                    and signal["trend_direction"] == "LONG"
                ) or (
                    self.current_position == "SHORT"
                    and signal["trend_direction"] == "SHORT"
                ):
                    should_add = True

            if should_close:
                print_ai_decision(f"AI决策: 平仓 - {close_reason}")
                self.close_position(close_reason)
            elif should_add:
                print_ai_decision(f"AI决策: 加仓 - 盈利{position_pnl_pct:.2f}%且趋势有利")
                self.add_position()
            else:
                print(f"AI决策: 持仓观望")
                self.check_position_status(df=df, latest_signal=signal)

            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return
        else:
            print(f"未检测到持仓 (返回值: pos_type={pos_type}, pos_amt={pos_amt})")
            
            # 检测是否从有持仓变为无持仓（包括交易所自动平仓）
            if self._last_had_position:
                print("[状态重置] 检测到从有持仓变为无持仓，完全重置所有状态")
                self._reset_full_state()
            else:
                # 一直无持仓，只重置持仓状态，保留开仓确认计数器
                self._reset_position_only()
            
            # 更新持仓状态标志
            self._last_had_position = False

        # 无持仓，检查风控后才能开仓
        risk_ok, risk_msg = self.check_risk_limits()
        self.last_risk_msg = risk_msg  # 保存风控消息供后面使用
        if not risk_ok:
            print(f"风控限制: {risk_msg}")
            # 即使有风控限制，也继续进行市场分析
            can_open_position = False
        else:
            can_open_position = True

        in_active_hours = self.check_trading_hours()
        if not in_active_hours:
            print("非活跃交易时段，信号过滤强度提升")

        # 风险管理器的市场波动性检查
        if self.risk_manager and len(df) >= 100:
            volatility_ok, volatility_msg = (
                self.risk_manager.check_market_volatility_risk(df)
            )
            if not volatility_ok:
                print(f"市场波动性风险: {volatility_msg}")
                # 如果波动性过高，可以考虑暂停交易或调整策略
                # 这里我们记录警告但不停止交易
                print("警告: 高波动性市场，谨慎交易")
            else:
                print(f"市场波动性检查: {volatility_msg}")

        # 自适应参数优化
        self._adaptive_parameter_optimization(df)

        if self.check_extreme_move(df):
            print("检测到极端行情，暂停开仓")
            can_open_position = False

        # 清理多余的止盈止损单
        try:
            self.binance.cancel_all_orders(self.symbol)
            self.binance.cancel_all_algo_orders(self.symbol)
            print("已清理多余的止盈止损单")
        except Exception as e:
            print(f"清理止盈止损单时出错: {e}")

        print("正在分析市场...")
        step3_start = time.time()
        signal = self.analyzer.get_enhanced_signal(df, analysis_callback=self.analysis_callback)
        step3_time = time.time() - step3_start
        print(f"Kronos分析耗时: {step3_time:.2f}秒")

        print(f"趋势方向: {signal['trend_direction']}")
        print(f"趋势强度: {signal['trend_strength']:.4f}")
        print(f"预测价格变化: {signal['price_change_pct']*100:.2f}%")
        print(
            f"预测支撑: ${signal['pred_support']:.2f}, 预测阻力: ${signal['pred_resistance']:.2f}"
        )

        # 运行FinGPT舆情分析和策略协调器
        if self.strategy_coordinator is not None:
            print("\n" + "="*60)
            print("[多智能体系统] 运行FinGPT舆情分析和信号过滤...")
            print("="*60)
            
            try:
                # 获取策略协调器的综合信号（传递已计算的Kronos信号，避免循环调用）
                coordinator_signal = self.strategy_coordinator.get_combined_signal(kronos_signal=signal)
                
                if coordinator_signal:
                    # 显示FinGPT分析结果
                    print(f"\n[FinGPT] 市场情绪分析:")
                    sentiment = coordinator_signal.get('sentiment', {})
                    print(f"  市场情绪: {sentiment.get('overall_sentiment', 'UNKNOWN')}")
                    print(f"  置信度: {sentiment.get('confidence', 0):.1%}")
                    print(f"  风险等级: {sentiment.get('risk_level', 'UNKNOWN')}")
                    
                    # 显示风险因素
                    risk_factors = sentiment.get('risk_factors', [])
                    if risk_factors:
                        print(f"  风险因素: {', '.join(risk_factors[:3])}")
                    
                    # 显示策略协调器决策
                    print(f"\n[策略协调器] 信号过滤结果:")
                    print(f"  Kronos原始信号: {coordinator_signal.get('kronos_signal', 'UNKNOWN')}")
                    print(f"  过滤后信号: {coordinator_signal.get('final_signal', 'UNKNOWN')}")
                    print(f"  信号强度: {coordinator_signal.get('signal_strength', 0):.3f}")
                    print(f"  执行建议: {coordinator_signal.get('recommendation', 'UNKNOWN')}")
                    
                    # 如果被过滤，显示原因
                    if coordinator_signal.get('filtered', False):
                        print(f"  ⚠️ 信号被过滤原因: {coordinator_signal.get('filter_reason', '未知')}")
                    
                    # 完全采用策略协调器的决策
                    final_signal = coordinator_signal.get('final_signal', 'NEUTRAL')
                    final_strength = coordinator_signal.get('signal_strength', 0)
                    is_filtered = coordinator_signal.get('filtered', False)
                    
                    if is_filtered:
                        print(f"\n[策略协调器] 信号被过滤，建议观望，不执行交易")
                        signal['signal_valid'] = False
                    elif final_signal in ['BUY', 'SELL', 'LONG', 'SHORT']:
                        print(f"\n[策略协调器] 采用综合信号: {final_signal} (强度: {final_strength:.3f})")
                        signal['trend_direction'] = final_signal
                        signal['trend_strength'] = final_strength
                        signal['signal_valid'] = True
                    else:
                        print(f"\n[策略协调器] 建议观望，不执行交易")
                        signal['signal_valid'] = False
                else:
                    print("[策略协调器] 未能生成综合信号，使用原始Kronos信号")
                    
            except Exception as e:
                print(f"[多智能体系统] 分析失败: {e}")
                print("[多智能体系统] 使用原始Kronos信号继续交易")
            
            print("="*60 + "\n")

        # 显示当前状态信息
        print(f"当前状态: 无持仓, can_open_position={can_open_position}, signal_valid={signal.get('signal_valid', True)}")

        # 如果不能开仓，分析完市场后就结束
        if not can_open_position:
            print_ai_decision(f"AI决策: 无法开仓 - {getattr(self, 'last_risk_msg', '风控限制')}")
            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return

        # 如果信号已经被策略协调器过滤，直接跳过
        if not signal.get("signal_valid", True):
            print_ai_decision("AI决策: 信号被策略协调器过滤，观望")
            # 重置开仓确认计数器
            self.consecutive_entry_count = 0
            self.last_entry_signal = None
            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return

        # 自动/时间策略切换
        effective_strategy = self._determine_effective_strategy(df, signal)
        if effective_strategy != self.current_effective_strategy:
            print_highlight(
                f"[策略切换] {self.current_effective_strategy} -> {effective_strategy}"
            )
            self.current_effective_strategy = effective_strategy
            self.strategy_profile = StrategyProfiles.get_profile(effective_strategy)
            
            # 重新加载完整策略配置
            self._load_strategy_config()
            
            self.strategy_switch_count += 1

        effective_strategy = self.current_effective_strategy

        # 只有在信号仍然有效时才进行入场检查
        if signal.get("signal_valid", True):
            if effective_strategy == "trend":
                signal["signal_valid"] = self._check_trend_entry(
                    signal, df, current_price, funding_rate
                )
            elif effective_strategy == "range":
                signal["signal_valid"] = self._check_range_entry(
                    signal, df, current_price, funding_rate
                )
            elif effective_strategy == "breakout":
                signal["signal_valid"] = self._check_breakout_entry(
                    signal, df, current_price, funding_rate
                )

        if not signal["signal_valid"]:
            print_ai_decision("AI决策: 信号无效，观望")
            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return

        entry_ok, entry_msg = self.check_entry_conditions(signal, df, funding_rate)
        if not entry_ok:
            print_ai_decision(f"AI决策: 入场条件不满足 - {entry_msg}")
            total_time = time.time() - start_time
            print(f"*******总执行耗时: {total_time:.2f}秒*******")
            return

        print_success(f"入场条件满足: {entry_msg}")

        # 1. 检查是否有强烈拐点或强烈信号
        has_strong_signal = False
        has_turning_point = signal.get("has_turning_point", False)
        signal_strength = signal.get("trend_strength", 0)
        
        # 判断强烈信号条件：
        # - 有拐点检测 → 强烈信号
        # - 信号强度 > 阈值的1.5倍 → 强烈信号
        if has_turning_point or signal_strength > self.threshold * 1.5:
            has_strong_signal = True

        # 2. 开仓连续确认机制（所有信号都遵循确认次数配置）
        current_entry_signal = signal["trend_direction"]

        if current_entry_signal == self.last_entry_signal:
            # 信号一致 → 增加计数
            self.consecutive_entry_count += 1
            if has_strong_signal:
                if has_turning_point:
                    print(f"  ⚡ 检测到强烈拐点{current_entry_signal}信号（强度: {signal_strength:.4f}，连续第{self.consecutive_entry_count}/{self.entry_confirm_count}次确认")
                else:
                    print(f"  ⚡ 检测到强{current_entry_signal}信号（强度: {signal_strength:.4f}，连续第{self.consecutive_entry_count}/{self.entry_confirm_count}次确认")
            else:
                print(
                    f"  检测到{current_entry_signal}信号，连续第{self.consecutive_entry_count}/{self.entry_confirm_count}次确认"
                )

            if self.consecutive_entry_count >= self.entry_confirm_count:
                print(f"  ✅ 连续{self.entry_confirm_count}次确认，执行开仓")
                self.open_position(signal)
                # 重置计数器
                self.consecutive_entry_count = 0
                self.last_entry_signal = None
            else:
                print(
                    f"  需要连续{self.entry_confirm_count}次确认，当前{self.consecutive_entry_count}次"
                )
        else:
            # 信号变化，重置计数器
            self.consecutive_entry_count = 1
            self.last_entry_signal = current_entry_signal
            if has_strong_signal:
                if has_turning_point:
                    print(f"  ⚡ 首次检测到强烈拐点{current_entry_signal}信号（强度: {signal_strength:.4f}，开始计数")
                else:
                    print(f"  ⚡ 首次检测到强{current_entry_signal}信号（强度: {signal_strength:.4f}，开始计数")
            else:
                print(f"  首次检测到{current_entry_signal}信号，开始计数")
        
        # 显示最终AI决策
        if self.consecutive_entry_count >= self.entry_confirm_count:
            print_ai_decision(f"AI决策: 开仓（{current_entry_signal}）")
        else:
            print_ai_decision(f"AI决策: 等待确认（{self.consecutive_entry_count}/{self.entry_confirm_count}次）")

        total_time = time.time() - start_time
        print(f"*******总执行耗时: {total_time:.2f}秒*******")

    def _check_trend_entry(self, signal, df, current_price, funding_rate):
        ai_filter = self.strategy_profile.get("ai_filter", {})
        entry_config = self.strategy_profile.get("entry", {})
        special_filters = self.strategy_profile.get("ai_filter", {})

        effective_threshold = self.threshold

        # Alpha信号增强检查
        is_alpha_signal = signal.get("is_alpha_signal", False)
        alpha_score = signal.get("alpha_score", 0)
        confidence_level = signal.get("confidence_level", "LOW")
        signal_category = signal.get("signal_category", "UNKNOWN")
        market_state = signal.get("market_state", "neutral")

        if is_alpha_signal:
            print(
                f"Alpha信号检测: score={alpha_score:.3f}, confidence={confidence_level}, category={signal_category}, market={market_state}"
            )

            # 基于Alpha分数的动态阈值调整
            if alpha_score >= 0.7:
                effective_threshold *= 0.8  # 强Alpha信号降低阈值要求
                print(
                    f"强Alpha信号({alpha_score:.3f})，阈值调整为{effective_threshold:.4f}"
                )
            elif alpha_score <= 0.3:
                effective_threshold *= 1.5  # 弱Alpha信号提高阈值要求
                print(
                    f"弱Alpha信号({alpha_score:.3f})，阈值调整为{effective_threshold:.4f}"
                )

            # 置信度过滤（降低严格程度：只过滤VERY_LOW，允许LOW置信度）
            if confidence_level == "VERY_LOW":
                print(f"置信度过低({confidence_level})，跳过信号")
                return False

            # LOW置信度允许交易，但降低要求
            if confidence_level == "LOW":
                print(f"低置信度({confidence_level})，降低交易要求")
                # 降低趋势强度要求
                effective_threshold *= 0.9
                print(f"  阈值调整为 {effective_threshold:.4f}")

        print("=" * 60)
        print("入场条件检查:")
        print("-" * 60)

        # 1. 趋势强度检查（使用策略配置的最小趋势强度）
        trend_min_threshold = ai_filter.get("min_trend_strength", self.ai_min_trend)
        trend_ok = signal["trend_strength"] >= trend_min_threshold
        print(
            f"  [1] 趋势强度: {signal['trend_strength']:.4f} {'✓' if trend_ok else '✗'} 最小要求 {trend_min_threshold}"
        )
        if not trend_ok:
            print("=" * 60)
            print(f"信号跳过 - 趋势强度不满足")
            print("=" * 60)
            return False

        # 2. 强方向要求检查
        require_strong_direction = ai_filter.get("require_strong_direction", False)
        if require_strong_direction:
            strong_direction_ok = abs(signal["price_change_pct"]) >= ai_filter.get("min_price_deviation", self.ai_min_deviation) * 1.5
            print(
                f"  [2] 强方向要求: {'✓' if strong_direction_ok else '✗'} (需要: {require_strong_direction})"
            )
            if not strong_direction_ok:
                print("=" * 60)
                print(f"信号跳过 - 强方向要求不满足")
                print("=" * 60)
                return False

        # 3. 预测偏离度检查
        price_deviation_pct = ai_filter.get("min_price_deviation", self.ai_min_deviation)
        deviation_ok = abs(signal["price_change_pct"]) >= price_deviation_pct
        print(
            f"  [3] 预测偏离度: {abs(signal['price_change_pct'])*100:.2f}% {'✓' if deviation_ok else '✗'} {price_deviation_pct*100}%"
        )
        if not deviation_ok:
            print("=" * 60)
            print(f"信号跳过 - 预测偏离度不满足")
            print("=" * 60)
            return False

        # 4. 资金费率检查（使用策略配置的参数，如果没有则使用GUI设置）
        max_funding = entry_config.get("max_funding_rate_long", self.max_funding)
        min_funding = entry_config.get("min_funding_rate_short", self.min_funding)
        funding_ok = min_funding <= funding_rate <= max_funding
        print(
            f"  [4] 资金费率: {funding_rate*100:.6f}% {'✓' if funding_ok else '✗'} [{min_funding*100:.2f}%, {max_funding*100:.2f}%]"
        )
        if not funding_ok:
            reason = "过高" if funding_rate > max_funding else "过低"
            print("=" * 60)
            print(f"信号跳过 - 资金费率{reason}")
            print("=" * 60)
            return False

        # 5. 市场状态过滤（震荡市禁止）
        if market_state == "ranging" and special_filters.get("forbid_in_range", False):
            print(f"  [5] 市场状态: {market_state} {'✗' if market_state == 'ranging' else '✓'} (震荡市禁止: {special_filters.get('forbid_in_range', False)})")
            print("=" * 60)
            print(f"信号跳过 - 震荡市场，禁止趋势交易")
            print("=" * 60)
            return False
        print(f"  [5] 市场状态: {market_state} ✓")

        print("-" * 60)
        print("  ✓ 所有入场条件检查通过")
        print("=" * 60)

        # 趋势爆发策略：只要趋势方向明确就开仓，不需要等突破
        print(f"触发趋势{signal['trend_direction']}: 趋势方向明确")
        print(f"  当前价格: ${current_price:.2f}")
        print(f"  预测方向: {signal['trend_direction']}")
        print(f"  预测变化: {signal['price_change_pct']*100:.2f}%")
        print(f"  预测支撑: ${signal['pred_support']:.2f}")
        print(f"  预测阻力: ${signal['pred_resistance']:.2f}")

        return True

    def _check_range_entry(self, signal, df, current_price, funding_rate):
        ai_filter = self.strategy_profile.get("ai_filter", {})
        entry_config = self.strategy_profile.get("entry", {})

        effective_threshold = self.threshold

        print("=" * 60)
        print("震荡套利入场条件检查:")
        print("-" * 60)

        # 1. 震荡市判断（趋势强度低）
        trend_max_threshold = ai_filter.get("min_trend_strength", 0.005) * 0.8  # 低于趋势策略阈值80%
        is_range = signal["trend_strength"] <= trend_max_threshold
        print(
            f"  [1] 震荡市判断: {signal['trend_strength']:.4f} {'✓' if is_range else '✗'} < {trend_max_threshold:.4f}"
        )

        # 2. 价格位置检查 - 判断是否在极端位置
        support_buffer = entry_config.get("support_buffer", 1.002)
        resistance_buffer = entry_config.get("resistance_buffer", 0.998)
        
        mid_price = (signal["pred_support"] + signal["pred_resistance"]) / 2
        position_in_range = (current_price - signal["pred_support"]) / (signal["pred_resistance"] - signal["pred_support"])
        price_at_extreme_pct = 0.15  # 认为在区间15%位置算极端
        is_at_extreme = position_in_range <= price_at_extreme_pct or position_in_range >= (1 - price_at_extreme_pct)
        extreme_side = "支撑" if position_in_range <= price_at_extreme_pct else "阻力"

        print(f"  [2] 价格位置检查:")
        print(f"      支撑位: ${signal['pred_support']:.2f}")
        print(f"      中间位: ${mid_price:.2f}")
        print(f"      当前价: ${current_price:.2f}")
        print(f"      阻力位: ${signal['pred_resistance']:.2f}")
        print(f"      价格在区间位置: {position_in_range*100:.1f}%")
        print(f"      极端位置阈值: {price_at_extreme_pct*100:.0f}%")
        print(f"      是否在{extreme_side}附近: {'✓' if is_at_extreme else '✗'}")
        
        if not is_at_extreme and signal["trend_direction"] in ["LONG", "SHORT"]:
            print("=" * 60)
            print(f"信号跳过 - 价格不在{extreme_side}附近")
            print("=" * 60)
            return False

        # 震荡套利：只要趋势强度低（震荡市）且价格在极端位置，就根据预测方向开仓
        print(f"  ✓ 震荡市交易机会")
        print(f"  触发震荡{signal['trend_direction']}: 震荡市根据方向开仓")
        print("=" * 60)
        return True

    def _check_breakout_entry(self, signal, df, current_price, funding_rate):
        ai_filter = self.strategy_profile.get("ai_filter", {})
        entry_config = self.strategy_profile.get("entry", {})
        
        effective_threshold = self.threshold

        print("=" * 60)
        print("消息突破入场条件检查:")
        print("-" * 60)

        # 1. 趋势强度检查（使用策略配置的最小趋势强度）
        trend_min_threshold = ai_filter.get("min_trend_strength", self.ai_min_trend)
        trend_ok = signal["trend_strength"] >= trend_min_threshold
        print(
            f"  [1] 趋势强度: {signal['trend_strength']:.4f} {'✓' if trend_ok else '✗'} 最小要求 {trend_min_threshold}"
        )
        if not trend_ok:
            print("=" * 60)
            print(f"信号跳过 - 趋势强度不足")
            print("=" * 60)
            return False

        # 2. 连续预测确认检查
        require_consecutive = ai_filter.get("require_consecutive_prediction", self.require_consecutive_prediction)
        if require_consecutive > 1:
            consecutive_ok = True
            print(
                f"  [2] 连续预测确认: {'✓' if consecutive_ok else '✗'} (需要: {require_consecutive}次)"
            )
            if not consecutive_ok:
                print("=" * 60)
                print(f"信号跳过 - 连续预测确认不足")
                print("=" * 60)
                return False

        # 3. 预测偏离度检查
        price_deviation_pct = ai_filter.get("min_price_deviation", self.ai_min_deviation)
        deviation_ok = abs(signal["price_change_pct"]) >= price_deviation_pct
        print(
            f"  [3] 预测偏离度: {abs(signal['price_change_pct'])*100:.2f}% {'✓' if deviation_ok else '✗'} {price_deviation_pct*100}%"
        )
        if not deviation_ok:
            print("=" * 60)
            print(f"信号跳过 - 预测偏离度不满足")
            print("=" * 60)
            return False

        # 4. 资金费率检查（使用策略配置的参数，如果没有则使用GUI设置）
        max_funding = entry_config.get("max_funding_rate_long", self.max_funding)
        min_funding = entry_config.get("min_funding_rate_short", self.min_funding)
        funding_ok = min_funding <= funding_rate <= max_funding
        print(
            f"  [4] 资金费率: {funding_rate*100:.6f}% {'✓' if funding_ok else '✗'} [{min_funding*100:.2f}%, {max_funding*100:.2f}%]"
        )
        if not funding_ok:
            reason = "过高" if funding_rate > max_funding else "过低"
            print("=" * 60)
            print(f"信号跳过 - 资金费率{reason}")
            print("=" * 60)
            return False

        # 5. FinGPT信号检查（如果策略要求）
        require_fingpt = ai_filter.get("require_fingpt_signal", False)
        if require_fingpt:
            # 检查是否有FinGPT信号
            has_fingpt_signal = signal.get("has_fingpt_signal", False)
            fingpt_sentiment = signal.get("fingpt_sentiment", "UNKNOWN")
            
            if fingpt_sentiment != "UNKNOWN":
                print(f"  [5] FinGPT信号: {'✓' if has_fingpt_signal else '✗'} (情绪: {fingpt_sentiment})")
            
            if not has_fingpt_signal:
                print("=" * 60)
                print(f"信号跳过 - 等待FinGPT舆情信号")
                print("=" * 60)
                return False

        # 6. 显示交易信息
        print(f"  [6] 交易信息:")
        print(f"      支撑位: ${signal['pred_support']:.2f}")
        print(f"      当前价: ${current_price:.2f}")
        print(f"      阻力位: ${signal['pred_resistance']:.2f}")
        print(f"      方向: {signal['trend_direction']}")

        print("-" * 60)
        print("  ✓ 所有入场条件检查通过")
        print("=" * 60)

        # 消息突破策略：只要趋势强度够就直接开仓
        print(f"  ✓ 触发消息{signal['trend_direction']}: 趋势强度足够")
        print("=" * 60)
        return True

    def run_loop(self, interval_seconds=300, stop_event=None):
        print(f"启动{self.strategy_profile['name']}...")
        print(f"交易对: {self.symbol}")
        print(f"时间周期: {self.timeframe}")
        print(f"杠杆: {self.leverage}x")
        print(f"检查间隔: {interval_seconds}秒")
        print(f"{'='*80}")

        while True:
            # 检查是否停止（优先检查）
            if stop_event and stop_event.is_set():
                print(f"\n交易机器人已停止 (stop_event已设置)")
                break

            try:
                start_time = time.time()
                self.run_once()
                execution_time = time.time() - start_time
                if execution_time > 10:  # 如果执行时间超过10秒，记录警告
                    print(f"执行耗时较长: {execution_time:.2f}秒")
                self.consecutive_errors = 0  # 重置连续错误计数
            except KeyboardInterrupt:
                print("\n交易机器人已停止 (KeyboardInterrupt)")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                import traceback

                traceback.print_exc()

                # 更新错误统计
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error_time = datetime.now()

                # 根据连续错误数量调整等待时间
                if self.consecutive_errors >= 5:
                    wait_time = 300  # 5次连续错误后等待5分钟
                    print(f"连续错误 {self.consecutive_errors} 次，暂停交易5分钟")
                elif self.consecutive_errors >= 3:
                    wait_time = 120  # 3次连续错误后等待2分钟
                    print(f"连续错误 {self.consecutive_errors} 次，暂停交易2分钟")
                else:
                    wait_time = 60  # 默认等待60秒

                # 分段等待，以便能响应stop_event
                elapsed = 0
                while elapsed < wait_time:
                    if stop_event and stop_event.is_set():
                        print(f"\n交易机器人已停止 (等待期间收到停止信号)")
                        return
                    sleep_chunk = min(1, wait_time - elapsed)
                    time.sleep(sleep_chunk)
                    elapsed += sleep_chunk
                continue

            # 分段等待，以便能响应stop_event
            elapsed = 0
            while elapsed < interval_seconds:
                if stop_event and stop_event.is_set():
                    print(f"\n交易机器人已停止 (等待期间收到停止信号)")
                    return
                sleep_chunk = min(1, interval_seconds - elapsed)
                time.sleep(sleep_chunk)
                elapsed += sleep_chunk



    def _save_backtest_state(self):
        """保存回测前的策略状态"""
        return {
            'current_position': self.current_position,
            'position_entry_price': self.position_entry_price,
            'position_size': self.position_size,
            'consecutive_reverse_count': self.consecutive_reverse_count,
            'last_reverse_signal': self.last_reverse_signal,
            'consecutive_entry_count': self.consecutive_entry_count,
            'last_entry_signal': self.last_entry_signal,
            'post_entry_time': self.post_entry_time,
            'post_entry_entry_count': self.post_entry_entry_count,
            '_last_had_position': self._last_had_position,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'current_balance': self.current_balance,
            'starting_balance': self.starting_balance,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
            'trade_history': self.trade_history.copy() if hasattr(self, 'trade_history') else [],
            'last_trade_time': self.last_trade_time,
            'current_effective_strategy': self.current_effective_strategy,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_1_price': self.take_profit_1_price,
            'take_profit_2_price': self.take_profit_2_price,
            'take_profit_3_price': self.take_profit_3_price,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'tp3_hit': self.tp3_hit,
            # 策略切换确认计数器
            'pending_strategy_switch': self.pending_strategy_switch,
            'strategy_switch_confirm_count': self.strategy_switch_confirm_count,
        }

    def _restore_backtest_state(self, state):
        """恢复回测前的策略状态"""
        self.current_position = state['current_position']
        self.position_entry_price = state['position_entry_price']
        self.position_size = state['position_size']
        self.consecutive_reverse_count = state['consecutive_reverse_count']
        self.last_reverse_signal = state['last_reverse_signal']
        self.consecutive_entry_count = state['consecutive_entry_count']
        self.last_entry_signal = state['last_entry_signal']
        self.post_entry_time = state['post_entry_time']
        self.post_entry_entry_count = state['post_entry_entry_count']
        self._last_had_position = state['_last_had_position']
        self.daily_pnl = state['daily_pnl']
        self.daily_trades = state['daily_trades']
        self.consecutive_losses = state['consecutive_losses']
        self.current_balance = state['current_balance']
        self.starting_balance = state.get('starting_balance', self.current_balance)
        self.peak_balance = state['peak_balance']
        self.max_drawdown = state['max_drawdown']
        self.trade_history = state['trade_history']
        self.last_trade_time = state.get('last_trade_time', None)
        self.current_effective_strategy = state.get('current_effective_strategy', 'auto')
        self.stop_loss_price = state.get('stop_loss_price', None)
        self.take_profit_1_price = state.get('take_profit_1_price', None)
        self.take_profit_2_price = state.get('take_profit_2_price', None)
        self.take_profit_3_price = state.get('take_profit_3_price', None)
        self.tp1_hit = state.get('tp1_hit', False)
        self.tp2_hit = state.get('tp2_hit', False)
        self.tp3_hit = state.get('tp3_hit', False)
        # 恢复策略切换确认计数器
        self.pending_strategy_switch = state.get('pending_strategy_switch', None)
        self.strategy_switch_confirm_count = state.get('strategy_switch_confirm_count', 0)

    def _reset_backtest_state(self, initial_capital):
        """重置回测状态"""
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        self._last_had_position = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.current_balance = initial_capital
        self.starting_balance = initial_capital
        self.peak_balance = initial_capital
        self.max_drawdown = 0.0
        self.trade_history = []
        self.last_trade_time = None
        self.current_effective_strategy = "auto"
        # 重置止盈止损相关变量
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_3_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        # 重置策略切换确认计数器
        self.pending_strategy_switch = None
        self.strategy_switch_confirm_count = 0
        # 初始化风险管理器（回测模式专用）
        from enhanced_risk_manager import EnhancedRiskManager
        self.risk_manager = EnhancedRiskManager(initial_capital, self.symbol)
        self._sync_risk_manager_config()

    def run_backtest(self, df_historical, initial_capital=10000, fee_rate=0.001, slippage=0.0005, progress_callback=None, log_callback=None, stop_event=None):
        """
        运行完整回测 - 完全和真实交易一致
        
        Args:
            df_historical: 历史K线数据 DataFrame
            initial_capital: 初始资金
            fee_rate: 手续费率
            slippage: 滑点
            progress_callback: 进度回调函数
            log_callback: 日志回调函数
            stop_event: 停止事件对象（threading.Event）
            
        Returns:
            dict: 回测结果
        """
        from datetime import datetime, timedelta
        
        original_state = self._save_backtest_state()
        
        try:
            self._reset_backtest_state(initial_capital)
            
            backtest_trades = []
            equity_curve = [initial_capital]
            total_fees = 0
            
            lookback = 100
            total_candles = len(df_historical)
            
            if log_callback:
                log_callback(f"[回测] 开始回测，共 {total_candles - lookback} 根K线")
            
            for i in range(lookback, total_candles):
                if stop_event and stop_event.is_set():
                    if log_callback:
                        log_callback(f"[回测] 用户停止回测")
                    break
                
                if progress_callback:
                    progress = int((i - lookback) / (total_candles - lookback) * 100)
                    progress_callback(progress)
                
                df_slice = df_historical.iloc[:i+1].copy()
                current_price = df_slice['close'].iloc[-1]
                candle_time = df_slice['timestamps'].iloc[-1] if 'timestamps' in df_slice.columns else datetime.now()
                
                try:
                    action, action_reason, trade_info, log_messages = self._backtest_run_once(
                        df_slice, current_price, candle_time, fee_rate, slippage
                    )
                    
                    if log_callback and log_messages:
                        for msg in log_messages:
                            log_callback(msg)
                    
                    if action and trade_info:
                        backtest_trades.append(trade_info)
                        if trade_info.get('fee'):
                            total_fees += trade_info['fee']
                    
                    equity_curve.append(self.current_balance)
                    
                except Exception as e:
                    if log_callback:
                        import traceback
                        log_callback(f"[回测错误] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - {str(e)}")
                        log_callback(traceback.format_exc())
                    continue
            
            results = self._calculate_backtest_metrics(
                backtest_trades, equity_curve, initial_capital, total_fees
            )
            
            return results
            
        finally:
            self._restore_backtest_state(original_state)

    def _backtest_run_once(self, df, current_price, candle_time, fee_rate, slippage):
        """回测的单次运行 - 完全和真实交易的run_once一致"""
        from datetime import datetime, timedelta
        
        action = False
        action_reason = ""
        trade_info = {}
        log_messages = []
        
        def log_msg(msg):
            log_messages.append(msg)
        
        funding_rate = 0.0
        
        if self.current_position:
            self._last_had_position = True
            
            position_pnl_pct = 0
            if self.position_entry_price and current_price:
                if self.current_position == "LONG":
                    position_pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price * 100
                else:
                    position_pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price * 100
            
            should_close = False
            should_add = False
            close_reason = ""
            
            # 1. 检查固定止损
            if self.current_position == "LONG":
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            else:
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            
            # 2. 检查固定止盈1、止盈2和止盈3（分级止盈，部分平仓）
            if not should_close:
                # 从take_profit_config获取止盈仓位比例（与实盘一致）
                tp1_position_ratio = self.take_profit_config.get("tp1_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                tp2_position_ratio = self.take_profit_config.get("tp2_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                
                if self.current_position == "LONG":
                    if not self.tp1_hit and self.take_profit_1_price and current_price >= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price >= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price >= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
                else:
                    if not self.tp1_hit and self.take_profit_1_price and current_price <= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price <= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price <= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
            
            try:
                signal = self.analyzer.get_enhanced_signal(df)
            except:
                return action, action_reason, trade_info, log_messages
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            if self.post_entry_time:
                if hasattr(candle_time, 'timestamp'):
                    hours_since_entry = (candle_time - self.post_entry_time).total_seconds() / 3600
                else:
                    hours_since_entry = 0
                
                if hours_since_entry < self.post_entry_hours:
                    is_same_direction = (
                        (self.current_position == "LONG" and signal["trend_direction"] == "LONG") or
                        (self.current_position == "SHORT" and signal["trend_direction"] == "SHORT")
                    )
                    
                    if is_same_direction:
                        self.post_entry_entry_count += 1
                        if self.post_entry_entry_count >= 2:
                            should_add = True
                    else:
                        self.post_entry_entry_count = 0
                else:
                    if position_pnl_pct >= self.take_profit_min_pct:
                        should_close = True
                        close_reason = f"超过{self.post_entry_hours}小时且盈利{position_pnl_pct:.2f}% > {self.take_profit_min_pct}%"
            
            if not should_close and self.current_position == "LONG" and signal["trend_direction"] == "SHORT":
                if "short" == self.last_reverse_signal:
                    self.consecutive_reverse_count += 1
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        if has_strong_signal:
                            if has_turning_point:
                                close_reason = "强烈拐点检测做空（连续确认）"
                            else:
                                close_reason = "强趋势反转做空（连续确认）"
                        else:
                            close_reason = "趋势反转做空（连续确认）"
                else:
                    self.consecutive_reverse_count = 1
                    self.last_reverse_signal = "short"
            elif not should_close and self.current_position == "SHORT" and signal["trend_direction"] == "LONG":
                if "long" == self.last_reverse_signal:
                    self.consecutive_reverse_count += 1
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        if has_strong_signal:
                            if has_turning_point:
                                close_reason = "强烈拐点检测做多（连续确认）"
                            else:
                                close_reason = "强趋势反转做多（连续确认）"
                        else:
                            close_reason = "趋势反转做多（连续确认）"
                else:
                    self.consecutive_reverse_count = 1
                    self.last_reverse_signal = "long"
            
            if not should_close and position_pnl_pct < -3:
                should_close = True
                close_reason = f"亏损过大 ({position_pnl_pct:.2f}%)"
            
            if not should_close and not should_add and (
                position_pnl_pct > 1
                and signal["trend_strength"] > self.threshold
            ):
                if (
                    self.current_position == "LONG"
                    and signal["trend_direction"] == "LONG"
                ) or (
                    self.current_position == "SHORT"
                    and signal["trend_direction"] == "SHORT"
                ):
                    should_add = True
            
            if should_close:
                action = True
                action_reason = close_reason
                log_msg(f"[平仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - {close_reason}")
                trade_info = self._simulate_close_position(current_price, candle_time, close_reason, fee_rate, slippage)
                if trade_info:
                    log_msg(f"[平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}")
            
        else:
            if self._last_had_position:
                self._reset_full_state()
            else:
                self._reset_position_only()
            
            self._last_had_position = False
            
            risk_ok, risk_msg = self.check_risk_limits()
            self.last_risk_msg = risk_msg
            if not risk_ok:
                can_open_position = False
            else:
                can_open_position = True
            
            if not can_open_position:
                return action, action_reason, trade_info, log_messages
            
            in_active_hours = True
            
            if self.risk_manager and len(df) >= 100:
                try:
                    volatility_ok, volatility_msg = self.risk_manager.check_market_volatility_risk(df)
                except:
                    volatility_ok = True
            
            self._adaptive_parameter_optimization(df)
            
            extreme_ok = True
            if len(df) >= 2:
                prev_close = df['close'].iloc[-2]
                change_pct = (current_price - prev_close) / prev_close * 100
                if abs(change_pct) > 5:
                    extreme_ok = False
                    can_open_position = False
            
            if not can_open_position:
                return action, action_reason, trade_info, log_messages
            
            try:
                signal = self.analyzer.get_enhanced_signal(df)
            except:
                return action, action_reason, trade_info, log_messages
            
            signal_valid = signal.get("signal_valid", True)
            
            if not signal_valid:
                self.consecutive_entry_count = 0
                self.last_entry_signal = None
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_direction"] not in ["LONG", "SHORT"]:
                return action, action_reason, trade_info, log_messages
            
            effective_strategy = self._determine_effective_strategy(df, signal)
            if effective_strategy != self.current_effective_strategy:
                self.current_effective_strategy = effective_strategy
                self.strategy_profile = StrategyProfiles.get_profile(effective_strategy)
                
                # 重新加载完整策略配置
                self._load_strategy_config()
            
            effective_strategy = self.current_effective_strategy
            
            if signal.get("signal_valid", True):
                if effective_strategy == "trend":
                    try:
                        signal["signal_valid"] = self._check_trend_entry(signal, df, current_price, funding_rate)
                    except:
                        signal["signal_valid"] = True
                elif effective_strategy == "range":
                    try:
                        signal["signal_valid"] = self._check_range_entry(signal, df, current_price, funding_rate)
                    except:
                        signal["signal_valid"] = True
                elif effective_strategy == "breakout":
                    try:
                        signal["signal_valid"] = self._check_breakout_entry(signal, df, current_price, funding_rate)
                    except:
                        signal["signal_valid"] = True
            
            if not signal.get("signal_valid", True):
                return action, action_reason, trade_info, log_messages
            
            entry_ok = True
            try:
                entry_ok, entry_msg = self.check_entry_conditions(signal, df, funding_rate)
            except:
                pass
            
            if not entry_ok:
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_strength"] < self.threshold:
                return action, action_reason, trade_info, log_messages
            
            if self.last_trade_time:
                if hasattr(candle_time, 'timestamp') and hasattr(self.last_trade_time, 'timestamp'):
                    minutes_since_trade = (candle_time - self.last_trade_time).total_seconds() / 60
                    if minutes_since_trade < 10:
                        return action, action_reason, trade_info, log_messages
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            current_entry_signal = signal["trend_direction"]
            
            if current_entry_signal == self.last_entry_signal:
                self.consecutive_entry_count += 1
                
                if self.consecutive_entry_count >= self.entry_confirm_count:
                    action = True
                    action_reason = "开仓信号确认"
                    log_msg(f"[开仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 方向: {signal['trend_direction']}, 趋势强度: {signal_strength:.4f}")
                    trade_info = self._simulate_open_position(
                        signal, current_price, candle_time, fee_rate, slippage
                    )
                    if trade_info:
                        log_msg(f"[开仓完成] 价格: {trade_info['price']:.2f}, 数量: {trade_info['size']:.4f}, 手续费: {trade_info['fee']:.2f}")
                    self.last_trade_time = candle_time
                    self.consecutive_entry_count = 0
                    self.last_entry_signal = None
            else:
                self.consecutive_entry_count = 1
                self.last_entry_signal = current_entry_signal
        
        return action, action_reason, trade_info, log_messages

    def _simulate_open_position(self, signal, price, time, fee_rate, slippage):
        """模拟开仓"""
        direction = signal["trend_direction"]
        
        slippage_price = price * (1 + slippage) if direction == "LONG" else price * (1 - slippage)
        
        position_size = (self.current_balance * 0.1) / slippage_price
        position_size = max(position_size, 0.001)
        
        fee = slippage_price * position_size * fee_rate
        
        self.current_position = direction
        self.position_entry_price = slippage_price
        self.position_size = position_size
        self.initial_position_size = position_size
        self.post_entry_time = time
        self.post_entry_entry_count = 0
        
        # 设置止盈止损价格（优先使用 AI 推荐值）
        if "ai_stop_loss" in signal and "ai_take_profit_1" in signal:
            self.stop_loss_price = signal["ai_stop_loss"]
            self.take_profit_1_price = signal["ai_take_profit_1"]
            self.take_profit_2_price = signal.get("ai_take_profit_2", None)
            self.take_profit_3_price = signal.get("ai_take_profit_3", None)
        else:
            # 备选方案：使用默认值
            if direction == "LONG":
                self.stop_loss_price = slippage_price * 0.985  # 止损1.5%
                self.take_profit_1_price = slippage_price * 1.012  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 1.025  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 1.04  # 止盈3 4.0%
            else:
                self.stop_loss_price = slippage_price * 1.015  # 止损1.5%
                self.take_profit_1_price = slippage_price * 0.988  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 0.975  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 0.96  # 止盈3 4.0%
        
        self.tp1_hit = False
        self.tp2_hit = False
        
        trade_info = {
            "type": "OPEN",
            "direction": direction,
            "price": slippage_price,
            "size": position_size,
            "fee": fee,
            "time": time,
            "reason": "开仓"
        }
        
        self.trade_history.append(trade_info)
        self.current_balance -= fee
        
        return trade_info

    def _simulate_partial_close_position(self, price, time, reason, close_size, fee_rate, slippage):
        """模拟部分平仓"""
        direction = self.current_position
        
        slippage_price = price * (1 - slippage) if direction == "LONG" else price * (1 + slippage)
        
        if direction == "LONG":
            pnl = (slippage_price - self.position_entry_price) * close_size
        else:
            pnl = (self.position_entry_price - slippage_price) * close_size
        
        fee = slippage_price * close_size * fee_rate
        
        self.current_balance += pnl - fee
        self.position_size -= close_size
        
        trade_info = {
            "type": "PARTIAL_CLOSE",
            "direction": direction,
            "entry_price": self.position_entry_price,
            "exit_price": slippage_price,
            "size": close_size,
            "remaining_size": self.position_size,
            "pnl": pnl,
            "pnl_pct": (pnl / (self.position_entry_price * close_size)) * 100 if self.position_entry_price else 0,
            "fee": fee,
            "time": time,
            "reason": reason
        }
        
        self.trade_history.append(trade_info)
        
        if self.peak_balance < self.current_balance:
            self.peak_balance = self.current_balance
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        return trade_info
    
    def _simulate_close_position(self, price, time, reason, fee_rate, slippage):
        """模拟平仓"""
        direction = self.current_position
        
        slippage_price = price * (1 - slippage) if direction == "LONG" else price * (1 + slippage)
        
        if direction == "LONG":
            pnl = (slippage_price - self.position_entry_price) * self.position_size
        else:
            pnl = (self.position_entry_price - slippage_price) * self.position_size
        
        fee = slippage_price * self.position_size * fee_rate
        
        self.current_balance += pnl - fee
        
        trade_info = {
            "type": "CLOSE",
            "direction": direction,
            "entry_price": self.position_entry_price,
            "exit_price": slippage_price,
            "size": self.position_size,
            "pnl": pnl,
            "pnl_pct": (pnl / (self.position_entry_price * self.position_size)) * 100 if self.position_entry_price else 0,
            "fee": fee,
            "time": time,
            "reason": reason
        }
        
        self.trade_history.append(trade_info)
        
        if self.peak_balance < self.current_balance:
            self.peak_balance = self.current_balance
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.initial_position_size = 0
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        
        return trade_info

    def _calculate_backtest_metrics(self, trades, equity_curve, initial_capital, total_fees):
        """计算回测指标"""
        import numpy as np
        
        closed_trades = [t for t in trades if t['type'] == 'CLOSE']
        
        total_trades = len(closed_trades)
        win_trades = len([t for t in closed_trades if t['pnl'] > 0])
        loss_trades = len([t for t in closed_trades if t['pnl'] <= 0])
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = total_pnl / initial_capital if initial_capital > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if win_trades > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in closed_trades if t['pnl'] <= 0]) if loss_trades > 0 else 0
        profit_factor = (avg_win * win_trades) / (avg_loss * loss_trades) if (avg_loss > 0 and loss_trades > 0) else float('inf')
        
        max_drawdown = self.max_drawdown
        
        avg_profit_pct = np.mean([t['pnl_pct'] for t in closed_trades]) if total_trades > 0 else 0
        
        return {
            "trades": trades,
            "closed_trades": closed_trades,
            "equity_curve": equity_curve,
            "initial_capital": initial_capital,
            "final_capital": self.current_balance,
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "avg_profit": avg_profit_pct,
            "total_fees": total_fees
        }

class EnhancedRiskManager:
    """增强型风险管理器 - 提供多层次的风险控制"""

    def __init__(self, initial_balance, symbol="BTCUSDT"):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.symbol = symbol
        self.trade_history = []
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.risk_level = "NORMAL"  # NORMAL, CAUTIOUS, AGGRESSIVE
        self.market_volatility = "NORMAL"

        # 策略配置参数（从面板配置中加载）
        self.single_trade_risk = 0.029  # 单笔交易风险
        self.daily_loss_limit = 0.12  # 单日亏损限制
        self.max_consecutive_losses = 6  # 最大连续亏损
        self.max_single_position = 0.29  # 最大单笔仓位
        self.max_daily_position = 0.85  # 最大日仓位

        # 风险参数配置
        self.risk_params = {
            "max_daily_drawdown": 0.05,  # 单日最大回撤5%
            "max_overall_drawdown": 0.15,  # 总体最大回撤15%
            "volatility_adjustment_factor": 0.5,  # 波动性调整系数
            "correlation_threshold": 0.7,  # 相关性阈值
            "liquidity_min_volume": 1000000,  # 最低流动性要求（USDT）
            "black_swan_detection_threshold": 0.10,  # 黑天鹅事件检测阈值（10%）
            "position_concentration_limit": 0.25,  # 头寸集中度限制
        }

    def update_balance(self, new_balance):
        """更新账户余额并计算回撤"""
        self.current_balance = new_balance

        # 更新峰值余额
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # 计算最大回撤
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - new_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def add_trade_record(self, trade_info):
        """添加交易记录"""
        self.trade_history.append({**trade_info, "timestamp": datetime.now()})
        # 限制历史记录长度，避免内存无限增长
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

    def check_market_volatility_risk(self, df, lookback=100):
        """检查市场波动性风险"""
        if df is None or len(df) < lookback:
            return True, "数据不足，跳过波动性检查"

        # 计算历史波动率
        returns = df["close"].pct_change().dropna()[-lookback:]
        volatility = returns.std() * np.sqrt(252 * 288)  # 年化波动率（5分钟数据）

        # 波动率分级
        if volatility > 0.8:  # 年化波动率>80%
            self.market_volatility = "HIGH"
            return False, f"市场波动性过高: {volatility:.1%}"
        elif volatility < 0.2:  # 年化波动率<20%
            self.market_volatility = "LOW"
            return True, f"市场波动性较低: {volatility:.1%}"
        else:
            self.market_volatility = "NORMAL"
            return True, f"市场波动性正常: {volatility:.1%}"

    def check_drawdown_limits(self):
        """检查回撤限制"""
        # 计算当前回撤
        current_drawdown = 0
        if self.peak_balance > 0:
            current_drawdown = (
                self.peak_balance - self.current_balance
            ) / self.peak_balance

        # 检查单日回撤限制
        if current_drawdown >= self.risk_params["max_daily_drawdown"]:
            return (
                False,
                f"触及单日回撤限制: {current_drawdown:.1%} >= {self.risk_params['max_daily_drawdown']:.1%}",
            )

        # 检查总体回撤限制
        if self.max_drawdown >= self.risk_params["max_overall_drawdown"]:
            return (
                False,
                f"触及总体回撤限制: {self.max_drawdown:.1%} >= {self.risk_params['max_overall_drawdown']:.1%}",
            )

        return (
            True,
            f"回撤检查通过: 当前{current_drawdown:.1%}, 最大{self.max_drawdown:.1%}",
        )

    def check_liquidity_risk(self, order_book=None, recent_volume=None):
        """检查流动性风险"""
        # 这里可以扩展为实际检查订单簿深度
        # 目前使用简化的流动性检查

        if (
            recent_volume is not None
            and recent_volume < self.risk_params["liquidity_min_volume"]
        ):
            return (
                False,
                f"流动性不足: 近期成交量${recent_volume:,.0f} < ${self.risk_params['liquidity_min_volume']:,.0f}",
            )

        return True, "流动性检查通过"

    def check_black_swan_event(self, df, threshold=None):
        """检查黑天鹅事件（极端行情）"""
        if df is None or len(df) < 20:
            return False, "数据不足"

        threshold = threshold or self.risk_params["black_swan_detection_threshold"]

        # 检查最近几根K线的极端变化
        recent_changes = []
        for i in range(1, min(6, len(df))):
            change = abs(
                (df["close"].iloc[-i] - df["close"].iloc[-i - 1])
                / df["close"].iloc[-i - 1]
            )
            recent_changes.append(change)

        max_change = max(recent_changes) if recent_changes else 0

        if max_change >= threshold:
            return True, f"检测到黑天鹅级别波动: {max_change:.1%} >= {threshold:.1%}"

        return False, f"无黑天鹅事件: 最大波动{max_change:.1%}"

    def calculate_position_size_with_risk_adjustment(self, base_size, df=None):
        """根据风险状态计算调整后的仓位大小"""
        adjusted_size = base_size

        # 根据波动性调整
        if self.market_volatility == "HIGH":
            adjusted_size *= 1.0 - self.risk_params["volatility_adjustment_factor"]
        elif self.market_volatility == "LOW":
            adjusted_size *= (
                1.0 + self.risk_params["volatility_adjustment_factor"] * 0.5
            )

        # 根据回撤情况调整
        current_drawdown = 0
        if self.peak_balance > 0:
            current_drawdown = (
                self.peak_balance - self.current_balance
            ) / self.peak_balance

        if current_drawdown > self.risk_params["max_daily_drawdown"] * 0.5:
            # 回撤超过日限制的一半，减少仓位
            reduction = min(
                current_drawdown / self.risk_params["max_daily_drawdown"], 0.5
            )
            adjusted_size *= 1.0 - reduction

        return max(adjusted_size, base_size * 0.1)  # 至少保留10%的基础仓位

    def calculate_var(self, df, confidence_level=0.95, horizon_days=1):
        """计算风险价值(VaR)"""
        if df is None or len(df) < 100:
            return 0, "数据不足"

        # 计算历史回报
        returns = df["close"].pct_change().dropna()

        # 历史模拟法计算VaR
        var_historical = -np.percentile(returns, (1 - confidence_level) * 100)

        # 转换为美元价值
        var_usd = var_historical * self.current_balance

        return (
            var_usd,
            f"{confidence_level*100:.0f}%置信度下，{horizon_days}天最大损失约${var_usd:.2f}",
        )

    def get_risk_report(self):
        """生成风险报告"""
        current_drawdown = 0
        if self.peak_balance > 0:
            current_drawdown = (
                self.peak_balance - self.current_balance
            ) / self.peak_balance

        return {
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": current_drawdown,
            "max_drawdown": self.max_drawdown,
            "market_volatility": self.market_volatility,
            "risk_level": self.risk_level,
            "total_trades": len(self.trade_history),
            "recent_trades": (
                self.trade_history[-5:]
                if len(self.trade_history) >= 5
                else self.trade_history
            ),
            "risk_params": self.risk_params,
        }

    def adjust_risk_level(self, market_conditions, performance_metrics):
        """根据市场条件和表现调整风险水平"""
        # 简化版本：根据波动性和回撤调整风险水平
        if self.market_volatility == "HIGH" or self.max_drawdown > 0.05:
            self.risk_level = "CAUTIOUS"
        elif self.market_volatility == "LOW" and self.max_drawdown < 0.02:
            self.risk_level = "AGGRESSIVE"
        else:
            self.risk_level = "NORMAL"

        return self.risk_level

    def _save_backtest_state(self):
        """保存回测前的策略状态"""
        return {
            'current_position': self.current_position,
            'position_entry_price': self.position_entry_price,
            'position_size': self.position_size,
            'consecutive_reverse_count': self.consecutive_reverse_count,
            'last_reverse_signal': self.last_reverse_signal,
            'consecutive_entry_count': self.consecutive_entry_count,
            'last_entry_signal': self.last_entry_signal,
            'post_entry_time': self.post_entry_time,
            'post_entry_entry_count': self.post_entry_entry_count,
            '_last_had_position': self._last_had_position,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
            'trade_history': self.trade_history.copy() if hasattr(self, 'trade_history') else [],
            'last_trade_time': self.last_trade_time,
            'current_effective_strategy': self.current_effective_strategy,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_1_price': self.take_profit_1_price,
            'take_profit_2_price': self.take_profit_2_price,
            'tp1_hit': self.tp1_hit,
            # 策略切换确认计数器
            'pending_strategy_switch': self.pending_strategy_switch,
            'strategy_switch_confirm_count': self.strategy_switch_confirm_count,
        }

    def _restore_backtest_state(self, state):
        """恢复回测前的策略状态"""
        self.current_position = state['current_position']
        self.position_entry_price = state['position_entry_price']
        self.position_size = state['position_size']
        self.consecutive_reverse_count = state['consecutive_reverse_count']
        self.last_reverse_signal = state['last_reverse_signal']
        self.consecutive_entry_count = state['consecutive_entry_count']
        self.last_entry_signal = state['last_entry_signal']
        self.post_entry_time = state['post_entry_time']
        self.post_entry_entry_count = state['post_entry_entry_count']
        self._last_had_position = state['_last_had_position']
        self.daily_pnl = state['daily_pnl']
        self.daily_trades = state['daily_trades']
        self.consecutive_losses = state['consecutive_losses']
        self.current_balance = state['current_balance']
        self.peak_balance = state['peak_balance']
        self.max_drawdown = state['max_drawdown']
        self.trade_history = state['trade_history']
        self.last_trade_time = state.get('last_trade_time', None)
        self.current_effective_strategy = state.get('current_effective_strategy', 'auto')
        self.stop_loss_price = state.get('stop_loss_price', None)
        self.take_profit_1_price = state.get('take_profit_1_price', None)
        self.take_profit_2_price = state.get('take_profit_2_price', None)
        self.tp1_hit = state.get('tp1_hit', False)
        # 恢复策略切换确认计数器
        self.pending_strategy_switch = state.get('pending_strategy_switch', None)
        self.strategy_switch_confirm_count = state.get('strategy_switch_confirm_count', 0)

    def _reset_backtest_state(self, initial_capital):
        """重置回测状态"""
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        self._last_had_position = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.current_balance = initial_capital
        self.peak_balance = initial_capital
        self.max_drawdown = 0.0
        self.trade_history = []
        self.last_trade_time = None
        self.current_effective_strategy = "auto"
        # 重置止盈止损相关变量
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.tp1_hit = False
        # 重置策略切换确认计数器
        self.pending_strategy_switch = None
        self.strategy_switch_confirm_count = 0

    def run_backtest(self, df_historical, initial_capital=10000, fee_rate=0.001, slippage=0.0005, progress_callback=None):
        """
        运行完整回测
        
        Args:
            df_historical: 历史K线数据 DataFrame
            initial_capital: 初始资金
            fee_rate: 手续费率
            slippage: 滑点
            progress_callback: 进度回调函数
            
        Returns:
            dict: 回测结果
        """
        from datetime import datetime, timedelta
        
        print(f"\n{'='*80}")
        print("开始策略回测")
        print(f"{'='*80}")
        
        original_state = self._save_backtest_state()
        
        try:
            self._reset_backtest_state(initial_capital)
            
            backtest_trades = []
            equity_curve = [initial_capital]
            total_fees = 0
            
            lookback = 100
            total_candles = len(df_historical)
            
            for i in range(lookback, total_candles):
                if progress_callback:
                    progress = int((i - lookback) / (total_candles - lookback) * 100)
                    progress_callback(progress)
                
                df_slice = df_historical.iloc[:i+1].copy()
                current_price = df_slice['close'].iloc[-1]
                candle_time = df_slice['timestamps'].iloc[-1] if 'timestamps' in df_slice.columns else datetime.now()
                
                try:
                    signal = self.analyzer.get_enhanced_signal(df_slice)
                    
                    action, action_reason, trade_info = self._simulate_trading_decision(
                        df_slice, signal, current_price, candle_time, fee_rate, slippage
                    )
                    
                    if action:
                        backtest_trades.append(trade_info)
                        if trade_info.get('fee'):
                            total_fees += trade_info['fee']
                    
                    equity_curve.append(self.current_balance)
                    
                except Exception as e:
                    print(f"[回测错误] 第{i}根K线: {e}")
                    continue
            
            results = self._calculate_backtest_metrics(
                backtest_trades, equity_curve, initial_capital, total_fees
            )
            
            print(f"\n{'='*80}")
            print("回测完成！")
            print(f"总收益率: {results['total_return']:.2%}")
            print(f"胜率: {results['win_rate']:.2%}")
            print(f"总交易次数: {results['total_trades']}")
            print(f"{'='*80}\n")
            
            return results
            
        finally:
            self._restore_backtest_state(original_state)

    def _simulate_trading_decision(self, df, signal, current_price, candle_time, fee_rate, slippage):
        """模拟交易决策 - 完全和主程序一致"""
        action = False
        action_reason = ""
        trade_info = {}
        
        if self.current_position:
            self._last_had_position = True
            
            position_pnl_pct = 0
            if self.position_entry_price and current_price:
                if self.current_position == "LONG":
                    position_pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price * 100
                else:
                    position_pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price * 100
            
            should_close = False
            should_add = False
            close_reason = ""
            
            # 1. 检查固定止损
            if self.current_position == "LONG":
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            else:
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            
            # 2. 检查固定止盈1、止盈2和止盈3（分级止盈，部分平仓）
            if not should_close:
                # 从take_profit_config获取止盈仓位比例（与实盘一致）
                tp1_position_ratio = self.take_profit_config.get("tp1_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                tp2_position_ratio = self.take_profit_config.get("tp2_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                
                if self.current_position == "LONG":
                    if not self.tp1_hit and self.take_profit_1_price and current_price >= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price >= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price >= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
                else:
                    if not self.tp1_hit and self.take_profit_1_price and current_price <= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price <= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price <= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            if self.post_entry_time:
                if hasattr(candle_time, 'timestamp'):
                    hours_since_entry = (candle_time - self.post_entry_time).total_seconds() / 3600
                else:
                    hours_since_entry = 0
                
                if hours_since_entry < self.post_entry_hours:
                    is_same_direction = (
                        (self.current_position == "LONG" and signal["trend_direction"] == "LONG") or
                        (self.current_position == "SHORT" and signal["trend_direction"] == "SHORT")
                    )
                    
                    if is_same_direction:
                        self.post_entry_entry_count += 1
                        if self.post_entry_entry_count >= 2:
                            should_add = True
                    else:
                        self.post_entry_entry_count = 0
                else:
                    if position_pnl_pct >= self.take_profit_min_pct:
                        should_close = True
                        close_reason = f"超过{self.post_entry_hours}小时且盈利{position_pnl_pct:.2f}% > {self.take_profit_min_pct}%"
            
            if not should_close:
                if self.current_position == "LONG" and signal["trend_direction"] == "SHORT":
                    if "short" == self.last_reverse_signal:
                        self.consecutive_reverse_count += 1
                        if self.consecutive_reverse_count >= self.reverse_confirm_count:
                            should_close = True
                            if has_strong_signal:
                                if has_turning_point:
                                    close_reason = "强烈拐点检测做空（连续确认）"
                                else:
                                    close_reason = "强趋势反转做空（连续确认）"
                            else:
                                close_reason = "趋势反转做空（连续确认）"
                    else:
                        self.consecutive_reverse_count = 1
                        self.last_reverse_signal = "short"
                elif self.current_position == "SHORT" and signal["trend_direction"] == "LONG":
                    if "long" == self.last_reverse_signal:
                        self.consecutive_reverse_count += 1
                        if self.consecutive_reverse_count >= self.reverse_confirm_count:
                            should_close = True
                            if has_strong_signal:
                                if has_turning_point:
                                    close_reason = "强烈拐点检测做多（连续确认）"
                                else:
                                    close_reason = "强趋势反转做多（连续确认）"
                            else:
                                close_reason = "趋势反转做多（连续确认）"
                    else:
                        self.consecutive_reverse_count = 1
                        self.last_reverse_signal = "long"
                else:
                    pass
            
            if not should_close and position_pnl_pct < -3:
                should_close = True
                close_reason = f"亏损过大 ({position_pnl_pct:.2f}%)"
            
            if not should_close and not should_add and (
                position_pnl_pct > 1
                and signal["trend_strength"] > self.threshold
            ):
                if (
                    self.current_position == "LONG"
                    and signal["trend_direction"] == "LONG"
                ) or (
                    self.current_position == "SHORT"
                    and signal["trend_direction"] == "SHORT"
                ):
                    should_add = True
            
            if should_close:
                action = True
                action_reason = close_reason
                log_msg(f"[平仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - {close_reason}")
                trade_info = self._simulate_close_position(current_price, candle_time, close_reason, fee_rate, slippage)
                if trade_info:
                    log_msg(f"[平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}")
            elif should_add:
                pass
            
        else:
            if self._last_had_position:
                self._reset_full_state()
            else:
                self._reset_position_only()
            
            self._last_had_position = False
            
            signal_valid = signal.get("signal_valid", True)
            
            if not signal_valid:
                self.consecutive_entry_count = 0
                self.last_entry_signal = None
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_direction"] not in ["LONG", "SHORT"]:
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_strength"] < self.threshold:
                return action, action_reason, trade_info, log_messages
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            current_entry_signal = signal["trend_direction"]
            
            if current_entry_signal == self.last_entry_signal:
                self.consecutive_entry_count += 1
                
                if self.consecutive_entry_count >= self.entry_confirm_count:
                    action = True
                    action_reason = "开仓信号确认"
                    log_msg(f"[开仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 方向: {signal['trend_direction']}, 趋势强度: {signal_strength:.4f}")
                    trade_info = self._simulate_open_position(
                        signal, current_price, candle_time, fee_rate, slippage
                    )
                    if trade_info:
                        log_msg(f"[开仓完成] 价格: {trade_info['price']:.2f}, 数量: {trade_info['size']:.4f}, 手续费: {trade_info['fee']:.2f}")
                    self.consecutive_entry_count = 0
                    self.last_entry_signal = None
            else:
                self.consecutive_entry_count = 1
                self.last_entry_signal = current_entry_signal
        
        return action, action_reason, trade_info, log_messages

    def _simulate_open_position(self, signal, price, time, fee_rate, slippage):
        """模拟开仓"""
        direction = signal["trend_direction"]
        
        slippage_price = price * (1 + slippage) if direction == "LONG" else price * (1 - slippage)
        
        position_size = (self.current_balance * 0.1) / slippage_price
        position_size = max(position_size, 0.001)
        
        fee = slippage_price * position_size * fee_rate
        
        self.current_position = direction
        self.position_entry_price = slippage_price
        self.position_size = position_size
        self.initial_position_size = position_size
        self.post_entry_time = time
        self.post_entry_entry_count = 0
        
        # 设置止盈止损价格（优先使用 AI 推荐值）
        if "ai_stop_loss" in signal and "ai_take_profit_1" in signal:
            self.stop_loss_price = signal["ai_stop_loss"]
            self.take_profit_1_price = signal["ai_take_profit_1"]
            self.take_profit_2_price = signal.get("ai_take_profit_2", None)
            self.take_profit_3_price = signal.get("ai_take_profit_3", None)
        else:
            # 备选方案：使用默认值
            if direction == "LONG":
                self.stop_loss_price = slippage_price * 0.985  # 止损1.5%
                self.take_profit_1_price = slippage_price * 1.012  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 1.025  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 1.04  # 止盈3 4.0%
            else:
                self.stop_loss_price = slippage_price * 1.015  # 止损1.5%
                self.take_profit_1_price = slippage_price * 0.988  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 0.975  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 0.96  # 止盈3 4.0%
        
        self.tp1_hit = False
        self.tp2_hit = False
        
        trade_info = {
            "type": "OPEN",
            "direction": direction,
            "price": slippage_price,
            "size": position_size,
            "fee": fee,
            "time": time,
            "reason": "开仓"
        }
        
        self.trade_history.append(trade_info)
        self.current_balance -= fee
        
        return trade_info

    def _simulate_close_position(self, price, time, reason, fee_rate, slippage):
        """模拟平仓"""
        direction = self.current_position
        
        slippage_price = price * (1 - slippage) if direction == "LONG" else price * (1 + slippage)
        
        if direction == "LONG":
            pnl = (slippage_price - self.position_entry_price) * self.position_size
        else:
            pnl = (self.position_entry_price - slippage_price) * self.position_size
        
        fee = slippage_price * self.position_size * fee_rate
        
        self.current_balance += pnl - fee
        
        trade_info = {
            "type": "CLOSE",
            "direction": direction,
            "entry_price": self.position_entry_price,
            "exit_price": slippage_price,
            "size": self.position_size,
            "pnl": pnl,
            "pnl_pct": (pnl / (self.position_entry_price * self.position_size)) * 100 if self.position_entry_price else 0,
            "fee": fee,
            "time": time,
            "reason": reason
        }
        
        self.trade_history.append(trade_info)
        
        if self.peak_balance < self.current_balance:
            self.peak_balance = self.current_balance
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        
        return trade_info

    def _calculate_backtest_metrics(self, trades, equity_curve, initial_capital, total_fees):
        """计算回测指标"""
        closed_trades = [t for t in trades if t['type'] == 'CLOSE']
        
        total_trades = len(closed_trades)
        win_trades = len([t for t in closed_trades if t['pnl'] > 0])
        loss_trades = len([t for t in closed_trades if t['pnl'] <= 0])
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = total_pnl / initial_capital if initial_capital > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if win_trades > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in closed_trades if t['pnl'] <= 0]) if loss_trades > 0 else 0
        profit_factor = (avg_win * win_trades) / (avg_loss * loss_trades) if (avg_loss > 0 and loss_trades > 0) else float('inf')
        
        max_drawdown = self.max_drawdown
        
        avg_profit_pct = np.mean([t['pnl_pct'] for t in closed_trades]) if total_trades > 0 else 0
        
        return {
            "trades": trades,
            "closed_trades": closed_trades,
            "equity_curve": equity_curve,
            "initial_capital": initial_capital,
            "final_capital": self.current_balance,
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "avg_profit": avg_profit_pct,
            "total_fees": total_fees
        }

    def _save_backtest_state(self):
        """保存回测前的策略状态"""
        return {
            'current_position': self.current_position,
            'position_entry_price': self.position_entry_price,
            'position_size': self.position_size,
            'consecutive_reverse_count': self.consecutive_reverse_count,
            'last_reverse_signal': self.last_reverse_signal,
            'consecutive_entry_count': self.consecutive_entry_count,
            'last_entry_signal': self.last_entry_signal,
            'post_entry_time': self.post_entry_time,
            'post_entry_entry_count': self.post_entry_entry_count,
            '_last_had_position': self._last_had_position,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
            'trade_history': self.trade_history.copy() if hasattr(self, 'trade_history') else [],
            'last_trade_time': self.last_trade_time,
            'current_effective_strategy': self.current_effective_strategy,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_1_price': self.take_profit_1_price,
            'take_profit_2_price': self.take_profit_2_price,
            'tp1_hit': self.tp1_hit,
            # 策略切换确认计数器
            'pending_strategy_switch': self.pending_strategy_switch,
            'strategy_switch_confirm_count': self.strategy_switch_confirm_count,
        }

    def _restore_backtest_state(self, state):
        """恢复回测前的策略状态"""
        self.current_position = state['current_position']
        self.position_entry_price = state['position_entry_price']
        self.position_size = state['position_size']
        self.consecutive_reverse_count = state['consecutive_reverse_count']
        self.last_reverse_signal = state['last_reverse_signal']
        self.consecutive_entry_count = state['consecutive_entry_count']
        self.last_entry_signal = state['last_entry_signal']
        self.post_entry_time = state['post_entry_time']
        self.post_entry_entry_count = state['post_entry_entry_count']
        self._last_had_position = state['_last_had_position']
        self.daily_pnl = state['daily_pnl']
        self.daily_trades = state['daily_trades']
        self.consecutive_losses = state['consecutive_losses']
        self.current_balance = state['current_balance']
        self.peak_balance = state['peak_balance']
        self.max_drawdown = state['max_drawdown']
        self.trade_history = state['trade_history']
        self.last_trade_time = state.get('last_trade_time', None)
        self.current_effective_strategy = state.get('current_effective_strategy', 'auto')
        self.stop_loss_price = state.get('stop_loss_price', None)
        self.take_profit_1_price = state.get('take_profit_1_price', None)
        self.take_profit_2_price = state.get('take_profit_2_price', None)
        self.tp1_hit = state.get('tp1_hit', False)
        # 恢复策略切换确认计数器
        self.pending_strategy_switch = state.get('pending_strategy_switch', None)
        self.strategy_switch_confirm_count = state.get('strategy_switch_confirm_count', 0)

    def _reset_backtest_state(self, initial_capital):
        """重置回测状态"""
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.consecutive_reverse_count = 0
        self.last_reverse_signal = None
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        self._last_had_position = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.current_balance = initial_capital
        self.peak_balance = initial_capital
        self.max_drawdown = 0.0
        self.trade_history = []
        self.last_trade_time = None
        self.current_effective_strategy = "auto"
        # 重置止盈止损相关变量
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.tp1_hit = False
        # 重置策略切换确认计数器
        self.pending_strategy_switch = None
        self.strategy_switch_confirm_count = 0

    def run_backtest(self, df_historical, initial_capital=10000, fee_rate=0.001, slippage=0.0005, progress_callback=None):
        """
        运行完整回测
        
        Args:
            df_historical: 历史K线数据 DataFrame
            initial_capital: 初始资金
            fee_rate: 手续费率
            slippage: 滑点
            progress_callback: 进度回调函数
            
        Returns:
            dict: 回测结果
        """
        from datetime import datetime, timedelta
        
        print(f"\n{'='*80}")
        print("开始策略回测")
        print(f"{'='*80}")
        
        original_state = self._save_backtest_state()
        
        try:
            self._reset_backtest_state(initial_capital)
            
            backtest_trades = []
            equity_curve = [initial_capital]
            total_fees = 0
            
            lookback = 100
            total_candles = len(df_historical)
            
            for i in range(lookback, total_candles):
                if progress_callback:
                    progress = int((i - lookback) / (total_candles - lookback) * 100)
                    progress_callback(progress)
                
                df_slice = df_historical.iloc[:i+1].copy()
                current_price = df_slice['close'].iloc[-1]
                candle_time = df_slice.index[-1] if hasattr(df_slice, 'index') else datetime.now()
                
                try:
                    signal = self.analyzer.get_enhanced_signal(df_slice)
                    
                    action, action_reason, trade_info = self._simulate_trading_decision(
                        df_slice, signal, current_price, candle_time, fee_rate, slippage
                    )
                    
                    if action:
                        backtest_trades.append(trade_info)
                        if trade_info.get('fee'):
                            total_fees += trade_info['fee']
                    
                    equity_curve.append(self.current_balance)
                    
                except Exception as e:
                    print(f"[回测错误] 第{i}根K线: {e}")
                    continue
            
            results = self._calculate_backtest_metrics(
                backtest_trades, equity_curve, initial_capital, total_fees
            )
            
            print(f"\n{'='*80}")
            print("回测完成！")
            print(f"总收益率: {results['total_return']:.2%}")
            print(f"胜率: {results['win_rate']:.2%}")
            print(f"总交易次数: {results['total_trades']}")
            print(f"{'='*80}\n")
            
            return results
            
        finally:
            self._restore_backtest_state(original_state)

    def _simulate_trading_decision(self, df, signal, current_price, candle_time, fee_rate, slippage):
        """模拟交易决策 - 完全和主程序一致"""
        action = False
        action_reason = ""
        trade_info = {}
        
        if self.current_position:
            self._last_had_position = True
            
            position_pnl_pct = 0
            if self.position_entry_price and current_price:
                if self.current_position == "LONG":
                    position_pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price * 100
                else:
                    position_pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price * 100
            
            should_close = False
            should_add = False
            close_reason = ""
            
            # 1. 检查固定止损
            if self.current_position == "LONG":
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            else:
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    should_close = True
                    close_reason = "固定止损"
                    log_msg(f"[止损] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止损价: ${self.stop_loss_price:.2f}")
            
            # 2. 检查固定止盈1、止盈2和止盈3（分级止盈，部分平仓）
            if not should_close:
                # 从take_profit_config获取止盈仓位比例（与实盘一致）
                tp1_position_ratio = self.take_profit_config.get("tp1_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                tp2_position_ratio = self.take_profit_config.get("tp2_position_ratio", 0.35) if hasattr(self, 'take_profit_config') else 0.35
                
                if self.current_position == "LONG":
                    if not self.tp1_hit and self.take_profit_1_price and current_price >= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price >= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price >= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
                else:
                    if not self.tp1_hit and self.take_profit_1_price and current_price <= self.take_profit_1_price:
                        self.tp1_hit = True
                        close_size = self.initial_position_size * tp1_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈1", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈1(部分平仓)"
                            log_msg(f"[止盈1] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈1价: ${self.take_profit_1_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and not self.tp2_hit and self.take_profit_2_price and current_price <= self.take_profit_2_price:
                        self.tp2_hit = True
                        close_size = self.initial_position_size * tp2_position_ratio
                        trade_info = self._simulate_partial_close_position(current_price, candle_time, "固定止盈2", close_size, fee_rate, slippage)
                        if trade_info:
                            action = True
                            action_reason = "固定止盈2(部分平仓)"
                            log_msg(f"[止盈2] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈2价: ${self.take_profit_2_price:.2f}")
                            log_msg(f"[部分平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}, 剩余仓位: {trade_info['remaining_size']:.4f}")
                    elif self.tp1_hit and self.tp2_hit and self.take_profit_3_price and current_price <= self.take_profit_3_price:
                        should_close = True
                        close_reason = "固定止盈3"
                        log_msg(f"[止盈3] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 当前价格: ${current_price:.2f}, 止盈3价: ${self.take_profit_3_price:.2f}")
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            if self.post_entry_time:
                if hasattr(candle_time, 'timestamp'):
                    hours_since_entry = (candle_time - self.post_entry_time).total_seconds() / 3600
                else:
                    hours_since_entry = 0
                
                if hours_since_entry < self.post_entry_hours:
                    is_same_direction = (
                        (self.current_position == "LONG" and signal["trend_direction"] == "LONG") or
                        (self.current_position == "SHORT" and signal["trend_direction"] == "SHORT")
                    )
                    
                    if is_same_direction:
                        self.post_entry_entry_count += 1
                        if self.post_entry_entry_count >= 2:
                            should_add = True
                    else:
                        self.post_entry_entry_count = 0
                else:
                    if position_pnl_pct >= self.take_profit_min_pct:
                        should_close = True
                        close_reason = f"超过{self.post_entry_hours}小时且盈利{position_pnl_pct:.2f}% > {self.take_profit_min_pct}%"
            
            if not should_close:
                if self.current_position == "LONG" and signal["trend_direction"] == "SHORT":
                    if "short" == self.last_reverse_signal:
                        self.consecutive_reverse_count += 1
                        if self.consecutive_reverse_count >= self.reverse_confirm_count:
                            should_close = True
                            if has_strong_signal:
                                if has_turning_point:
                                    close_reason = "强烈拐点检测做空（连续确认）"
                                else:
                                    close_reason = "强趋势反转做空（连续确认）"
                            else:
                                close_reason = "趋势反转做空（连续确认）"
                    else:
                        self.consecutive_reverse_count = 1
                        self.last_reverse_signal = "short"
                elif self.current_position == "SHORT" and signal["trend_direction"] == "LONG":
                    if "long" == self.last_reverse_signal:
                        self.consecutive_reverse_count += 1
                        if self.consecutive_reverse_count >= self.reverse_confirm_count:
                            should_close = True
                            if has_strong_signal:
                                if has_turning_point:
                                    close_reason = "强烈拐点检测做多（连续确认）"
                                else:
                                    close_reason = "强趋势反转做多（连续确认）"
                            else:
                                close_reason = "趋势反转做多（连续确认）"
                    else:
                        self.consecutive_reverse_count = 1
                        self.last_reverse_signal = "long"
                else:
                    pass
            
            if not should_close and position_pnl_pct < -3:
                should_close = True
                close_reason = f"亏损过大 ({position_pnl_pct:.2f}%)"
            
            if not should_close and not should_add and (
                position_pnl_pct > 1
                and signal["trend_strength"] > self.threshold
            ):
                if (
                    self.current_position == "LONG"
                    and signal["trend_direction"] == "LONG"
                ) or (
                    self.current_position == "SHORT"
                    and signal["trend_direction"] == "SHORT"
                ):
                    should_add = True
            
            if should_close:
                action = True
                action_reason = close_reason
                log_msg(f"[平仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - {close_reason}")
                trade_info = self._simulate_close_position(current_price, candle_time, close_reason, fee_rate, slippage)
                if trade_info:
                    log_msg(f"[平仓完成] 盈亏: {trade_info['pnl']:+.2f} ({trade_info['pnl_pct']:+.2f}%), 手续费: {trade_info['fee']:.2f}")
            elif should_add:
                pass
            
        else:
            if self._last_had_position:
                self._reset_full_state()
            else:
                self._reset_position_only()
            
            self._last_had_position = False
            
            signal_valid = signal.get("signal_valid", True)
            
            if not signal_valid:
                self.consecutive_entry_count = 0
                self.last_entry_signal = None
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_direction"] not in ["LONG", "SHORT"]:
                return action, action_reason, trade_info, log_messages
            
            if signal["trend_strength"] < self.threshold:
                return action, action_reason, trade_info, log_messages
            
            has_strong_signal = False
            has_turning_point = signal.get("has_turning_point", False)
            signal_strength = signal.get("trend_strength", 0)
            
            if has_turning_point or signal_strength > self.threshold * 1.5:
                has_strong_signal = True
            
            current_entry_signal = signal["trend_direction"]
            
            if current_entry_signal == self.last_entry_signal:
                self.consecutive_entry_count += 1
                
                if self.consecutive_entry_count >= self.entry_confirm_count:
                    action = True
                    action_reason = "开仓信号确认"
                    log_msg(f"[开仓] {candle_time.strftime('%Y-%m-%d %H:%M:%S')} - 方向: {signal['trend_direction']}, 趋势强度: {signal_strength:.4f}")
                    trade_info = self._simulate_open_position(
                        signal, current_price, candle_time, fee_rate, slippage
                    )
                    if trade_info:
                        log_msg(f"[开仓完成] 价格: {trade_info['price']:.2f}, 数量: {trade_info['size']:.4f}, 手续费: {trade_info['fee']:.2f}")
                    self.consecutive_entry_count = 0
                    self.last_entry_signal = None
            else:
                self.consecutive_entry_count = 1
                self.last_entry_signal = current_entry_signal
        
        return action, action_reason, trade_info, log_messages

    def _simulate_open_position(self, signal, price, time, fee_rate, slippage):
        """模拟开仓"""
        direction = signal["trend_direction"]
        
        slippage_price = price * (1 + slippage) if direction == "LONG" else price * (1 - slippage)
        
        position_size = (self.current_balance * 0.1) / slippage_price
        position_size = max(position_size, 0.001)
        
        fee = slippage_price * position_size * fee_rate
        
        self.current_position = direction
        self.position_entry_price = slippage_price
        self.position_size = position_size
        self.initial_position_size = position_size
        self.post_entry_time = time
        self.post_entry_entry_count = 0
        
        # 设置止盈止损价格（优先使用 AI 推荐值）
        if "ai_stop_loss" in signal and "ai_take_profit_1" in signal:
            self.stop_loss_price = signal["ai_stop_loss"]
            self.take_profit_1_price = signal["ai_take_profit_1"]
            self.take_profit_2_price = signal.get("ai_take_profit_2", None)
            self.take_profit_3_price = signal.get("ai_take_profit_3", None)
        else:
            # 备选方案：使用默认值
            if direction == "LONG":
                self.stop_loss_price = slippage_price * 0.985  # 止损1.5%
                self.take_profit_1_price = slippage_price * 1.012  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 1.025  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 1.04  # 止盈3 4.0%
            else:
                self.stop_loss_price = slippage_price * 1.015  # 止损1.5%
                self.take_profit_1_price = slippage_price * 0.988  # 止盈1 1.2%
                self.take_profit_2_price = slippage_price * 0.975  # 止盈2 2.5%
                self.take_profit_3_price = slippage_price * 0.96  # 止盈3 4.0%
        
        self.tp1_hit = False
        self.tp2_hit = False
        
        trade_info = {
            "type": "OPEN",
            "direction": direction,
            "price": slippage_price,
            "size": position_size,
            "fee": fee,
            "time": time,
            "reason": "开仓"
        }
        
        self.trade_history.append(trade_info)
        self.current_balance -= fee
        
        return trade_info

    def _simulate_close_position(self, price, time, reason, fee_rate, slippage):
        """模拟平仓"""
        direction = self.current_position
        
        slippage_price = price * (1 - slippage) if direction == "LONG" else price * (1 + slippage)
        
        if direction == "LONG":
            pnl = (slippage_price - self.position_entry_price) * self.position_size
        else:
            pnl = (self.position_entry_price - slippage_price) * self.position_size
        
        fee = slippage_price * self.position_size * fee_rate
        
        self.current_balance += pnl - fee
        
        trade_info = {
            "type": "CLOSE",
            "direction": direction,
            "entry_price": self.position_entry_price,
            "exit_price": slippage_price,
            "size": self.position_size,
            "pnl": pnl,
            "pnl_pct": (pnl / (self.position_entry_price * self.position_size)) * 100 if self.position_entry_price else 0,
            "fee": fee,
            "time": time,
            "reason": reason
        }
        
        self.trade_history.append(trade_info)
        
        if self.peak_balance < self.current_balance:
            self.peak_balance = self.current_balance
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.post_entry_time = None
        self.post_entry_entry_count = 0
        
        return trade_info

    def _calculate_backtest_metrics(self, trades, equity_curve, initial_capital, total_fees):
        """计算回测指标"""
        closed_trades = [t for t in trades if t['type'] == 'CLOSE']
        
        total_trades = len(closed_trades)
        win_trades = len([t for t in closed_trades if t['pnl'] > 0])
        loss_trades = len([t for t in closed_trades if t['pnl'] <= 0])
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = total_pnl / initial_capital if initial_capital > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if win_trades > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in closed_trades if t['pnl'] <= 0]) if loss_trades > 0 else 0
        profit_factor = (avg_win * win_trades) / (avg_loss * loss_trades) if (avg_loss > 0 and loss_trades > 0) else float('inf')
        
        max_drawdown = self.max_drawdown
        
        avg_profit_pct = np.mean([t['pnl_pct'] for t in closed_trades]) if total_trades > 0 else 0
        
        return {
            "trades": trades,
            "closed_trades": closed_trades,
            "equity_curve": equity_curve,
            "initial_capital": initial_capital,
            "final_capital": self.current_balance,
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "avg_profit": avg_profit_pct,
            "total_fees": total_fees
        }
