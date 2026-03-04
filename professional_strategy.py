import os
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance_api import BinanceAPI
from enhanced_kronos import EnhancedKronosAnalyzer
from strategy_config import StrategyConfig
from strategy_profiles import StrategyProfiles
from market_state_analyzer import MarketStateAnalyzer, TimeStrategyAnalyzer
from binance.client import Client

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
    ):
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
        self.model_name = model_name or "kronos-small"
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

        # 趋势反转连续确认次数（默认2次）
        self.reverse_confirm_count = 2
        self.consecutive_reverse_count = 0

        # 开仓连续确认次数（默认2次）
        self.entry_confirm_count = 2
        self.consecutive_entry_count = 0
        self.last_entry_signal = None

        self.strategy_profile = StrategyProfiles.get_profile(strategy_type)

        # 自动/时间策略分析器
        self.market_analyzer = MarketStateAnalyzer(lookback_candles=100)
        self.time_analyzer = TimeStrategyAnalyzer()
        self.current_effective_strategy = strategy_type
        self.strategy_switch_count = 0

        self.binance = BinanceAPI()
        self.analyzer = EnhancedKronosAnalyzer(model_name=self.model_name)
        if self.analyzer is None:
            print("[错误] EnhancedKronosAnalyzer创建失败!")
            # 创建备用的技术分析器
            from enhanced_kronos import TechnicalAnalyzer, AlphaSignalProcessor

            class FallbackAnalyzer:
                def __init__(self):
                    self.use_kronos = False
                    self.tech_analyzer = TechnicalAnalyzer()
                    self.alpha_processor = AlphaSignalProcessor()

                def get_enhanced_signal(self, df):
                    return self.tech_analyzer.analyze(df)

            self.analyzer = FallbackAnalyzer()
            print("[备用] 使用技术分析器作为备用")
        self.binance.set_leverage(self.symbol, self.leverage)

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
        self.tp1_hit = False

        self.trade_history = []
        self.risk_manager = None
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
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

        # 初始化多智能体系统
        self.fingpt_analyzer = None
        self.strategy_coordinator = None
        self._initialize_multi_agent_system()

        self._initialize_balance()

    def _initialize_multi_agent_system(self):
        """初始化多智能体量化交易系统"""
        global MULTI_AGENT_AVAILABLE, FinGPTSentimentAnalyzer, StrategyCoordinator
        
        # 动态导入多智能体系统组件
        if FinGPTSentimentAnalyzer is None or StrategyCoordinator is None:
            try:
                print("[多智能体系统] 正在导入模块...")
                from fingpt_analyzer import FinGPTSentimentAnalyzer as FGSA
                from strategy_coordinator import StrategyCoordinator as SC
                FinGPTSentimentAnalyzer = FGSA
                StrategyCoordinator = SC
                MULTI_AGENT_AVAILABLE = True
                print("[多智能体系统] 模块导入成功")
            except Exception as e:
                print(f"[多智能体系统] 模块导入失败: {e}")
                MULTI_AGENT_AVAILABLE = False
                return
        
        if not MULTI_AGENT_AVAILABLE:
            print("[多智能体系统] 模块不可用，跳过初始化")
            return

        try:
            print("[多智能体系统] 正在初始化...")

            # 初始化FinGPT舆情分析器
            print("  正在初始化FinGPT舆情分析器...")
            self.fingpt_analyzer = FinGPTSentimentAnalyzer(
                use_local_model=True
            )
            print("  ✓ FinGPT舆情分析器初始化完成")

            # 初始化策略协调器
            print("  正在初始化策略协调器...")
            coin_symbol = self.symbol.replace("USDT", "")
            self.strategy_coordinator = StrategyCoordinator(
                symbol=coin_symbol,
                use_fingpt=True,
                kronos_analyzer=self.analyzer,
                fingpt_analyzer=self.fingpt_analyzer
            )
            print("  ✓ 策略协调器初始化完成")
            
            # 将多智能体系统实例传递给Kronos分析器
            self.analyzer.fingpt_analyzer = self.fingpt_analyzer
            self.analyzer.strategy_coordinator = self.strategy_coordinator

            print("[多智能体系统] 初始化完成!")

        except Exception as e:
            print(f"[多智能体系统] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.fingpt_analyzer = None
            self.strategy_coordinator = None

    def _initialize_balance(self):
        balance = self.binance.get_wallet_balance()
        if balance:
            self.starting_balance = balance
            print(f"初始余额(不含盈亏): ${self.starting_balance:.2f}")
            # 初始化风险管理器
            self.risk_manager = EnhancedRiskManager(self.starting_balance, self.symbol)
            print(f"风险管理器已初始化")
        else:
            # 如果获取余额失败，使用默认值初始化
            self.starting_balance = 0.0
            self.risk_manager = EnhancedRiskManager(0.0, self.symbol)
            print("警告: 获取余额失败，使用默认值初始化风险管理器")
        
        # 初始化余额缓存
        self._cached_balance = self.starting_balance

    def _reset_position_only(self):
        """只重置持仓状态，不重置开仓确认计数器（用于无持仓时）"""
        # 持仓状态
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.tp1_hit = False
        
        # 预测记录
        self.current_prediction = None
        
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
        self.tp1_hit = False
        
        # 开仓确认计数器
        self.consecutive_entry_count = 0
        self.last_entry_signal = None
        self.consecutive_reverse_count = 0
        
        # 预测记录
        self.current_prediction = None
        
        print("[状态重置] 所有持仓和开仓状态已完全重置")

    def get_total_balance(self):
        """获取合约账户总资金（包含可用+持仓盈亏）"""
        try:
            total_balance = self.binance.get_total_balance()
            if total_balance:
                # 更新风险管理器的余额
                if self.risk_manager:
                    self.risk_manager.update_balance(total_balance)
                # 缓存成功的余额值
                self._cached_balance = total_balance
                return total_balance
            
            # 如果获取失败，返回缓存的余额并尝试更新风险管理器
            if self.risk_manager:
                self.risk_manager.update_balance(0.0)
            return 0.0
            
        except Exception as e:
            print(f"获取总资金异常: {e}")
            # 网络错误时返回缓存的余额，不中断流程
            if hasattr(self, '_cached_balance') and self._cached_balance:
                print(f"使用缓存的余额: {self._cached_balance}")
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
            if daily_loss >= StrategyConfig.RISK_MANAGEMENT["daily_loss_limit"]:
                return False, f"触及单日亏损限制 {daily_loss*100:.1f}%"

        if self.daily_trades >= StrategyConfig.TRADE_FREQUENCY["max_daily_trades"]:
            return False, f"触及单日最大交易次数 {self.daily_trades}"

        if self.last_trade_time:
            time_since_last = (now - self.last_trade_time).total_seconds() / 60
            if (
                time_since_last
                < StrategyConfig.TRADE_FREQUENCY["min_trade_interval_minutes"]
            ):
                return (
                    False,
                    f"距离上次交易不足 {StrategyConfig.TRADE_FREQUENCY['min_trade_interval_minutes']} 分钟",
                )

        # 增强风险管理检查
        if self.risk_manager:
            # 检查回撤限制
            drawdown_ok, drawdown_msg = self.risk_manager.check_drawdown_limits()
            if not drawdown_ok:
                return False, f"风险管理: {drawdown_msg}"

            # 检查黑天鹅事件
            df = self.binance.get_recent_klines(
                self.symbol, self.timeframe, lookback=50
            )
            if df is not None and len(df) >= 20:
                black_swan, black_swan_msg = self.risk_manager.check_black_swan_event(
                    df
                )
                if black_swan:
                    print(f"警告: {black_swan_msg}")
                    # 黑天鹅事件不一定要停止交易，但可以记录警告

        return True, "风控检查通过"

    def _adaptive_parameter_optimization(self, df):
        """自适应参数优化"""
        from datetime import datetime, timedelta
        
        # 每30分钟优化一次参数
        if datetime.now() - self.adaptive_params["last_optimization"] < timedelta(minutes=30):
            return
            
        print("\n[自适应优化] 开始参数优化...")
        
        # 分析市场波动性
        volatility = df["close"].pct_change().std()
        
        # 根据波动性调整阈值
        if volatility > 0.02:  # 高波动市场
            new_threshold = min(self.threshold * 1.5, 0.03)
            print(f"  高波动市场，阈值调整: {self.threshold:.4f} -> {new_threshold:.4f}")
            self.threshold = new_threshold
        elif volatility < 0.005:  # 低波动市场
            new_threshold = max(self.threshold * 0.7, 0.005)
            print(f"  低波动市场，阈值调整: {self.threshold:.4f} -> {new_threshold:.4f}")
            self.threshold = new_threshold
            
        # 根据预测准确性调整确认次数
        if len(self.prediction_history) >= 10:
            recent_predictions = self.prediction_history[-10:]
            accuracy_rate = sum(1 for p in recent_predictions if p.get('direction_correct', False)) / 10
            
            if accuracy_rate > 0.7:  # 高准确率
                new_entry_count = max(1, self.entry_confirm_count - 1)
                new_reverse_count = max(1, self.reverse_confirm_count - 1)
                print(f"  高准确率({accuracy_rate:.1%})，确认次数减少: {self.entry_confirm_count}->{new_entry_count}")
                self.entry_confirm_count = new_entry_count
                self.reverse_confirm_count = new_reverse_count
            elif accuracy_rate < 0.3:  # 低准确率
                new_entry_count = min(5, self.entry_confirm_count + 1)
                new_reverse_count = min(5, self.reverse_confirm_count + 1)
                print(f"  低准确率({accuracy_rate:.1%})，确认次数增加: {self.entry_confirm_count}->{new_entry_count}")
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
            print(f"[性能反馈] 最近5次预测准确率: {recent_accuracy:.1%}")
            
            # 如果连续表现不佳，考虑策略切换
            if recent_accuracy < 0.2 and len(self.prediction_history) >= 20:
                last_20_accuracy = sum(1 for p in self.prediction_history[-20:] if p.get('direction_correct', False)) / 20
                if last_20_accuracy < 0.3:
                    print("⚠️ 警告: 预测准确率持续偏低，建议检查策略有效性")
        
        print("[自适应优化] 参数优化完成")

    def _determine_effective_strategy(self, df, signal):
        """根据市场状态和时间确定有效策略"""
        if self.strategy_type not in ["auto", "time"]:
            return self.strategy_type

        if len(df) < 50:
            return "trend"

        if self.strategy_type == "auto":
            market_state = self.market_analyzer.analyze(df)
            print(
                f"[市场状态] {market_state['state']}, 强度: {market_state['strength']:.2f}, 置信度: {market_state['confidence']:.2f}"
            )
            self.time_analyzer.record_market_state(df, market_state)
            return market_state["state"]

        elif self.strategy_type == "time":
            market_state = self.market_analyzer.analyze(df)
            self.time_analyzer.record_market_state(df, market_state)

            recommendation = self.time_analyzer.get_comprehensive_recommendation(
                self.market_analyzer
            )
            recommended = recommendation["recommended"]
            print(f"[时间策略] 推荐: {recommended}, 分数: {recommendation['scores']}")
            return recommended

        return "trend"

    def check_trading_hours(self):
        now = datetime.now()
        hour = now.hour
        start = StrategyConfig.TRADE_FREQUENCY["active_hours_start"]
        end = StrategyConfig.TRADE_FREQUENCY["active_hours_end"]

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
        return recent_change >= StrategyConfig.RISK_MANAGEMENT["extreme_move_threshold"]

    def calculate_kline_change(self, df):
        if len(df) < 2:
            return 0.0
        return (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1]

    def check_entry_conditions(self, signal, df, funding_rate):
        current_price = signal["current_price"]
        trend_direction = signal["trend_direction"]

        kline_change = self.calculate_kline_change(df)

        # 统一方向标识：BUY = LONG, SELL = SHORT
        if trend_direction in ["LONG", "BUY"]:
            if (
                current_price
                < signal["pred_support"] * StrategyConfig.ENTRY_FILTER["support_buffer"]
            ):
                return False, "价格低于支撑位缓冲"
            if kline_change < -StrategyConfig.ENTRY_FILTER["max_kline_change"]:
                return False, f"K线跌幅过大 {kline_change*100:.2f}%"
            if funding_rate > StrategyConfig.ENTRY_FILTER["max_funding_rate_long"]:
                return False, f"资金费率过高 {funding_rate*100:.6f}%"
            return True, "做多条件满足"

        elif trend_direction in ["SHORT", "SELL"]:
            if (
                current_price
                > signal["pred_resistance"]
                * StrategyConfig.ENTRY_FILTER["resistance_buffer"]
            ):
                return False, "价格高于阻力位缓冲"
            if kline_change > StrategyConfig.ENTRY_FILTER["max_kline_change"]:
                return False, f"K线涨幅过大 {kline_change*100:.2f}%"
            if funding_rate < StrategyConfig.ENTRY_FILTER["min_funding_rate_short"]:
                return False, f"资金费率过低 {funding_rate*100:.6f}%"
            return True, "做空条件满足"

        return False, f"无明确趋势方向: {trend_direction}"

    def calculate_position_size(self, entry_price, stop_loss_price):
        step_size, min_notional = self._get_symbol_filters()

        total_balance = self.get_total_balance()
        risk_amount = (
            total_balance * StrategyConfig.RISK_MANAGEMENT["single_trade_risk"]
        )
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk <= 0:
            size = (
                total_balance
                * StrategyConfig.RISK_MANAGEMENT["max_single_position"]
                / entry_price
            )
        else:
            size = (risk_amount / price_risk) * self.leverage

        max_size = (
            total_balance
            * StrategyConfig.RISK_MANAGEMENT["max_single_position"]
            * self.leverage
        ) / entry_price
        size = min(size, max_size)

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

        exit_rules = self.strategy_profile["exit_rules"]
        stop_loss_rules = exit_rules["stop_loss"]

        # 统一方向标识：BUY = LONG, SELL = SHORT
        if trend_direction in ["LONG", "BUY"]:
            side = Client.SIDE_BUY
        elif trend_direction in ["SHORT", "SELL"]:
            side = Client.SIDE_SELL
        else:
            print(f"❌ 无效的趋势方向: {trend_direction}")
            return False

        # 计算止损价
        stop_loss_type = stop_loss_rules["type"]
        if stop_loss_type in ["fixed", "tight"]:
            if trend_direction in ["LONG", "BUY"]:
                stop_loss = current_price * (1 - stop_loss_rules["long_pct"])
            else:
                stop_loss = current_price * (1 + stop_loss_rules["short_pct"])
        elif stop_loss_type == "candle_extreme":
            if trend_direction in ["LONG", "BUY"]:
                candle_low = signal.get(
                    "candle_low", signal.get("pred_low", current_price * 0.995)
                )
                stop_loss = candle_low * (1 - 0.001)  # 微小偏移
            else:
                candle_high = signal.get(
                    "candle_high", signal.get("pred_high", current_price * 1.005)
                )
                stop_loss = candle_high * (1 + 0.001)  # 微小偏移
        elif stop_loss_type == "ai_predicted":
            # 使用AI预测的支撑阻力位作为止损止盈
            if trend_direction in ["LONG", "BUY"]:
                stop_loss = (
                    signal["pred_support"] * StrategyConfig.STOP_LOSS["long_buffer"]
                )
            else:
                stop_loss = (
                    signal["pred_resistance"] * StrategyConfig.STOP_LOSS["short_buffer"]
                )
            direction_text = "pred_support" if trend_direction in ["LONG", "BUY"] else "pred_resistance"
            print(f"  使用AI预测: 止损={stop_loss:.2f} (基于{direction_text})")
        else:
            if trend_direction in ["LONG", "BUY"]:
                stop_loss = (
                    signal["pred_support"] * StrategyConfig.STOP_LOSS["long_buffer"]
                )
            else:
                stop_loss = (
                    signal["pred_resistance"] * StrategyConfig.STOP_LOSS["short_buffer"]
                )

        # 计算止盈价
        take_profit_rules = exit_rules["take_profit"]
        take_profit_type = take_profit_rules["type"]

        # 优先使用AI预测的目标价
        if stop_loss_type == "ai_predicted":
            if trend_direction in ["LONG", "BUY"]:
                tp1 = signal["pred_resistance"] * 0.995  # AI预测阻力位下方
                tp2 = signal["pred_high"] * 0.99  # AI预测最高价下方
            else:
                tp1 = signal["pred_support"] * 1.005  # AI预测支撑位上方
                tp2 = signal["pred_low"] * 1.01  # AI预测最低价上方
            print(f"  使用AI预测: 止盈1={tp1:.2f}, 止盈2={tp2:.2f}")
        elif take_profit_type == "trailing":
            if trend_direction in ["LONG", "BUY"]:
                tp1 = current_price * (1 + take_profit_rules.get("tp1_pct", 0.012))
                tp2 = current_price * (1 + take_profit_rules.get("tp2_pct", 0.025))
            else:
                tp1 = current_price * (1 - take_profit_rules.get("tp1_pct", 0.012))
                tp2 = current_price * (1 - take_profit_rules.get("tp2_pct", 0.025))
        elif take_profit_type == "midpoint":
            target_range = take_profit_rules.get("target_pct_range", [0.008, 0.012])
            target_pct = (target_range[0] + target_range[1]) / 2  # 取中间值
            if trend_direction in ["LONG", "BUY"]:
                tp1 = current_price * (1 + target_pct)
                tp2 = None  # 震荡策略可能只有一个止盈位
            else:
                tp1 = current_price * (1 - target_pct)
                tp2 = None
        elif take_profit_type == "fixed":
            target_pct = take_profit_rules.get("target_pct", 0.025)
            if trend_direction in ["LONG", "BUY"]:
                tp1 = current_price * (1 + target_pct)
                tp2 = None  # 固定止盈策略可能只有一个止盈位
            else:
                tp1 = current_price * (1 - target_pct)
                tp2 = None
        else:
            if trend_direction in ["LONG", "BUY"]:
                tp1 = current_price * (1 + 0.012)
                tp2 = current_price * (1 + 0.025)
            else:
                tp1 = current_price * (1 - 0.012)
                tp2 = current_price * (1 - 0.025)

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

        entry_rules = self.strategy_profile["entry_rules"]

        if entry_rules.get("position_type") == "staged":
            initial_size = position_size * entry_rules.get("first_position_pct", 0.5)
        else:
            initial_size = position_size

        tp2_display = f"${tp2:.2f}" if tp2 is not None else "无"
        print(f"开仓: {trend_direction}")
        print(f"入场价: ${current_price:.2f}")
        print(f"止损: ${stop_loss:.2f} ({stop_loss_rules.get('type', 'default')})")
        print(f"止盈1: ${tp1:.2f}, 止盈2: {tp2_display}")
        print(f"初始仓位: {initial_size}")

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
        self.stop_loss_price = stop_loss
        self.take_profit_1_price = tp1
        self.take_profit_2_price = tp2
        self.tp1_hit = False
        self.last_trade_time = datetime.now()
        self.daily_trades += 1

        # 保存当前预测，用于后续验证准确性
        self.current_prediction = {
            "open_time": datetime.now(),
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

        return True

    def close_position(self, reason=""):
        if not self.current_position:
            return False

        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt <= 0:
            self.current_position = None
            return False

        print(f"平仓: {self.current_position}, 原因: {reason}")

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
        
        # 计算止损价
        stop_loss_type = stop_loss_rules["type"]
        if stop_loss_type in ["fixed", "tight"]:
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * (1 - stop_loss_rules["long_pct"])
            else:
                new_stop = avg_price * (1 + stop_loss_rules["short_pct"])
        elif stop_loss_type == "ai_predicted":
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * StrategyConfig.STOP_LOSS["long_buffer"]
            else:
                new_stop = avg_price * StrategyConfig.STOP_LOSS["short_buffer"]
        else:
            if self.current_position in ["LONG", "BUY"]:
                new_stop = avg_price * StrategyConfig.STOP_LOSS["long_buffer"]
            else:
                new_stop = avg_price * StrategyConfig.STOP_LOSS["short_buffer"]
        
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

    def check_position_status(self):
        if not self.current_position:
            return

        current_price = self.get_current_price()
        if not current_price:
            return

        pos_type, pos_amt = self.get_current_position_info()
        if pos_amt <= 0:
            self.current_position = None
            return

        if self.current_position == "LONG":
            if self.stop_loss_price and current_price <= self.stop_loss_price:
                self.close_position("止损")
                self.consecutive_losses += 1
                # 止损后立即强制同步持仓状态
                pos_type, pos_amt = self.get_current_position_info()
                if pos_amt <= 0:
                    self.current_position = None
                    self.position_size = 0
                # 移除交易暂停机制，确保止损后继续分析
                # if (
                #     self.consecutive_losses
                #     >= StrategyConfig.RISK_MANAGEMENT["max_consecutive_losses"]
                # ):
                #     self.trading_paused_until = datetime.now() + timedelta(
                #         minutes=StrategyConfig.RISK_MANAGEMENT[
                #             "pause_after_losses_minutes"
                #         ]
                #     )
            elif (
                not self.tp1_hit
                and self.take_profit_1_price
                and current_price >= self.take_profit_1_price
            ):
                tp1_size = pos_amt * StrategyConfig.TAKE_PROFIT["tp1_position_ratio"]
                remaining_size = pos_amt - tp1_size
                print(f"触发止盈1，平仓 {tp1_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_sell(self.symbol, tp1_size)
                self.tp1_hit = True
                self.stop_loss_price = self.position_entry_price * 1.001
                # 为剩余仓位重新设置止损和止盈2订单
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
                and self.take_profit_2_price is not None
                and current_price >= self.take_profit_2_price
            ):
                self.close_position("止盈2")
                self.consecutive_losses = 0

        elif self.current_position == "SHORT":
            if self.stop_loss_price and current_price >= self.stop_loss_price:
                self.close_position("止损")
                self.consecutive_losses += 1
                # 止损后立即强制同步持仓状态
                pos_type, pos_amt = self.get_current_position_info()
                if pos_amt <= 0:
                    self.current_position = None
                    self.position_size = 0
                # 移除交易暂停机制，确保止损后继续分析
                # if (
                #     self.consecutive_losses
                #     >= StrategyConfig.RISK_MANAGEMENT["max_consecutive_losses"]
                # ):
                #     self.trading_paused_until = datetime.now() + timedelta(
                #         minutes=StrategyConfig.RISK_MANAGEMENT[
                #             "pause_after_losses_minutes"
                #         ]
                #     )
            elif (
                not self.tp1_hit
                and self.take_profit_1_price
                and current_price <= self.take_profit_1_price
            ):
                tp1_size = pos_amt * StrategyConfig.TAKE_PROFIT["tp1_position_ratio"]
                remaining_size = pos_amt - tp1_size
                print(f"触发止盈1，平仓 {tp1_size}，剩余仓位 {remaining_size}")
                self.binance.cancel_all_orders(self.symbol)
                self.binance.cancel_all_algo_orders(self.symbol)
                self.binance.place_market_buy(self.symbol, tp1_size)
                self.tp1_hit = True
                self.stop_loss_price = self.position_entry_price * 0.999
                # 为剩余仓位重新设置止损和止盈2订单
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
                and self.take_profit_2_price is not None
                and current_price <= self.take_profit_2_price
            ):
                self.close_position("止盈2")
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
            self.symbol, self.timeframe, lookback=StrategyConfig.LOOKBACK_PERIOD
        )
        if df is None or len(df) < 200:
            print("获取K线数据失败")
            total_time = time.time() - start_time
            print(f"总执行耗时: {total_time:.2f}秒")
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
            signal = self.analyzer.get_enhanced_signal(df)
            print(
                f"AI分析: 方向={signal['trend_direction']}, 强度={signal['trend_strength']:.4f}"
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
                        print(f"  过滤后信号: {coordinator_signal.get('final_signal', 'UNKNOWN')}")
                        print(f"  信号强度: {coordinator_signal.get('signal_strength', 0):.3f}")
                        print(f"  执行建议: {coordinator_signal.get('recommendation', 'UNKNOWN')}")
                        
                        if coordinator_signal.get('filtered', False):
                            print(f"  ⚠️ 信号被过滤原因: {coordinator_signal.get('filter_reason', '未知')}")
                        
                        final_signal = coordinator_signal.get('final_signal', 'NEUTRAL')
                        final_strength = coordinator_signal.get('signal_strength', 0)
                        
                        if final_signal != 'NEUTRAL':
                            print(f"\n[策略协调器] 采用综合信号: {final_signal} (强度: {final_strength:.3f})")
                            signal['trend_direction'] = final_signal
                            signal['trend_strength'] = final_strength
                        else:
                            print(f"\n[策略协调器] 建议观望，不执行交易")
                    else:
                        print("[策略协调器] 未能生成综合信号，使用原始Kronos信号")
                        
                except Exception as e:
                    print(f"[多智能体系统] 分析失败: {e}")
                    print("[多智能体系统] 使用原始Kronos信号继续交易")
                
                print("="*60 + "\n")

            # 获取当前盈亏
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

            print(f"当前盈亏: {position_pnl_pct:.2f}%")

            # AI决策
            should_close = False
            should_add = False
            close_reason = ""

            # 1. 强反转信号立即平仓（最高优先级）
            # 趋势强度超过阈值2倍时，不需要连续确认，立即平仓
            if self.current_position == "LONG" and signal["trend_direction"] == "SHORT":
                if signal["trend_strength"] > self.threshold * 2.0:
                    should_close = True
                    close_reason = "强趋势反转做空（立即平仓）"
                    print(f"  ⚡ 检测到强做空信号（强度: {signal['trend_strength']:.4f}），立即平仓")
                elif signal["trend_strength"] > self.threshold * 1.5:
                    self.consecutive_reverse_count += 1
                    print(f"  检测到做空信号，连续第{self.consecutive_reverse_count}次")
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        close_reason = "趋势反转做空（连续确认）"
                else:
                    self.consecutive_reverse_count = 0
            elif (
                self.current_position == "SHORT" and signal["trend_direction"] == "LONG"
            ):
                if signal["trend_strength"] > self.threshold * 2.0:
                    should_close = True
                    close_reason = "强趋势反转做多（立即平仓）"
                    print(f"  ⚡ 检测到强做多信号（强度: {signal['trend_strength']:.4f}），立即平仓")
                elif signal["trend_strength"] > self.threshold * 1.5:
                    self.consecutive_reverse_count += 1
                    print(f"  检测到做多信号，连续第{self.consecutive_reverse_count}次")
                    if self.consecutive_reverse_count >= self.reverse_confirm_count:
                        should_close = True
                        close_reason = "趋势反转做多（连续确认）"
                else:
                    self.consecutive_reverse_count = 0
            else:
                # 趋势未反转，重置计数
                self.consecutive_reverse_count = 0

            # 2. 亏损过大强制平仓
            if position_pnl_pct < -3:
                should_close = True
                close_reason = f"亏损过大 ({position_pnl_pct:.2f}%)"

            # 盈利且趋势有利可加仓
            if (
                not should_close
                and position_pnl_pct > 1
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
                print(f"AI决策: 平仓 - {close_reason}")
                self.close_position(close_reason)
            elif should_add:
                print(f"AI决策: 加仓 - 盈利{position_pnl_pct:.2f}%且趋势有利")
                self.add_position()
            else:
                print(f"AI决策: 持仓观望")
                self.check_position_status()

            total_time = time.time() - start_time
            print(f"总执行耗时: {total_time:.2f}秒")
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
        signal = self.analyzer.get_enhanced_signal(df)
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
                    
                    if final_signal != 'NEUTRAL':
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

        # 如果不能开仓，分析完市场后就结束
        if not can_open_position:
            total_time = time.time() - start_time
            print(f"总执行耗时: {total_time:.2f}秒")
            return

        # 自动/时间策略切换
        effective_strategy = self._determine_effective_strategy(df, signal)
        if effective_strategy != self.current_effective_strategy:
            print(
                f"[策略切换] {self.current_effective_strategy} -> {effective_strategy}"
            )
            self.current_effective_strategy = effective_strategy
            self.strategy_profile = StrategyProfiles.get_profile(effective_strategy)
            self.strategy_switch_count += 1

        effective_strategy = self.current_effective_strategy

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
            print("信号无效，跳过")
            total_time = time.time() - start_time
            print(f"总执行耗时: {total_time:.2f}秒")
            return

        entry_ok, entry_msg = self.check_entry_conditions(signal, df, funding_rate)
        if not entry_ok:
            print(f"入场条件不满足: {entry_msg}")
            total_time = time.time() - start_time
            print(f"总执行耗时: {total_time:.2f}秒")
            return

        print(f"入场条件满足: {entry_msg}")

        # 开仓连续确认机制
        current_entry_signal = signal["trend_direction"]

        if current_entry_signal == self.last_entry_signal:
            self.consecutive_entry_count += 1
            print(
                f"  检测到{current_entry_signal}信号，连续第{self.consecutive_entry_count}次"
            )

            if self.consecutive_entry_count >= self.entry_confirm_count:
                print(f"  连续{self.entry_confirm_count}次确认，执行开仓")
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
            print(f"  首次检测到{current_entry_signal}信号，开始计数")

        total_time = time.time() - start_time
        print(f"总执行耗时: {total_time:.2f}秒")

    def _check_trend_entry(self, signal, df, current_price, funding_rate):
        self.strategy_profile["ai_filter"]
        self.strategy_profile["entry_rules"]
        special_filters = self.strategy_profile.get("special_filters", {})

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

        # 1. 趋势强度检查（使用GUI设置的阈值）
        trend_ok = signal["trend_strength"] >= effective_threshold
        print(
            f"  [1] 趋势强度: {signal['trend_strength']:.4f} {'✓' if trend_ok else '✗'} 阈值 {effective_threshold}"
        )
        if not trend_ok:
            print("=" * 60)
            print(f"信号跳过 - 趋势强度不满足")
            print("=" * 60)
            return False

        # 2. AI最小趋势强度检查
        ai_trend_ok = signal["trend_strength"] >= self.ai_min_trend
        print(
            f"  [2] AI趋势强度: {signal['trend_strength']:.4f} {'✓' if ai_trend_ok else '✗'} 最小要求 {self.ai_min_trend}"
        )
        if not ai_trend_ok:
            print("=" * 60)
            print(f"信号跳过 - AI趋势强度不满足")
            print("=" * 60)
            return False

        # 3. 预测偏离度检查
        price_deviation_pct = self.ai_min_deviation
        deviation_ok = abs(signal["price_change_pct"]) >= price_deviation_pct
        print(
            f"  [3] 预测偏离度: {abs(signal['price_change_pct'])*100:.2f}% {'✓' if deviation_ok else '✗'} {price_deviation_pct*100}%"
        )
        if not deviation_ok:
            print("=" * 60)
            print(f"信号跳过 - 预测偏离度不满足")
            print("=" * 60)
            return False

        # 4. 资金费率检查（使用GUI设置的参数）
        funding_ok = self.min_funding <= funding_rate <= self.max_funding
        print(
            f"  [4] 资金费率: {funding_rate*100:.6f}% {'✓' if funding_ok else '✗'} [{self.min_funding*100:.2f}%, {self.max_funding*100:.2f}%]"
        )
        if not funding_ok:
            reason = "过高" if funding_rate > self.max_funding else "过低"
            print("=" * 60)
            print(f"信号跳过 - 资金费率{reason}")
            print("=" * 60)
            return False

        print("-" * 60)
        print("  ✓ 所有入场条件检查通过")
        print("=" * 60)

        # 市场状态过滤
        if is_alpha_signal:
            if (
                market_state == "ranging"
                and "forbid_in_range" in special_filters
                and special_filters["forbid_in_range"]
            ):
                print(f"震荡市场({market_state})，禁止趋势交易")
                return False

        # 趋势爆发策略：只要趋势方向明确就开仓，不需要等突破
        print(f"触发趋势{signal['trend_direction']}: 趋势方向明确")
        print(f"  当前价格: ${current_price:.2f}")
        print(f"  预测方向: {signal['trend_direction']}")
        print(f"  预测变化: {signal['price_change_pct']*100:.2f}%")
        print(f"  预测支撑: ${signal['pred_support']:.2f}")
        print(f"  预测阻力: ${signal['pred_resistance']:.2f}")

        return True

    def _check_range_entry(self, signal, df, current_price, funding_rate):
        ai_filter = self.strategy_profile["ai_filter"]
        self.strategy_profile.get("special_filters", {})

        effective_threshold = self.threshold

        print("=" * 60)
        print("震荡套利入场条件检查:")
        print("-" * 60)

        # 1. 震荡市检查（趋势强度不能太大）
        range_ok = signal["trend_strength"] <= effective_threshold
        print(
            f"  [1] 震荡市检查: 趋势强度 {signal['trend_strength']:.4f} {'✓' if range_ok else '✗'} <= {effective_threshold}"
        )
        if not range_ok:
            print("=" * 60)
            print(f"信号跳过 - 趋势强度过大，非震荡市")
            print("=" * 60)
            return False

        # 2. AI最小趋势强度检查
        ai_trend_ok = signal["trend_strength"] >= self.ai_min_trend
        print(
            f"  [2] AI趋势强度: {signal['trend_strength']:.4f} {'✓' if ai_trend_ok else '✗'} 最小要求 {self.ai_min_trend}"
        )
        if not ai_trend_ok:
            print("=" * 60)
            print(f"信号跳过 - AI趋势强度不满足")
            print("=" * 60)
            return False

        # 3. 预测偏离度检查
        price_deviation_pct = self.ai_min_deviation
        deviation_ok = abs(signal["price_change_pct"]) >= price_deviation_pct
        print(
            f"  [3] 预测偏离度: {abs(signal['price_change_pct'])*100:.2f}% {'✓' if deviation_ok else '✗'} {price_deviation_pct*100}%"
        )
        if not deviation_ok:
            print("=" * 60)
            print(f"信号跳过 - 预测偏离度不满足")
            print("=" * 60)
            return False

        # 4. 资金费率检查
        funding_ok = self.min_funding <= funding_rate <= self.max_funding
        print(
            f"  [4] 资金费率: {funding_rate*100:.6f}% {'✓' if funding_ok else '✗'} [{self.min_funding*100:.2f}%, {self.max_funding*100:.2f}%]"
        )
        if not funding_ok:
            reason = "过高" if funding_rate > self.max_funding else "过低"
            print("=" * 60)
            print(f"信号跳过 - 资金费率{reason}")
            print("=" * 60)
            return False

        # 5. 支撑阻力距离检查
        sr_distance = (signal["pred_resistance"] - signal["pred_support"]) / signal[
            "pred_support"
        ]
        sr_ok = sr_distance <= ai_filter["max_support_resistance_distance"]
        print(
            f"  [5] 支撑阻力距离: {sr_distance*100:.2f}% {'✓' if sr_ok else '✗'} <= {ai_filter['max_support_resistance_distance']*100}%"
        )
        if not sr_ok:
            print("=" * 60)
            print(f"信号跳过 - 支撑阻力距离过大")
            print("=" * 60)
            return False

        # 6. 价格位置检查 - 震荡套利：只要在震荡市就根据方向开仓
        mid_price = (signal["pred_support"] + signal["pred_resistance"]) / 2
        distance_from_mid = (current_price - mid_price) / mid_price

        print(f"  [6] 震荡区间:")
        print(f"      支撑位: ${signal['pred_support']:.2f}")
        print(f"      中间位: ${mid_price:.2f}")
        print(f"      当前价: ${current_price:.2f}")
        print(f"      阻力位: ${signal['pred_resistance']:.2f}")
        print(f"      偏离中间: {distance_from_mid*100:.2f}%")

        # 震荡套利：只要趋势强度低（震荡市），就根据预测方向开仓
        print(f"  ✓ 震荡市交易机会")
        print(f"  触发震荡{signal['trend_direction']}: 震荡市根据方向开仓")
        print("=" * 60)
        return True

    def _check_breakout_entry(self, signal, df, current_price, funding_rate):
        self.strategy_profile["ai_filter"]
        self.strategy_profile.get("special_filters", {})

        effective_threshold = self.threshold

        print("=" * 60)
        print("消息突破入场条件检查:")
        print("-" * 60)

        # 1. 趋势强度检查
        trend_ok = signal["trend_strength"] >= effective_threshold
        print(
            f"  [1] 趋势强度: {signal['trend_strength']:.4f} {'✓' if trend_ok else '✗'} >= {effective_threshold}"
        )
        if not trend_ok:
            print("=" * 60)
            print(f"信号跳过 - 趋势强度不足")
            print("=" * 60)
            return False

        # 2. AI最小趋势强度检查
        ai_trend_ok = signal["trend_strength"] >= self.ai_min_trend
        print(
            f"  [2] AI趋势强度: {signal['trend_strength']:.4f} {'✓' if ai_trend_ok else '✗'} 最小要求 {self.ai_min_trend}"
        )
        if not ai_trend_ok:
            print("=" * 60)
            print(f"信号跳过 - AI趋势强度不满足")
            print("=" * 60)
            return False

        # 3. 预测偏离度检查
        price_deviation_pct = self.ai_min_deviation
        deviation_ok = abs(signal["price_change_pct"]) >= price_deviation_pct
        print(
            f"  [3] 预测偏离度: {abs(signal['price_change_pct'])*100:.2f}% {'✓' if deviation_ok else '✗'} {price_deviation_pct*100}%"
        )
        if not deviation_ok:
            print("=" * 60)
            print(f"信号跳过 - 预测偏离度不满足")
            print("=" * 60)
            return False

        # 4. 资金费率检查
        funding_ok = self.min_funding <= funding_rate <= self.max_funding
        print(
            f"  [4] 资金费率: {funding_rate*100:.6f}% {'✓' if funding_ok else '✗'} [{self.min_funding*100:.2f}%, {self.max_funding*100:.2f}%]"
        )
        if not funding_ok:
            reason = "过高" if funding_rate > self.max_funding else "过低"
            print("=" * 60)
            print(f"信号跳过 - 资金费率{reason}")
            print("=" * 60)
            return False

        # 5. 显示交易信息
        print(f"  [5] 交易信息:")
        print(f"      支撑位: ${signal['pred_support']:.2f}")
        print(f"      当前价: ${current_price:.2f}")
        print(f"      阻力位: ${signal['pred_resistance']:.2f}")
        print(f"      方向: {signal['trend_direction']}")

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
