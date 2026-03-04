import argparse
from dotenv import load_dotenv
from professional_strategy import ProfessionalTradingStrategy
from binance_api import BinanceAPI
from enhanced_kronos import EnhancedKronosAnalyzer
from strategy_config import StrategyConfig


def test_connection():
    print("=" * 80)
    print("测试币安API连接")
    print("=" * 80)
    load_dotenv()
    api = BinanceAPI(testnet=False)

    balance = api.get_account_balance()
    if balance:
        print("连接成功！")
        print("\n账户余额:")
        for b in balance:
            available = float(b["availableBalance"])
            wallet = float(b["crossWalletBalance"])
            if available + wallet > 0:
                print(f"  {b['asset']}: 可用 {available:.2f}, 钱包 {wallet:.2f}")

    symbol = "BTCUSDT"
    funding_rate = api.get_funding_rate(symbol)
    print(f"\n{symbol} 资金费率: {funding_rate*100:.4f}%")

    df = api.get_recent_klines(symbol, "5m", lookback=10)
    if df is not None:
        print(f"\n最近5分钟K线数据 ({len(df)}条):")
        print(df[["timestamps", "open", "high", "low", "close"]].tail())

    print("\n" + "=" * 80)


def test_signal():
    print("=" * 80)
    print("测试Kronos增强信号分析")
    print("=" * 80)
    load_dotenv()

    api = BinanceAPI(testnet=False)
    symbol = "BTCUSDT"

    print(f"正在获取 {StrategyConfig.LOOKBACK_PERIOD} 根5分钟K线数据...")
    df = api.get_recent_klines(symbol, "5m", lookback=StrategyConfig.LOOKBACK_PERIOD)
    if df is None:
        print("获取数据失败！")
        return

    print(f"获取到 {len(df)} 条K线数据")
    print(f"最新价格: ${df['close'].iloc[-1]:.2f}")

    print("\n正在加载Kronos模型并分析...")
    analyzer = EnhancedKronosAnalyzer()
    signal = analyzer.get_enhanced_signal(df)

    print("\n" + "=" * 80)
    print("Kronos 增强信号分析结果")
    print("=" * 80)
    print(f"趋势方向: {signal['trend_direction']}")
    print(f"趋势强度: {signal['trend_strength']:.4f}")
    print(f"趋势强度阈值: {StrategyConfig.TREND_STRENGTH_THRESHOLD}")
    print(f"信号有效性: {'✓ 有效' if signal['signal_valid'] else '✗ 无效'}")
    print(f"\n当前价格: ${signal['current_price']:.2f}")
    print(f"预测价格: ${signal['predicted_price']:.2f}")
    print(f"预测价格变化: {signal['price_change_pct']*100:+.2f}%")
    print(f"\n预测支撑位: ${signal['pred_support']:.2f}")
    print(f"预测阻力位: ${signal['pred_resistance']:.2f}")
    print(f"预测最低: ${signal['pred_low']:.2f}")
    print(f"预测最高: ${signal['pred_high']:.2f}")

    print("\n" + "=" * 80)


def run_once():
    print("=" * 80)
    print("执行一次专业策略分析")
    print("=" * 80)
    strategy = ProfessionalTradingStrategy(testnet=False)
    strategy.run_once()


def run_continuous():
    print("=" * 80)
    print("启动Kronos专业交易策略 - 连续运行模式")
    print("=" * 80)
    strategy = ProfessionalTradingStrategy(testnet=False)
    interval = 120
    strategy.run_loop(interval_seconds=interval)


def main():
    pass

    parser = argparse.ArgumentParser(
        description="Kronos专业交易策略 - BTC/USDT 5分钟短线"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="continuous",
        choices=["test", "signal", "once", "continuous"],
        help="运行模式: test(测试连接), signal(测试信号), once(执行一次), continuous(连续运行)",
    )

    args = parser.parse_args()

    if args.mode == "test":
        test_connection()
    elif args.mode == "signal":
        test_signal()
    elif args.mode == "once":
        run_once()
    elif args.mode == "continuous":
        run_continuous()


if __name__ == "__main__":
    main()
