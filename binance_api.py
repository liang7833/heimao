import os
import time
import math
import datetime
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from binance.client import Client

try:
    from binance.exceptions import BinanceAPIError, BinanceOrderException
except ImportError:
    try:
        from binance.exceptions import BinanceAPIException as BinanceAPIError
    except:
        BinanceAPIError = Exception
    BinanceOrderException = Exception
import requests


class BinanceAPI:
    def __init__(self):
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
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.timestamp_offset = 0
        self.client = self._init_client()
        self._sync_time()

    def _sync_time(self):
        try:
            server_time = self.client.futures_time()
            server_ts = server_time["serverTime"]
            local_ts = int(time.time() * 1000)
            time_diff = server_ts - local_ts
            self.timestamp_offset = time_diff
            self.client.timestamp_offset = time_diff
            print(f"时间同步完成，偏移: {time_diff}ms")
        except Exception as e:
            print(f"时间同步失败: {e}")
            self.timestamp_offset = 0
            self.client.timestamp_offset = 0

    def _init_client(self):
        client = Client(self.api_key, self.api_secret, requests_params={"timeout": 30})
        # 尝试设置recvWindow属性（某些版本的binance库支持）
        try:
            client.recvWindow = 10000
        except:
            pass
        return client

    def get_klines(self, symbol, interval, limit=1000):
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval=interval, limit=limit
            )
            if not klines or len(klines) == 0:
                print(f"获取K线数据失败：返回空数据")
                return None
                
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            df = df.astype(
                {
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float,
                    "quote_asset_volume": float,
                    "taker_buy_base": float,
                    "taker_buy_quote": float,
                }
            )
            df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["amount"] = df["quote_asset_volume"]
            
            result_df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
            
            if result_df.isnull().any().any():
                print(f"警告：获取到的数据包含NaN值，已清理")
                result_df = result_df.dropna()
            
            return result_df
        except Exception as e:
            print(f"获取K线数据错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_recent_klines(self, symbol, interval, lookback=512):
        return self.get_klines(symbol, interval, limit=lookback)

    def get_historical_klines(
        self, symbol, interval, start_str=None, end_str=None, limit=1500
    ):
        """
        获取历史K线数据，支持分批次下载
        """
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_str,
                endTime=end_str,
                limit=limit,
            )
            if not klines or len(klines) == 0:
                print(f"获取历史K线数据失败：返回空数据")
                return None
                
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            df = df.astype(
                {
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float,
                    "quote_asset_volume": float,
                    "taker_buy_base": float,
                    "taker_buy_quote": float,
                }
            )
            df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["amount"] = df["quote_asset_volume"]
            
            result_df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
            
            if result_df.isnull().any().any():
                print(f"警告：获取到的历史数据包含NaN值，已清理")
                result_df = result_df.dropna()
            
            return result_df
        except Exception as e:
            print(f"获取历史K线数据错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_account_balance(self):
        try:
            # 使用较大的recvWindow以减少时间戳错误
            balance = self.client.futures_account_balance(recvWindow=10000)
            return balance
        except Exception as e:
            error_str = str(e)
            print(f"获取账户余额错误: {error_str}")

            # 检查是否是时间戳错误（错误代码-1021）
            if (
                "Timestamp for this request is outside of the recvWindow" in error_str
                or "code=-1021" in error_str
            ):
                print("检测到时间戳错误，尝试重新同步时间并重试...")
                try:
                    # 重新同步时间
                    self._sync_time()
                    # 重试一次，使用更大的recvWindow
                    balance = self.client.futures_account_balance(recvWindow=15000)
                    print("时间重新同步成功，重试获取余额")
                    return balance
                except Exception as retry_e:
                    print(f"重试失败: {retry_e}")

            return None

    def get_total_balance(self):
        """获取合约账户总资金（包含可用+持仓盈亏）"""
        try:
            account_info = self.client.futures_account(recvWindow=10000)
            if account_info:
                total_balance = float(account_info.get("totalMarginBalance", 0))
                return total_balance
            return None
        except Exception as e:
            print(f"获取总资金错误: {e}")
            return None

    def get_wallet_balance(self):
        """获取合约账户钱包余额（不含持仓盈亏）"""
        try:
            account_info = self.client.futures_account(recvWindow=10000)
            if account_info:
                wallet_balance = float(account_info.get("totalWalletBalance", 0))
                return wallet_balance
            return None
        except Exception as e:
            print(f"获取钱包余额错误: {e}")
            return None

    def get_position(self, symbol):
        """获取持仓信息（带重试机制）"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                positions = self.client.futures_position_information(symbol=symbol)
                for pos in positions:
                    if pos["symbol"] == symbol:
                        return pos
                if attempt == 0:
                    print(f"  未找到 {symbol} 的持仓信息，返回None")
                return None
            except Exception as e:
                print(f"获取持仓信息错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None

    def set_leverage(self, symbol, leverage):
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"杠杆设置为: {leverage}x")
        except Exception as e:
            print(f"设置杠杆错误: {e}")

    def place_order(self, symbol, side, quantity, order_type=Client.ORDER_TYPE_MARKET):
        try:
            # 先调整精度
            adjusted_quantity, _ = self._adjust_quantity_and_price(symbol, quantity)
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=adjusted_quantity,
                recvWindow=10000,
            )
            print(f"订单已提交: {order}")
            return order
        except Exception as e:
            error_str = str(e)
            print(f"下单错误: {error_str}")

            # 检查是否是精度错误（错误代码-1111）
            if "code=-1111" in error_str:
                print("检测到精度错误，尝试使用精度调整重试...")
                try:
                    # 使用精度调整函数
                    adjusted_qty, _ = self._adjust_quantity_and_price(symbol, quantity)
                    print(f"调整精度后quantity: {adjusted_qty}")
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type=order_type,
                        quantity=adjusted_qty,
                        recvWindow=10000,
                    )
                    print(f"精度调整后订单成功: {order}")
                    return order
                except Exception as retry_e:
                    print(f"精度调整后仍失败: {retry_e}")

            # 检查是否是时间戳错误（错误代码-1021）
            if (
                "Timestamp for this request is outside of the recvWindow" in error_str
                or "code=-1021" in error_str
            ):
                print("检测到时间戳错误，尝试重新同步时间并重试...")
                try:
                    # 重新同步时间
                    self._sync_time()
                    # 重试一次，使用更大的recvWindow
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type=order_type,
                        quantity=quantity,
                        recvWindow=15000,
                    )
                    print("时间重新同步成功，重试下单")
                    return order
                except Exception as retry_e:
                    print(f"重试失败: {retry_e}")

            return None

    def place_market_buy(self, symbol, quantity):
        return self.place_order(symbol, Client.SIDE_BUY, quantity)

    def place_market_sell(self, symbol, quantity):
        return self.place_order(symbol, Client.SIDE_SELL, quantity)

    def get_symbol_info(self, symbol):
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    return s
            return None
        except Exception as e:
            print(f"获取交易对信息错误: {e}")
            return None

    def _adjust_quantity_and_price(self, symbol, quantity, price=None):
        """调整数量和价格精度
        
        Args:
            symbol: 交易对符号
            quantity: 原始数量
            price: 原始价格（可选）
            
        Returns:
            (adjusted_quantity, adjusted_price) 或 (adjusted_quantity, None)
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return quantity, price
        
        adjusted_quantity = quantity
        adjusted_price = price
        
        print(f"  精度调整前: quantity={quantity}, price={price}")
        
        # 调整数量精度
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = float(f.get("stepSize", 0.001))
                quantity_precision = len(str(step_size).split(".")[-1].rstrip("0"))
                # 向下取整到步长精度，而不是四舍五入
                adjusted_quantity = math.floor(quantity / step_size) * step_size
                adjusted_quantity = round(adjusted_quantity, quantity_precision)
                print(f"  数量步长: {step_size}, 精度: {quantity_precision}, 调整后数量: {adjusted_quantity}")
                break
        
        # 调整价格精度
        if price is not None:
            price_precision = symbol_info.get("pricePrecision", 2)
            # 使用字符串格式化确保精确的小数位数
            adjusted_price = float("{0:.{1}f}".format(price, price_precision))
            print(f"  价格精度: {price_precision}, 调整后价格: {adjusted_price}")
        
        # 确保返回Python原生类型，避免numpy类型导致签名不一致
        if hasattr(adjusted_quantity, 'item'):
            # numpy标量类型，使用.item()方法转换为Python原生类型
            adjusted_quantity = adjusted_quantity.item()
        else:
            # 其他类型，尝试转换为float
            try:
                adjusted_quantity = float(adjusted_quantity)
            except (TypeError, ValueError):
                pass
        
        if adjusted_price is not None:
            if hasattr(adjusted_price, 'item'):
                adjusted_price = adjusted_price.item()
            else:
                try:
                    adjusted_price = float(adjusted_price)
                except (TypeError, ValueError):
                    pass
        
        print(f"  精度调整后: quantity={adjusted_quantity}, price={adjusted_price}")
        return adjusted_quantity, adjusted_price

    def _get_mark_price_info(self, symbol):
        try:
            return self.client.futures_mark_price(symbol=symbol)
        except Exception as e:
            print(f"获取标记价格信息错误: {e}")
            return None

    def get_funding_rate(self, symbol):
        try:
            mark_price_info = self._get_mark_price_info(symbol)
            if mark_price_info and "lastFundingRate" in mark_price_info:
                return float(mark_price_info["lastFundingRate"])
            return 0.0
        except Exception as e:
            print(f"获取资金费率错误: {e}")
            return 0.0

    def get_mark_price(self, symbol):
        try:
            mark_price_info = self._get_mark_price_info(symbol)
            if mark_price_info and "markPrice" in mark_price_info:
                return float(mark_price_info["markPrice"])
            return None
        except Exception as e:
            print(f"获取标记价格错误: {e}")
            return None

    def get_order_book(self, symbol, limit=20):
        try:
            order_book = self.client.futures_order_book(symbol=symbol, limit=limit)
            return order_book
        except Exception as e:
            print(f"获取订单簿错误: {e}")
            return None

    def place_stop_loss_order(self, symbol, side, quantity, stop_price):
        """创建止损订单（使用算法订单API）"""
        try:
            return self.place_algo_stop_loss(symbol, side, quantity, stop_price)
        except Exception as e:
            print(f"止损订单错误: {e}")
            return None

    def place_take_profit_order(self, symbol, side, quantity, stop_price):
        """创建止盈订单（使用算法订单API）"""
        try:
            return self.place_algo_take_profit(symbol, side, quantity, stop_price)
        except Exception as e:
            print(f"止盈订单错误: {e}")
            return None

    def place_traditional_stop_loss(self, symbol, side, quantity, stop_price):
        """创建传统止损订单"""
        try:
            # 调整精度
            adjusted_quantity, adjusted_stop_price = self._adjust_quantity_and_price(
                symbol, quantity, stop_price
            )
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=adjusted_quantity,
                stopPrice=adjusted_stop_price,
                workingType="CONTRACT_PRICE",
                recvWindow=10000,
            )
            print(f"传统止损订单已提交: {order}")
            return order
        except Exception as e:
            print(f"传统止损订单错误: {e}")
            return None

    def place_traditional_take_profit(self, symbol, side, quantity, stop_price):
        """创建传统止盈订单"""
        try:
            # 调整精度
            adjusted_quantity, adjusted_stop_price = self._adjust_quantity_and_price(
                symbol, quantity, stop_price
            )
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                quantity=adjusted_quantity,
                stopPrice=adjusted_stop_price,
                workingType="CONTRACT_PRICE",
                recvWindow=10000,
            )
            print(f"传统止盈订单已提交: {order}")
            return order
        except Exception as e:
            print(f"传统止盈订单错误: {e}")
            return None

    def test_stop_loss_order(self, symbol, side, quantity, stop_price):
        """测试止损订单（使用算法订单API）"""
        try:
            # 调整精度
            adjusted_quantity, adjusted_stop_price = self._adjust_quantity_and_price(
                symbol, quantity, stop_price
            )
            
            # 使用positionSide="BOTH"（适用于双向持仓模式）
            position_side = "BOTH"
            
            print(f"测试止损订单参数（算法订单）:")
            print(f"  symbol: {symbol}")
            print(f"  side: {side}")
            print(f"  positionSide: {position_side}")
            print(f"  algoType: CONDITIONAL")
            print(f"  type: STOP_MARKET")
            print(f"  quantity: {adjusted_quantity}")
            print(f"  triggerPrice: {adjusted_stop_price}")
            print(f"  workingType: CONTRACT_PRICE")
            
            # 构建算法订单参数
            timestamp = int(time.time() * 1000) + self.timestamp_offset
            params = {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "type": "STOP_MARKET",
                "quantity": str(adjusted_quantity),
                "triggerPrice": str(adjusted_stop_price),
                "workingType": "CONTRACT_PRICE",
                "timeInForce": "GTC",
                "recvWindow": 10000,
                "timestamp": timestamp
            }
            
            try:
                response = self.client._request_futures_api('post', 'algoOrder', True, data=params)
                print(f"✓ 止损订单（算法）测试成功！")
                return response
            except Exception as e:
                print(f"✗ 止损订单（算法）测试失败: {e}")
                return None
                
        except Exception as e:
            print(f"测试止损订单错误: {e}")
            return None

    def test_take_profit_order(self, symbol, side, quantity, stop_price):
        """测试止盈订单（使用算法订单API）"""
        try:
            # 调整精度
            adjusted_quantity, adjusted_stop_price = self._adjust_quantity_and_price(
                symbol, quantity, stop_price
            )
            
            # 使用positionSide="BOTH"（适用于双向持仓模式）
            position_side = "BOTH"
            
            print(f"测试止盈订单参数（算法订单）:")
            print(f"  symbol: {symbol}")
            print(f"  side: {side}")
            print(f"  positionSide: {position_side}")
            print(f"  algoType: CONDITIONAL")
            print(f"  type: TAKE_PROFIT_MARKET")
            print(f"  quantity: {adjusted_quantity}")
            print(f"  triggerPrice: {adjusted_stop_price}")
            print(f"  workingType: CONTRACT_PRICE")
            
            # 构建算法订单参数
            timestamp = int(time.time() * 1000) + self.timestamp_offset
            params = {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": str(adjusted_quantity),
                "triggerPrice": str(adjusted_stop_price),
                "workingType": "CONTRACT_PRICE",
                "timeInForce": "GTC",
                "recvWindow": 10000,
                "timestamp": timestamp
            }
            
            try:
                response = self.client._request_futures_api('post', 'algoOrder', True, data=params)
                print(f"✓ 止盈订单（算法）测试成功！")
                return response
            except Exception as e:
                print(f"✗ 止盈订单（算法）测试失败: {e}")
                return None
                
        except Exception as e:
            print(f"测试止盈订单错误: {e}")
            return None

    def cancel_all_orders(self, symbol):
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            print(f"已取消所有订单: {result}")
            return result
        except Exception as e:
            print(f"取消订单错误: {e}")
            return None

    def get_open_orders(self, symbol=None):
        try:
            if symbol:
                orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders = self.client.futures_get_open_orders()
            return orders
        except Exception as e:
            print(f"获取挂单错误: {e}")
            return []

    def place_oco_order(self, symbol, side, quantity, stop_price, limit_price):
        try:
            # 调整精度
            adjusted_quantity, adjusted_stop_price = self._adjust_quantity_and_price(
                symbol, quantity, stop_price
            )
            # 调整限价精度
            _, adjusted_limit_price = self._adjust_quantity_and_price(
                symbol, quantity, limit_price
            )
            order = self.client.futures_create_oco_order(
                symbol=symbol,
                side=side,
                quantity=adjusted_quantity,
                stopPrice=adjusted_stop_price,
                stopLimitPrice=adjusted_stop_price,
                stopLimitTimeInForce="GTC",
                limitClientOrderId=f"kronos_tp_{int(datetime.now().timestamp())}",
                stopClientOrderId=f"kronos_sl_{int(datetime.now().timestamp())}",
            )
            print(f"OCO订单已提交: {order}")
            return order
        except Exception as e:
            print(f"OCO订单错误: {e}")
            return None

    def place_algo_stop_loss(self, symbol, side, quantity, trigger_price):
        try:
            # 调整精度
            adjusted_quantity, adjusted_trigger_price = self._adjust_quantity_and_price(
                symbol, quantity, trigger_price
            )
            
            # 使用positionSide="BOTH"（适用于双向持仓模式）
            position_side = "BOTH"
            
            print(f"创建算法止损订单: {symbol}, {side}, 数量={adjusted_quantity}, 触发价={adjusted_trigger_price}")
            print(f"  positionSide={position_side}")
            
            # 直接使用python-binance库的方法
            # 先尝试查找是否有官方算法订单方法
            try:
                # 检查是否有 futures_create_algo_order 或类似方法
                if hasattr(self.client, 'futures_create_algo_order'):
                    order = self.client.futures_create_algo_order(
                        algoType="CONDITIONAL",
                        symbol=symbol,
                        side=side,
                        positionSide=position_side,
                        type="STOP_MARKET",
                        quantity=adjusted_quantity,
                        triggerPrice=adjusted_trigger_price,
                        workingType="CONTRACT_PRICE",
                        timeInForce="GTC",
                        recvWindow=10000,
                    )
                    print(f"Algo止损单已提交: {symbol}, {side}, 数量={quantity}, 触发价={adjusted_trigger_price}")
                    return order
            except Exception as e:
                print(f"官方算法订单方法失败: {e}")
            
            # 如果官方方法不可用，使用手动API调用
            print("使用手动API调用...")
            timestamp = int(time.time() * 1000) + self.timestamp_offset
            
            # 构建参数
            params = {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "type": "STOP_MARKET",
                "quantity": str(adjusted_quantity),
                "triggerPrice": str(adjusted_trigger_price),
                "workingType": "CONTRACT_PRICE",
                "timeInForce": "GTC",
                "recvWindow": 10000,
                "timestamp": timestamp
            }
            
            # 使用client的内部请求方法
            try:
                response = self.client._request_futures_api('post', 'algoOrder', True, data=params)
                print(f"Algo止损单已提交: {symbol}, {side}, 数量={quantity}, 触发价={adjusted_trigger_price}")
                return response
            except Exception as e:
                print(f"_request_futures_api错误: {e}")
                return None
                
        except Exception as e:
            print(f"Algo止损单错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def place_algo_take_profit(self, symbol, side, quantity, trigger_price):
        try:
            # 调整精度
            adjusted_quantity, adjusted_trigger_price = self._adjust_quantity_and_price(
                symbol, quantity, trigger_price
            )
            
            # 使用positionSide="BOTH"（适用于双向持仓模式）
            position_side = "BOTH"
            
            print(f"创建算法止盈订单: {symbol}, {side}, 数量={adjusted_quantity}, 触发价={adjusted_trigger_price}")
            print(f"  positionSide={position_side}")
            
            # 直接使用python-binance库的方法
            # 先尝试查找是否有官方算法订单方法
            try:
                # 检查是否有 futures_create_algo_order 或类似方法
                if hasattr(self.client, 'futures_create_algo_order'):
                    order = self.client.futures_create_algo_order(
                        algoType="CONDITIONAL",
                        symbol=symbol,
                        side=side,
                        positionSide=position_side,
                        type="TAKE_PROFIT_MARKET",
                        quantity=adjusted_quantity,
                        triggerPrice=adjusted_trigger_price,
                        workingType="CONTRACT_PRICE",
                        timeInForce="GTC",
                        recvWindow=10000,
                    )
                    print(f"Algo止盈单已提交: {symbol}, {side}, 数量={quantity}, 触发价={adjusted_trigger_price}")
                    return order
            except Exception as e:
                print(f"官方算法订单方法失败: {e}")
            
            # 如果官方方法不可用，使用手动API调用
            print("使用手动API调用...")
            timestamp = int(time.time() * 1000) + self.timestamp_offset
            
            # 构建参数
            params = {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": str(adjusted_quantity),
                "triggerPrice": str(adjusted_trigger_price),
                "workingType": "CONTRACT_PRICE",
                "timeInForce": "GTC",
                "recvWindow": 10000,
                "timestamp": timestamp
            }
            
            # 使用client的内部请求方法
            try:
                response = self.client._request_futures_api('post', 'algoOrder', True, data=params)
                print(f"Algo止盈单已提交: {symbol}, {side}, 数量={quantity}, 触发价={adjusted_trigger_price}")
                return response
            except Exception as e:
                print(f"_request_futures_api错误: {e}")
                return None
                
        except Exception as e:
            print(f"Algo止盈单错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cancel_algo_order(self, symbol, algo_id):
        try:
            timestamp = int(time.time() * 1000) + self.timestamp_offset

            params = {"symbol": symbol, "algoId": algo_id, "recvWindow": 10000, "timestamp": timestamp}

            # 使用Client的_request_futures_api方法（自动处理签名和时间戳）
            try:
                response = self.client._request_futures_api('delete', 'algoOrder', True, data=params)
                # _request_futures_api返回已解析的JSON响应
                print(f"Algo订单已取消: {algo_id}")
                return response
            except Exception as api_error:
                # 如果_request_futures_api失败，回退到手动请求
                print(f"_request_futures_api错误: {api_error}")
                # 使用手动签名（原有代码）
                import urllib.parse
                query_parts = []
                for key, value in sorted(params.items()):
                    str_value = str(value)
                    encoded_value = urllib.parse.quote(str_value, safe='.')
                    query_parts.append(f"{key}={encoded_value}")
                query_string = "&".join(query_parts)
                
                import hmac
                import hashlib
                signature = hmac.new(
                    self.api_secret.encode(), query_string.encode(), hashlib.sha256
                ).hexdigest()
                params["signature"] = signature

                url = "https://fapi.binance.com/fapi/v1/algoOrder"
                headers = {"X-MBX-APIKEY": self.api_key}

                response = requests.delete(url, headers=headers, data=params, timeout=10)
                if response.status_code == 200:
                    print(f"Algo订单已取消: {algo_id}")
                    return response.json()
                else:
                    print(f"取消Algo订单错误: {response.text}")
                    return None
        except Exception as e:
            print(f"取消Algo订单错误: {e}")
            return None

    def cancel_all_algo_orders(self, symbol):
        try:
            timestamp = int(time.time() * 1000) + self.timestamp_offset

            params = {"symbol": symbol, "recvWindow": 10000, "timestamp": timestamp}

            # 使用Client的_request_futures_api方法（自动处理签名和时间戳）
            try:
                response = self.client._request_futures_api('delete', 'algoOpenOrders', True, data=params)
                # _request_futures_api返回已解析的JSON响应
                print(f"已取消所有Algo订单")
                return response
            except Exception as api_error:
                # 如果_request_futures_api失败，回退到手动请求
                print(f"_request_futures_api错误: {api_error}")
                # 使用手动签名（原有代码）
                import urllib.parse
                query_parts = []
                for key, value in sorted(params.items()):
                    str_value = str(value)
                    encoded_value = urllib.parse.quote(str_value, safe='.')
                    query_parts.append(f"{key}={encoded_value}")
                query_string = "&".join(query_parts)
                
                import hmac
                import hashlib
                signature = hmac.new(
                    self.api_secret.encode(), query_string.encode(), hashlib.sha256
                ).hexdigest()
                params["signature"] = signature

                url = "https://fapi.binance.com/fapi/v1/algoOpenOrders"
                headers = {"X-MBX-APIKEY": self.api_key}

                response = requests.delete(url, headers=headers, data=params, timeout=10)
                if response.status_code == 200:
                    print(f"已取消所有Algo订单")
                    return response.json()
                else:
                    print(f"取消所有Algo订单错误: {response.text}")
                    return None
        except Exception as e:
            print(f"取消所有Algo订单错误: {e}")
            return None
