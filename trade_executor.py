#!/usr/bin/env python
"""交易执行器 - 执行策略适配器生成的交易指令，仅支持币安交易所"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 导入现有模块
try:
    from binance_api import BinanceAPI
except ImportError as e:
    print(f"币安API模块导入失败: {e}")
    BinanceAPI = None


class TradeExecutor:
    """交易执行器 - 仅支持币安交易所"""
    
    def __init__(self, 
                 testnet: bool = True,
                 symbol: str = "BTCUSDT"):
        """
        初始化交易执行器
        
        Args:
            testnet: 是否使用测试网络
            symbol: 交易品种
        """
        self.testnet = testnet
        self.symbol = symbol
        
        # 交易所客户端
        self.binance_client = None
        
        # 交易参数
        self.leverage = 3  # 默认3倍杠杆
        self.position_mode = "HEDGE"  # 持仓模式: HEDGE(双向) or ONE_WAY(单向)
        
        # 执行状态
        self.active_orders = []
        self.position_info = None
        self.account_balance = 0.0
        self.last_execution_time = None
        self.execution_history = []
        
        print(f"初始化交易执行器: Binance (测试网: {testnet})")
        self._initialize_binance()
    
    def _initialize_binance(self):
        """初始化币安客户端"""
        if BinanceAPI is None:
            print("  ✗ 币安API模块不可用")
            return
        
        try:
            # 注意: BinanceAPI类已经处理了测试网配置
            self.binance_client = BinanceAPI()
            
            # 设置交易参数
            self._set_binance_trading_params()
            
            # 获取账户信息
            self._update_account_info()
            
            print("  ✓ 币安客户端初始化成功")
            
        except Exception as e:
            print(f"  ✗ 币安客户端初始化失败: {e}")
    
    def _set_binance_trading_params(self):
        """设置币安交易参数"""
        try:
            # 设置杠杆
            self.binance_client.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )
            print(f"    杠杆设置: {self.leverage}x")
            
            # 设置持仓模式
            if self.position_mode == "HEDGE":
                self.binance_client.client.futures_change_position_mode(dualSidePosition=True)
            else:
                self.binance_client.client.futures_change_position_mode(dualSidePosition=False)
            print(f"    持仓模式: {self.position_mode}")
            
        except Exception as e:
            print(f"    交易参数设置失败: {e}")
    
    def _update_account_info(self):
        """更新账户信息"""
        try:
            # 获取账户余额
            balance_info = self.binance_client.get_account_balance()
            if balance_info:
                for asset in balance_info:
                    if asset['asset'] == 'USDT':
                        self.account_balance = float(asset['availableBalance'])
                        print(f"    账户余额: ${self.account_balance:.2f} USDT")
                        break
            
            # 获取持仓信息
            positions = self.binance_client.client.futures_position_information(
                symbol=self.symbol
            )
            if positions and len(positions) > 0:
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        self.position_info = {
                            'symbol': pos['symbol'],
                            'position_amt': float(pos['positionAmt']),
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_profit': float(pos['unRealizedProfit']),
                            'position_side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT'
                        }
                        print(f"    当前持仓: {self.position_info['position_side']} "
                              f"{abs(self.position_info['position_amt']):.4f} {self.symbol}")
                        break
            
        except Exception as e:
            print(f"    账户信息更新失败: {e}")
    
    def execute_trade(self, instruction: Dict) -> Dict:
        """执行交易指令
        
        Args:
            instruction: 交易指令字典
            
        Returns:
            执行结果字典
        """
        if not instruction:
            return {"status": "ERROR", "message": "无效的交易指令"}
        
        # 提取指令信息
        symbol = instruction.get("symbol", self.symbol)
        side = instruction.get("side", "BUY")
        quantity = instruction.get("quantity", 0.0)
        order_type = instruction.get("type", "MARKET")
        
        # 检查指令有效性
        if quantity <= 0:
            return {"status": "ERROR", "message": "交易数量必须大于0"}
        
        # 执行币安交易
        if self.binance_client:
            execution_result = self._execute_binance_trade(instruction)
        else:
            execution_result = {
                "status": "ERROR", 
                "message": "币安客户端未初始化"
            }
        
        # 记录执行历史
        if execution_result.get("status") == "SUCCESS":
            self._record_execution_history(instruction, execution_result)
            self.last_execution_time = datetime.now()
            
            # 更新账户信息
            self._update_account_info()
        
        return execution_result
    
    def _execute_binance_trade(self, instruction: Dict) -> Dict:
        """执行币安交易
        
        Args:
            instruction: 交易指令
            
        Returns:
            执行结果
        """
        try:
            symbol = instruction.get("symbol", self.symbol)
            side = instruction.get("side", "BUY")
            quantity = instruction.get("quantity", 0.0)
            order_type = instruction.get("type", "MARKET")
            
            # 计算交易数量 (转换为币的数量)
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return {"status": "ERROR", "message": "无法获取当前价格"}
            
            # 计算币的数量 (保留6位小数)
            coin_quantity = quantity / current_price
            coin_quantity = round(coin_quantity, 6)
            
            # 执行市价单
            if order_type == "MARKET":
                if side == "BUY":
                    order_result = self.binance_client.client.futures_create_order(
                        symbol=symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=coin_quantity
                    )
                else:  # SELL
                    order_result = self.binance_client.client.futures_create_order(
                        symbol=symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=coin_quantity
                    )
                
                # 解析订单结果
                execution_price = float(order_result.get('avgPrice', current_price))
                executed_qty = float(order_result.get('executedQty', coin_quantity))
                
                result = {
                    "status": "SUCCESS",
                    "message": "交易执行成功",
                    "exchange": "binance",
                    "order_id": order_result.get('orderId'),
                    "symbol": symbol,
                    "side": side,
                    "quantity": executed_qty,
                    "price": execution_price,
                    "order_type": order_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"  ✓ 币安交易执行成功: {side} {executed_qty:.6f} {symbol} @ ${execution_price:.2f}")
                
                return result
            
            else:
                return {"status": "ERROR", "message": f"不支持的订单类型: {order_type}"}
                
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ 币安交易执行失败: {error_msg}")
            return {
                "status": "ERROR",
                "message": f"交易执行失败: {error_msg}",
                "exchange": "binance",
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格
        
        Args:
            symbol: 交易品种
            
        Returns:
            当前价格
        """
        try:
            if self.binance_client:
                # 从币安获取最新价格
                ticker = self.binance_client.client.futures_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            
            # 默认返回0
            return 0.0
            
        except Exception as e:
            print(f"  获取价格失败: {e}")
            return 0.0
    
    def _record_execution_history(self, instruction: Dict, execution_result: Dict):
        """记录执行历史"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "exchange": "binance",
            "instruction": instruction.copy(),
            "execution_result": execution_result.copy(),
            "account_balance_before": self.account_balance
        }
        
        # 移除可能的安全敏感信息
        if 'api_key' in history_entry['instruction']:
            del history_entry['instruction']['api_key']
        if 'api_secret' in history_entry['instruction']:
            del history_entry['instruction']['api_secret']
        
        self.execution_history.append(history_entry)
        
        # 保留最近100条记录
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def place_stop_loss_order(self, symbol: str, stop_price: float, quantity: float) -> Dict:
        """设置止损订单
        
        Args:
            symbol: 交易品种
            stop_price: 止损价格
            quantity: 数量
            
        Returns:
            订单结果
        """
        if self.binance_client:
            return self._place_binance_stop_loss(symbol, stop_price, quantity)
        else:
            return {"status": "ERROR", "message": "交易所客户端未初始化"}
    
    def _place_binance_stop_loss(self, symbol: str, stop_price: float, quantity: float) -> Dict:
        """设置币安止损订单"""
        try:
            # 判断当前持仓方向
            if self.position_info and self.position_info['position_amt'] > 0:
                # 多仓止损 -> 市价卖出
                side = "SELL"
            else:
                # 空仓止损 -> 市价买入
                side = "BUY"
            
            # 设置止损市价单
            order_result = self.binance_client.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=abs(quantity),
                stopPrice=stop_price
            )
            
            # 添加到活跃订单列表
            self.active_orders.append({
                'order_id': order_result['orderId'],
                'symbol': symbol,
                'type': 'STOP_LOSS',
                'price': stop_price,
                'quantity': quantity
            })
            
            return {
                "status": "SUCCESS",
                "message": "止损订单设置成功",
                "order_id": order_result['orderId']
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """取消所有订单
        
        Args:
            symbol: 交易品种 (如果为None则取消所有品种)
            
        Returns:
            取消结果
        """
        if self.binance_client:
            return self._cancel_binance_orders(symbol)
        else:
            return {"status": "ERROR", "message": "交易所客户端未初始化"}
    
    def _cancel_binance_orders(self, symbol: str = None) -> Dict:
        """取消币安订单"""
        try:
            if symbol:
                # 取消指定品种的订单
                result = self.binance_client.client.futures_cancel_all_open_orders(symbol=symbol)
            else:
                # 取消所有订单
                result = self.binance_client.client.futures_cancel_all_open_orders()
            
            # 清空活跃订单列表
            self.active_orders = []
            
            return {
                "status": "SUCCESS",
                "message": "订单取消成功",
                "result": result
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history 
                                     if e['execution_result'].get('status') == 'SUCCESS'])
        failed_executions = total_executions - successful_executions
        
        # 计算总交易量
        total_volume = 0.0
        for execution in self.execution_history:
            if execution['execution_result'].get('status') == 'SUCCESS':
                quantity = execution['execution_result'].get('quantity', 0)
                price = execution['execution_result'].get('price', 0)
                total_volume += quantity * price
        
        return {
            "exchange": "binance",
            "testnet": self.testnet,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "total_volume": total_volume,
            "account_balance": self.account_balance,
            "has_position": self.position_info is not None,
            "active_orders_count": len(self.active_orders),
            "last_execution_time": self.last_execution_time
        }


# 使用示例
if __name__ == "__main__":
    print("=== 交易执行器测试 ===")
    
    # 测试币安执行器
    print("\n1. 测试币安执行器...")
    try:
        binance_executor = TradeExecutor(
            testnet=True,
            symbol="BTCUSDT"
        )
        
        # 获取执行摘要
        summary = binance_executor.get_execution_summary()
        print(f"  交易所: {summary['exchange']}")
        print(f"  测试网: {summary['testnet']}")
        print(f"  账户余额: ${summary['account_balance']:.2f}")
        
        # 模拟交易指令
        test_instruction = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 100.0,  # 100 USDT
            "confidence": 0.75,
            "reasoning": ["测试交易"]
        }
        
        print("\n  模拟交易指令:")
        print(f"    方向: {test_instruction['side']}")
        print(f"    金额: ${test_instruction['quantity']:.2f}")
        
        # 注意: 实际执行需要有效的API密钥
        # 这里只演示流程，不实际执行
        print("\n  ⚠ 注意: 需要有效的API密钥才能实际执行交易")
        
    except Exception as e:
        print(f"  币安执行器测试失败: {e}")
    
    print("\n交易执行器测试完成！")
