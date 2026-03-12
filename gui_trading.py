import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import os
import sys
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import psutil
import yaml
import json
import pynvml
import matplotlib
import pandas as pd
import numpy as np

# 配置日志 - 兼容 PyInstaller 打包环境（无终端模式）
log_file = None
try:
    # 如果是打包后的无终端模式，创建日志文件
    if getattr(sys, 'frozen', False):
        import tempfile
        log_dir = os.path.join(os.path.dirname(sys.executable), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'kronos_trading.log')
    
    # 配置日志
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a',
            encoding='utf-8'
        )
    else:
        # 开发模式，输出到控制台
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
except Exception:
    # 备用方案：不配置日志，避免崩溃
    pass

# 禁用 transformers 的详细日志
try:
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.ERROR)
except Exception:
    pass

matplotlib.use("TkAgg")  # 设置后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 新增模块导入 - 多智能体量化交易系统
try:
    from fingpt_analyzer import FinGPTSentimentAnalyzer
except ImportError as e:
    print(f"FinGPT模块导入失败: {e}")
    FinGPTSentimentAnalyzer = None

try:
    from strategy_coordinator import StrategyCoordinator
except ImportError as e:
    print(f"策略协调器模块导入失败: {e}")
    StrategyCoordinator = None

# 新增模块导入 - BTC新闻爬虫
try:
    from btc_news_crawler import BTCNewsCrawler
except ImportError as e:
    print(f"BTC新闻爬虫模块导入失败: {e}")
    BTCNewsCrawler = None

# 新增模块导入 - 自动化优化系统
try:
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"性能监控模块导入失败: {e}")
    PerformanceMonitor = None

try:
    from auto_optimization_pipeline import AutoOptimizationPipeline
except ImportError as e:
    print(f"自动化优化管道模块导入失败: {e}")
    AutoOptimizationPipeline = None

try:
    from parameter_integrator import ParameterIntegrator
except ImportError as e:
    print(f"参数集成器模块导入失败: {e}")
    ParameterIntegrator = None

# 配置matplotlib中文字体（最可靠的方法）
try:
    plt.rcParams["font.sans-serif"] = [
        "微软雅黑",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Arial Unicode MS",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    # print("已配置matplotlib中文字体支持")
except Exception as e:
    print(f"配置中文字体时出错: {e}")

os.environ["HF_HUB_DISABLE_XET"] = "1"

# 尝试加载.env文件以获取HF_TOKEN
try:
    from dotenv import load_dotenv
    
    # 加载.env文件
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # 如果.env文件中设置了HF_TOKEN，则设置环境变量
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.strip() and not hf_token.startswith("your_"):
        os.environ["HF_TOKEN"] = hf_token
        print(f"已设置HF_TOKEN (长度: {len(hf_token)})")
    else:
        print("提示: 如需更快的模型下载速度，请在.env文件中设置有效的HF_TOKEN")
        print("获取地址: https://huggingface.co/settings/tokens")
except ImportError:
    print("dotenv未安装，跳过.env文件加载")
except Exception as e:
    print(f"加载HF_TOKEN时出错: {e}")

LARGE_FONT = ("微软雅黑", 12)
BOLD_FONT = ("微软雅黑", 12, "bold")
TITLE_FONT = ("微软雅黑", 14, "bold")


class OutputRedirector:
    def __init__(self, queue, progress_callback=None):
        self.queue = queue
        self.progress_callback = progress_callback
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]
            for line in lines[:-1]:
                if not line.strip():
                    continue

                # 进度检测逻辑
                progress_detected = False

                # 1. 检测 "100%" 模式
                if "100%" in line:
                    if self.progress_callback:
                        self.progress_callback(100)
                        # self.queue.put(("log", f"[进度] 100% 完成"))
                    progress_detected = True

                # 2. 检测 "it/s]" 模式 (tqdm进度条)
                elif "it/s]" in line:
                    try:
                        import re

                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if self.progress_callback:
                                self.progress_callback(pct)
                                # self.queue.put(("log", f"[进度] {pct}%"))
                            progress_detected = True
                    except:
                        pass

                # 3. 检测简单百分比模式 (例如: "进度: 50%")
                elif "%" in line and not progress_detected:
                    try:
                        import re

                        # 查找任何数字后跟%的模式
                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if 0 <= pct <= 100:
                                if self.progress_callback:
                                    self.progress_callback(pct)
                                    # self.queue.put(("log", f"[进度] 检测到百分比: {pct}%"))
                                progress_detected = True
                    except:
                        pass

                # 4. 检测进度条可视化模式 (例如: "[====>    ] 50%")
                elif (
                    "[" in line and "]" in line and "%" in line
                ) and not progress_detected:
                    try:
                        import re

                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if 0 <= pct <= 100:
                                if self.progress_callback:
                                    self.progress_callback(pct)
                                    # self.queue.put(("log", f"[进度] 检测到百分比: {pct}%"))
                                progress_detected = True
                    except:
                        pass

                # 如果不是进度信息，则作为普通日志输出
                if not progress_detected:
                    self.queue.put(("log", line))

    def flush(self):
        pass

    def isatty(self):
        return False  # OutputRedirector不是终端设备
    
    def fileno(self):
        # 返回一个有效的文件描述符，通常为1(stdout)或2(stderr)
        return 1
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False
    
    def close(self):
        pass
    
    @property
    def closed(self):
        return False
    
    @property
    def mode(self):
        return 'w'
    
    @property
    def name(self):
        return '<OutputRedirector>'
    
    @property
    def encoding(self):
        return 'utf-8'
    
    @property
    def newlines(self):
        return None
    
    def tell(self):
        return 0
    
    def truncate(self, size=None):
        pass
    
    def read(self, size=-1):
        raise io.UnsupportedOperation('read')
    
    def readline(self, size=-1):
        raise io.UnsupportedOperation('read')
    
    def readlines(self, hint=-1):
        raise io.UnsupportedOperation('read')
    
    def seek(self, offset, whence=0):
        raise io.UnsupportedOperation('seek')
    
    def write(self, text):
        # 这是原有的write方法，需要保持
        self.buffer += text
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]
            for line in lines[:-1]:
                if not line.strip():
                    continue

                # 进度检测逻辑
                progress_detected = False

                # 1. 检测 "100%" 模式
                if "100%" in line:
                    if self.progress_callback:
                        self.progress_callback(100)
                    progress_detected = True

                # 2. 检测 "it/s]" 模式 (tqdm进度条)
                elif "it/s]" in line:
                    try:
                        import re
                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if self.progress_callback:
                                self.progress_callback(pct)
                            progress_detected = True
                    except:
                        pass

                # 3. 检测简单百分比模式
                elif "%" in line and not progress_detected:
                    try:
                        import re
                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if 0 <= pct <= 100:
                                if self.progress_callback:
                                    self.progress_callback(pct)
                                progress_detected = True
                    except:
                        pass

                # 4. 检测进度条可视化模式
                elif ("[" in line and "]" in line and "%" in line) and not progress_detected:
                    try:
                        import re
                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if 0 <= pct <= 100:
                                if self.progress_callback:
                                    self.progress_callback(pct)
                                progress_detected = True
                    except:
                        pass

                # 如果不是进度信息，则作为普通日志输出
                if not progress_detected:
                    self.queue.put(("log", line))


class StrategyConfigDialog:
    """策略参数配置对话框"""
    
    def __init__(self, parent, current_config=None):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("⚙️ 策略参数配置")
        self.dialog.geometry("1200x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.result = None
        self.current_config = current_config if current_config else self._get_default_config()
        
        # 创建主框架
        self._create_main_frame()
        self._load_config_to_ui()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "coordinator": {
                "min_signal_strength": 0.15,
                "max_position_size": 0.1,
                "sentiment_weight": 0.3,
                "technical_weight": 0.7,
                "black_swan_threshold": "HIGH",
                "enable_adaptive_filtering": True
            },
            "basic": {
                "POSITION_MULTIPLIER": 1,
                "TREND_STRENGTH_THRESHOLD": 0.0047,
                "LOOKBACK_PERIOD": 91,
                "PREDICTION_LENGTH": 90,
                "CHECK_INTERVAL": 180
            },
            "entry": {
                "max_kline_change": 0.015,
                "max_funding_rate_long": 0.03,
                "min_funding_rate_short": -0.03,
                "support_buffer": 1.001,
                "resistance_buffer": 0.999
            },
            "stop_loss": {
                "long_buffer": 0.996,
                "short_buffer": 1.004
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
                "tp3_position_ratio": 0.30
            },
            "risk": {
                "single_trade_risk": 0.029,
                "daily_loss_limit": 0.12,
                "max_consecutive_losses": 6,
                "max_single_position": 0.29,
                "max_daily_position": 0.85
            },
            "frequency": {
                "max_daily_trades": 55,
                "min_trade_interval_minutes": 3,
                "active_hours_start": 0,
                "active_hours_end": 24
            },
            "position": {
                "initial_entry_ratio": 0.35,
                "confirm_interval_kline": 2,
                "add_on_profit": True,
                "add_ratio": 0.25,
                "max_add_times": 3
            },
            "strategy": {
                "entry_confirm_count": 3,
                "reverse_confirm_count": 2,
                "require_consecutive_prediction": 2,
                "post_entry_hours": 6,
                "take_profit_min_pct": 0.5
            }
        }
    
    def _create_main_frame(self):
        """创建主框架"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部按钮栏
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 预设选择
        ttk.Label(button_frame, text="📋 交易风格预设:").pack(side=tk.LEFT, padx=(0, 5))
        self.preset_var = tk.StringVar(value="平衡型")
        preset_combo = ttk.Combobox(
            button_frame, 
            textvariable=self.preset_var,
            values=["激进超短线", "趋势追踪", "平衡型", "震荡套利", "稳健长线", "消息驱动"],
            state="readonly",
            width=15
        )
        preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="📁 载入配置", command=self._load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="💾 保存配置", command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="↩️ 重置默认", command=self._reset_default).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="✅ 应用", command=self._apply_config, style="Accent.TButton").pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="❌ 取消", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 5))
        
        # 创建Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个标签页
        self.config_vars = {}
        self._create_coordinator_tab()
        self._create_basic_tab()
        self._create_entry_tab()
        self._create_stop_loss_tab()
        self._create_take_profit_tab()
        self._create_risk_tab()
        self._create_frequency_tab()
        self._create_position_tab()
    
    def _create_param_row(self, parent, category, param_name, display_name, param_type, min_val=None, max_val=None, description=""):
        """创建参数行"""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=2)
        
        # 参数名称
        name_label = ttk.Label(row_frame, text=display_name, width=25, anchor=tk.W)
        name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # 输入控件
        var_key = f"{category}.{param_name}"
        
        if param_type == "float":
            var = tk.DoubleVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        elif param_type == "int":
            var = tk.IntVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        elif param_type == "bool":
            var = tk.BooleanVar()
            entry = ttk.Checkbutton(row_frame, variable=var)
        elif param_type == "select":
            var = tk.StringVar()
            entry = ttk.Combobox(row_frame, textvariable=var, values=["LOW", "MEDIUM", "HIGH"], state="readonly", width=10)
        else:
            var = tk.StringVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        
        entry.pack(side=tk.LEFT, padx=(0, 5))
        self.config_vars[var_key] = var
        
        # 取值范围
        if min_val is not None and max_val is not None:
            range_label = ttk.Label(row_frame, text=f"[{min_val}-{max_val}]", width=15, foreground="#7f8c8d")
            range_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # 说明
        desc_label = ttk.Label(row_frame, text=description, foreground="#7f8c8d", wraplength=500)
        desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _create_coordinator_tab(self):
        """创建协调器参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="1. 协调器参数")
        
        ttk.Label(frame, text="【协调器参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "coordinator", "min_signal_strength", "最小信号强度", "float", 0.0, 1.0, "只有信号强度超过此值才触发交易")
        self._create_param_row(frame, "coordinator", "max_position_size", "最大仓位比例", "float", 0.0, 1.0, "单次交易的最大仓位比例")
        self._create_param_row(frame, "coordinator", "sentiment_weight", "舆情信号权重", "float", 0.0, 1.0, "FinGPT舆情分析的权重")
        self._create_param_row(frame, "coordinator", "technical_weight", "技术信号权重", "float", 0.0, 1.0, "Kronos技术分析的权重")
        self._create_param_row(frame, "coordinator", "black_swan_threshold", "黑天鹅阈值", "select", None, None, "极端行情敏感度阈值")
        self._create_param_row(frame, "coordinator", "enable_adaptive_filtering", "自适应过滤", "bool", None, None, "启用动态参数调整")
    
    def _create_basic_tab(self):
        """创建基础参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="2. 基础参数")
        
        ttk.Label(frame, text="【基础参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "basic", "LEVERAGE", "杠杆倍数", "int", 1, 125, "交易使用的杠杆大小")
        self._create_param_row(frame, "basic", "TREND_STRENGTH_THRESHOLD", "趋势强度阈值", "float", 0.001, 0.05, "判断趋势有效的阈值")
        self._create_param_row(frame, "basic", "LOOKBACK_PERIOD", "回看K线数量", "int", 64, 2048, "技术分析使用的历史数据长度")
        self._create_param_row(frame, "basic", "PREDICTION_LENGTH", "预测K线数量", "int", 4, 200, "Kronos预测的未来K线数量")
        self._create_param_row(frame, "basic", "CHECK_INTERVAL", "检查间隔(秒)", "int", 30, 3600, "系统检查交易信号的间隔")
    
    def _create_entry_tab(self):
        """创建入场过滤参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="3. 入场过滤")
        
        ttk.Label(frame, text="【入场过滤参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "entry", "max_kline_change", "最大单K变化", "float", 0.001, 0.05, "限制单根K线的最大涨跌幅")
        self._create_param_row(frame, "entry", "max_funding_rate_long", "多头最大资金费率", "float", -0.05, 0.05, "开多仓时资金费率上限")
        self._create_param_row(frame, "entry", "min_funding_rate_short", "空头最小资金费率", "float", -0.05, 0.05, "开空仓时资金费率下限")
        self._create_param_row(frame, "entry", "support_buffer", "支撑位缓冲", "float", 1.000, 1.010, "在支撑位上方多少比例入场")
        self._create_param_row(frame, "entry", "resistance_buffer", "阻力位缓冲", "float", 0.990, 1.000, "在阻力位下方多少比例入场")
    
    def _create_stop_loss_tab(self):
        """创建止损参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="4. 止损参数")
        
        ttk.Label(frame, text="【止损参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "stop_loss", "long_buffer", "多头止损缓冲", "float", 0.900, 0.999, "多头止损相对于入场价的比例")
        self._create_param_row(frame, "stop_loss", "short_buffer", "空头止损缓冲", "float", 1.001, 1.100, "空头止损相对于入场价的比例")
    
    def _create_take_profit_tab(self):
        """创建止盈参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="5. 止盈参数")
        
        ttk.Label(frame, text="【止盈参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "take_profit", "tp1_multiplier_long", "多头第一止盈", "float", 1.001, 1.100, "多头第一止盈目标")
        self._create_param_row(frame, "take_profit", "tp2_multiplier_long", "多头第二止盈", "float", 1.001, 1.200, "多头第二止盈目标")
        self._create_param_row(frame, "take_profit", "tp3_multiplier_long", "多头第三止盈", "float", 1.001, 1.300, "多头第三止盈目标")
        self._create_param_row(frame, "take_profit", "tp1_multiplier_short", "空头第一止盈", "float", 0.900, 0.999, "空头第一止盈目标")
        self._create_param_row(frame, "take_profit", "tp2_multiplier_short", "空头第二止盈", "float", 0.800, 0.999, "空头第二止盈目标")
        self._create_param_row(frame, "take_profit", "tp3_multiplier_short", "空头第三止盈", "float", 0.700, 0.999, "空头第三止盈目标")
        self._create_param_row(frame, "take_profit", "tp1_position_ratio", "第一止盈仓位", "float", 0.1, 1.0, "达到第一止盈时平掉的仓位比例")
        self._create_param_row(frame, "take_profit", "tp2_position_ratio", "第二止盈仓位", "float", 0.1, 1.0, "达到第二止盈时平掉的仓位比例")
        self._create_param_row(frame, "take_profit", "tp3_position_ratio", "第三止盈仓位", "float", 0.1, 1.0, "达到第三止盈时平掉的仓位比例")
    
    def _create_risk_tab(self):
        """创建风险管理参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="6. 风险管理")
        
        ttk.Label(frame, text="【风险管理参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "risk", "single_trade_risk", "单笔风险比例", "float", 0.001, 0.10, "单笔交易最大亏损比例")
        self._create_param_row(frame, "risk", "daily_loss_limit", "每日亏损限制", "float", 0.01, 0.20, "单日累计亏损达到此值停止交易")
        self._create_param_row(frame, "risk", "max_consecutive_losses", "最大连续亏损", "int", 1, 10, "连续亏损次数达到此值暂停交易")
        self._create_param_row(frame, "risk", "max_single_position", "最大单笔仓位", "float", 0.01, 1.0, "单笔交易的最大仓位比例")
        self._create_param_row(frame, "risk", "max_daily_position", "最大日仓位", "float", 0.01, 1.0, "单日累计开仓的最大仓位比例")
    
    def _create_frequency_tab(self):
        """创建交易频率参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="7. 交易频率")
        
        ttk.Label(frame, text="【交易频率参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "frequency", "max_daily_trades", "每日最大交易", "int", 1, 100, "限制单日最多交易多少次")
        self._create_param_row(frame, "frequency", "min_trade_interval_minutes", "最小间隔(分)", "int", 1, 60, "两次交易之间的最小间隔")
        self._create_param_row(frame, "frequency", "active_hours_start", "活跃开始时间", "int", 0, 23, "只在此时间之后进行交易")
        self._create_param_row(frame, "frequency", "active_hours_end", "活跃结束时间", "int", 1, 24, "只在此时间之前进行交易")
    
    def _create_position_tab(self):
        """创建仓位管理参数标签页"""
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="8. 仓位管理")
        
        ttk.Label(frame, text="【仓位管理参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_param_row(frame, "position", "initial_entry_ratio", "初始入场比例", "float", 0.1, 1.0, "首次开仓时使用目标仓位的比例")
        self._create_param_row(frame, "position", "confirm_interval_kline", "确认K线数量", "int", 1, 10, "初始入场后等待多少根K线确认趋势")
        self._create_param_row(frame, "position", "add_on_profit", "盈利加仓", "int", 0, 1, "是否在盈利时加仓(0=否,1=是)")
        self._create_param_row(frame, "position", "add_ratio", "加仓比例", "float", 0.1, 0.5, "每次加仓占目标仓位的比例")
        self._create_param_row(frame, "position", "max_add_times", "最大加仓次数", "int", 1, 10, "最多可以加仓多少次")
    
    def _load_config_to_ui(self):
        """将配置加载到UI"""
        for category, params in self.current_config.items():
            for param_name, value in params.items():
                var_key = f"{category}.{param_name}"
                if var_key in self.config_vars:
                    self.config_vars[var_key].set(value)
    
    def _get_config_from_ui(self):
        """从UI获取配置"""
        config = {}
        for category in ["coordinator", "basic", "entry", "stop_loss", "take_profit", "risk", "frequency", "position"]:
            config[category] = {}
        
        for var_key, var in self.config_vars.items():
            category, param_name = var_key.split(".", 1)
            config[category][param_name] = var.get()
        
        return config
    
    def _load_config(self):
        """载入配置文件"""
        file_path = filedialog.askopenfilename(
            title="载入配置",
            filetypes=[("YAML文件", "*.yaml"), ("YAML文件", "*.yml"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.current_config = yaml.safe_load(f)
                self._load_config_to_ui()
                messagebox.showinfo("成功", "配置已载入！")
            except Exception as e:
                messagebox.showerror("错误", f"载入配置失败: {e}")
    
    def _save_config(self):
        """保存配置文件"""
        config = self._get_config_from_ui()
        file_path = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".yaml",
            filetypes=[("YAML文件", "*.yaml"), ("YAML文件", "*.yml"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                messagebox.showinfo("成功", "配置已保存！")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def _reset_default(self):
        """重置为默认配置"""
        if messagebox.askyesno("确认", "确定要重置为默认配置吗？"):
            self.current_config = self._get_default_config()
            self._load_config_to_ui()
    
    def _apply_config(self):
        """应用配置"""
        self.result = self._get_config_from_ui()
        self.dialog.destroy()
    
    def _get_preset_config(self, preset_name):
        """获取预设配置"""
        presets = {
            "激进超短线": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.8,
                    "TREND_STRENGTH_THRESHOLD": 0.003,
                    "LOOKBACK_PERIOD": 48,
                    "PREDICTION_LENGTH": 8,
                    "CHECK_INTERVAL": 60
                },
                "entry": {
                    "max_kline_change": 0.025,
                    "max_funding_rate_long": 0.05,
                    "min_funding_rate_short": -0.05,
                    "support_buffer": 1.0005,
                    "resistance_buffer": 0.9995
                },
                "stop_loss": {
                    "long_buffer": 0.985,
                    "short_buffer": 1.015
                },
                "take_profit": {
                    "tp1_multiplier_long": 0.965,
                    "tp2_multiplier_long": 1.03,
                    "tp3_multiplier_long": 1.06,
                    "tp1_multiplier_short": 1.035,
                    "tp2_multiplier_short": 0.97,
                    "tp3_multiplier_short": 0.94,
                    "tp1_position_ratio": 0.5,
                    "tp2_position_ratio": 0.3,
                    "tp3_position_ratio": 0.2
                },
                "risk": {
                    "single_trade_risk": 0.05,
                    "daily_loss_limit": 0.2,
                    "max_consecutive_losses": 4,
                    "max_single_position": 0.4,
                    "max_daily_position": 1.0,
                    "extreme_move_threshold": 0.015
                },
                "frequency": {
                    "max_daily_trades": 50,
                    "min_trade_interval_minutes": 3,
                    "active_hours_start": 0,
                    "active_hours_end": 0
                },
                "position": {
                    "initial_entry_ratio": 0.5,
                    "confirm_interval_kline": 1,
                    "add_on_profit": True,
                    "add_ratio": 0.5,
                    "max_add_times": 2
                },
                "strategy": {
                    "entry_confirm_count": 1,
                    "reverse_confirm_count": 1,
                    "require_consecutive_prediction": 1,
                    "post_entry_hours": 2.0,
                    "take_profit_min_pct": 0.3
                }
            },
            "趋势追踪": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.5,
                    "TREND_STRENGTH_THRESHOLD": 0.006,
                    "LOOKBACK_PERIOD": 128,
                    "PREDICTION_LENGTH": 36,
                    "CHECK_INTERVAL": 300
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
                    "tp1_multiplier_long": 0.99,
                    "tp2_multiplier_long": 1.08,
                    "tp3_multiplier_long": 1.2,
                    "tp1_multiplier_short": 1.01,
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
                    "max_daily_position": 0.9,
                    "extreme_move_threshold": 0.025
                },
                "frequency": {
                    "max_daily_trades": 10,
                    "min_trade_interval_minutes": 30,
                    "active_hours_start": 0,
                    "active_hours_end": 0
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
            },
            "平衡型": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.3,
                    "TREND_STRENGTH_THRESHOLD": 0.0047,
                    "LOOKBACK_PERIOD": 96,
                    "PREDICTION_LENGTH": 24,
                    "CHECK_INTERVAL": 180
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
                    "tp1_multiplier_long": 0.975,
                    "tp2_multiplier_long": 1.05,
                    "tp3_multiplier_long": 1.14,
                    "tp1_multiplier_short": 1.025,
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
                    "max_daily_position": 0.85,
                    "extreme_move_threshold": 0.02
                },
                "frequency": {
                    "max_daily_trades": 20,
                    "min_trade_interval_minutes": 10,
                    "active_hours_start": 0,
                    "active_hours_end": 0
                },
                "position": {
                    "initial_entry_ratio": 0.35,
                    "confirm_interval_kline": 3,
                    "add_on_profit": True,
                    "add_ratio": 1.0,
                    "max_add_times": 3
                },
                "strategy": {
                    "entry_confirm_count": 1,
                    "reverse_confirm_count": 2,
                    "require_consecutive_prediction": 2,
                    "post_entry_hours": 6.0,
                    "take_profit_min_pct": 0.5
                }
            },
            "震荡套利": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.1,
                    "TREND_STRENGTH_THRESHOLD": 0.0025,
                    "LOOKBACK_PERIOD": 64,
                    "PREDICTION_LENGTH": 12,
                    "CHECK_INTERVAL": 120
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
                    "tp1_multiplier_long": 0.962,
                    "tp2_multiplier_long": 1.022,
                    "tp3_multiplier_long": 1.035,
                    "tp1_multiplier_short": 1.038,
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
                    "max_daily_position": 0.7,
                    "extreme_move_threshold": 0.01
                },
                "frequency": {
                    "max_daily_trades": 35,
                    "min_trade_interval_minutes": 8,
                    "active_hours_start": 0,
                    "active_hours_end": 0
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
            },
            "稳健长线": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.0,
                    "TREND_STRENGTH_THRESHOLD": 0.008,
                    "LOOKBACK_PERIOD": 192,
                    "PREDICTION_LENGTH": 48,
                    "CHECK_INTERVAL": 600
                },
                "entry": {
                    "max_kline_change": 0.006,
                    "max_funding_rate_long": 0.01,
                    "min_funding_rate_short": -0.01,
                    "support_buffer": 1.003,
                    "resistance_buffer": 0.997
                },
                "stop_loss": {
                    "long_buffer": 0.95,
                    "short_buffer": 1.05
                },
                "take_profit": {
                    "tp1_multiplier_long": 1.01,
                    "tp2_multiplier_long": 1.12,
                    "tp3_multiplier_long": 1.3,
                    "tp1_multiplier_short": 0.99,
                    "tp2_multiplier_short": 0.88,
                    "tp3_multiplier_short": 0.7,
                    "tp1_position_ratio": 0.2,
                    "tp2_position_ratio": 0.3,
                    "tp3_position_ratio": 0.5
                },
                "risk": {
                    "single_trade_risk": 0.015,
                    "daily_loss_limit": 0.05,
                    "max_consecutive_losses": 10,
                    "max_single_position": 0.2,
                    "max_daily_position": 0.5,
                    "extreme_move_threshold": 0.03
                },
                "frequency": {
                    "max_daily_trades": 5,
                    "min_trade_interval_minutes": 60,
                    "active_hours_start": 0,
                    "active_hours_end": 0
                },
                "position": {
                    "initial_entry_ratio": 0.25,
                    "confirm_interval_kline": 6,
                    "add_on_profit": True,
                    "add_ratio": 0.25,
                    "max_add_times": 4
                },
                "strategy": {
                    "entry_confirm_count": 3,
                    "reverse_confirm_count": 4,
                    "require_consecutive_prediction": 4,
                    "post_entry_hours": 24.0,
                    "take_profit_min_pct": 1.2
                }
            },
            "消息驱动": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.3,
                    "TREND_STRENGTH_THRESHOLD": 0.005,
                    "LOOKBACK_PERIOD": 80,
                    "PREDICTION_LENGTH": 18,
                    "CHECK_INTERVAL": 90
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
                    "tp1_multiplier_long": 0.97,
                    "tp2_multiplier_long": 1.045,
                    "tp3_multiplier_long": 1.09,
                    "tp1_multiplier_short": 1.03,
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
                    "max_daily_position": 0.88,
                    "extreme_move_threshold": 0.018
                },
                "frequency": {
                    "max_daily_trades": 25,
                    "min_trade_interval_minutes": 5,
                    "active_hours_start": 0,
                    "active_hours_end": 0
                },
                "position": {
                    "initial_entry_ratio": 0.4,
                    "confirm_interval_kline": 2,
                    "add_on_profit": True,
                    "add_ratio": 0.4,
                    "max_add_times": 2
                },
                "strategy": {
                    "entry_confirm_count": 1,
                    "reverse_confirm_count": 2,
                    "require_consecutive_prediction": 2,
                    "post_entry_hours": 5.0,
                    "take_profit_min_pct": 0.4
                }
            }
        }
        return presets.get(preset_name, presets["平衡型"])
    
    def _apply_preset(self, preset_name):
        """应用预设配置"""
        preset_config = self._get_preset_config(preset_name)
        
        # 更新配置字典
        for category, params in preset_config.items():
            if category in self.current_config:
                self.current_config[category].update(params)
        
        # 更新UI
        self._load_config_to_ui()
        print(f"已应用预设: {preset_name}")
    
    def _on_preset_changed(self, event):
        """预设选择改变时触发"""
        preset_name = self.preset_var.get()
        self._apply_preset(preset_name)
    
    def _cancel(self):
        """取消"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """显示对话框并返回结果"""
        self.dialog.wait_window()
        return self.result


class KronosTradingGUI:
    def _center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        
        # 获取窗口宽度和高度
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        
        # 获取屏幕宽度和高度
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 计算窗口位置
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # 设置窗口位置
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # 对齐完成后显示窗口
        self.root.deiconify()
    
    def __init__(self, root):
        self.root = root
        self.root.title("黑猫交易系统v2.0")
        self.root.geometry("1650x980")
        self.root.configure(bg="#f0f0f0")
        
        # 窗口居中显示
        self._center_window()
        
        # 设置应用图标
        try:
            import sys
            import os
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(__file__)
            icon_path = os.path.join(base_dir, 'app_icon.ico')
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
                # 同时设置 iconphoto 以确保任务栏图标正确显示
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(icon_path)
                    photo = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, photo)
                except Exception:
                    pass
        except Exception as e:
            print(f"设置图标失败: {e}")

        self.trading_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.output_redirector = OutputRedirector(self.queue, self.set_progress)
        
        # 亏损触发AI优化相关变量
        self.optimization_baseline_balance = 0.0  # 优化基准资金
        self.loss_check_timer = None  # 亏损检查定时器
        
        # 交易间隔计时器变量
        self.interval_seconds = 120  # 默认间隔
        self.remaining_seconds = 0   # 剩余秒数
        self.interval_timer = None   # 计时器ID
        self.is_interval_running = False  # 是否正在倒计时

        # 初始化日志文件
        self._init_log_files()

        # Kronos模型缓存（用于可视化）
        self.kronos_model = None
        self.kronos_tokenizer = None
        self.kronos_predictor = None
        self.kronos_model_name = "custom:custom_model"  # 默认模型

        # 多智能体量化交易系统属性
        self.fingpt_analyzer = None
        self.strategy_coordinator = None
        self.use_sentiment_analysis = True  # 默认启用舆情分析
        self.sentiment_filter_enabled = True  # 默认启用信号过滤
        
        # BTC新闻爬虫
        self.news_crawler = None
        self.news_list = []

        # GPU检测初始化
        self._nvml_initialized = False
        self._gpu_available = False
        self._gpu_handle = None

        # 尝试初始化GPU检测
        try:
            print("初始化GPU检测...")
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
                gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                print(f"GPU检测成功: {gpu_name}")
            else:
                print("无GPU设备")
            self._nvml_initialized = True
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            self._nvml_initialized = False

        self.create_styles()
        
        # 初始化实盘监控相关变量（在create_widgets之前）
        self.is_live_monitoring = False
        self.live_monitor_thread = None
        self.trade_executor = None
        self.live_exchange_var = tk.StringVar(value="binance")
        self.live_symbol_var = tk.StringVar(value="BTCUSDT")
        self.account_balance_var = tk.StringVar(value="$0.00")
        self.position_info_var = tk.StringVar(value="暂无持仓")
        self.current_price_var = tk.StringVar(value="--")
        self.price_change_var = tk.StringVar(value="0.00%")
        self.monitor_status_var = tk.StringVar(value="未启动")
        
        # 初始化AI策略中心相关变量
        self.fingpt_status_var = tk.StringVar(value="未初始化")
        self.coordinator_status_var = tk.StringVar(value="未初始化")
        
        # 交易统计变量
        self.today_trades_var = tk.StringVar(value="0")
        self.today_profit_var = tk.StringVar(value="$0.00")
        self.week_trades_var = tk.StringVar(value="0")
        self.week_profit_var = tk.StringVar(value="$0.00")
        self.month_trades_var = tk.StringVar(value="0")
        self.month_profit_var = tk.StringVar(value="$0.00")
        
        # 策略确认次数变量
        self.entry_confirm_count_var = tk.StringVar(value="2")
        self.reverse_confirm_count_var = tk.StringVar(value="2")
        self.require_consecutive_prediction_var = tk.StringVar(value="3")
        
        # 开仓后计时参数
        self.post_entry_hours_var = tk.StringVar(value="2")
        self.take_profit_min_pct_var = tk.StringVar(value="0.6")
        
        # 初始化自动化优化系统变量
        self.performance_monitor = None
        self.auto_optimization_pipeline = None
        self.parameter_integrator = None
        self.is_auto_optimization_enabled = False
        self.auto_optimization_status_var = tk.StringVar(value="禁用")
        self.optimization_threshold_var = tk.StringVar(value="中等")
        self.optimization_frequency_var = tk.StringVar(value="每5分钟")
        self.optimization_judgment_mode_var = tk.StringVar(value="固定阈值")
        
        # 新闻自动刷新相关
        self.news_auto_refresh_job = None
        
        # 初始化多智能体量化交易系统
        self.initialize_multi_agent_system()
        
        self.create_widgets()
        self.load_settings()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.process_queue)
        # 启动新闻自动刷新（30秒后开始，每2分钟刷新）
        self.root.after(30000, self._start_news_auto_refresh)
        # 自动刷新IP地址（界面加载后500ms）
        self.root.after(500, self.refresh_ip_address)
        
        # 创建信号文件，通知启动器主程序已准备好
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            signal_file = os.path.join(base_dir, "_splash_close.txt")
            with open(signal_file, 'w') as f:
                f.write("ready")
        except Exception as e:
            print(f"创建信号文件失败: {e}")

    def _init_log_files(self):
        """初始化日志文件"""
        import logging

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 交易日志
        trade_log = os.path.join(
            log_dir, f"trade_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.trade_logger = logging.getLogger("trade")
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.handlers = []
        self.trade_logger.addHandler(logging.FileHandler(trade_log, encoding="utf-8"))

        # 训练日志
        train_log = os.path.join(
            log_dir, f"train_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.train_logger = logging.getLogger("train")
        self.train_logger.setLevel(logging.INFO)
        self.train_logger.handlers = []
        self.train_logger.addHandler(logging.FileHandler(train_log, encoding="utf-8"))

        # 可视化日志
        visualize_log = os.path.join(
            log_dir, f"visualize_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.visualize_logger = logging.getLogger("visualize")
        self.visualize_logger.setLevel(logging.INFO)
        self.visualize_logger.handlers = []
        self.visualize_logger.addHandler(
            logging.FileHandler(visualize_log, encoding="utf-8")
        )

        # 回测日志
        backtest_log = os.path.join(
            log_dir, f"backtest_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.backtest_logger = logging.getLogger("backtest")
        self.backtest_logger.setLevel(logging.INFO)
        self.backtest_logger.handlers = []
        self.backtest_logger.addHandler(logging.FileHandler(backtest_log, encoding="utf-8"))

    def _load_balance_data(self):
        """加载历史资金数据"""
        try:
            if os.path.exists(self.balance_data_file):
                df = pd.read_csv(self.balance_data_file)
                self.balance_data = df.to_dict('records')
                print(f"已加载 {len(self.balance_data)} 条历史资金数据")
            else:
                self.balance_data = []
        except Exception as e:
            print(f"加载资金数据失败: {e}")
            self.balance_data = []

    def _save_balance_data(self):
        """保存资金数据到文件"""
        try:
            if self.balance_data:
                df = pd.DataFrame(self.balance_data)
                df.to_csv(self.balance_data_file, index=False)
        except Exception as e:
            print(f"保存资金数据失败: {e}")

    def _record_balance(self):
        """记录当前资金数据（每分钟调用）"""
        try:
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 使用合约仓位总资金（包含持仓盈亏）
            initial_balance = self.initial_futures_balance
            
            current_balance = 0.0
            is_strategy_running = False
            
            try:
                if hasattr(self, 'strategy') and self.strategy:
                    # 使用合约仓位总资金（包含持仓盈亏）
                    balance = self.strategy.binance.get_total_balance()
                    if balance and balance > 0:
                        current_balance = balance
                        is_strategy_running = True
            except:
                pass
            
            # 只有在策略运行时才记录数据
            if is_strategy_running:
                data_entry = {
                    'timestamp': timestamp,
                    'initial_balance': initial_balance,
                    'current_balance': current_balance
                }
                
                self.balance_data.append(data_entry)
                
                if len(self.balance_data) > 50000:
                    self.balance_data = self.balance_data[-50000:]
                
                self._save_balance_data()
                
                if hasattr(self, 'balance_chart_canvas'):
                    self._update_balance_chart()
            
        except Exception as e:
            print(f"记录资金数据失败: {e}")
        
        self.balance_record_timer = self.root.after(60000, self._record_balance)

    def _start_balance_recording(self):
        """开始记录资金数据"""
        if self.balance_record_timer is None:
            self._record_balance()

    def _stop_balance_recording(self):
        """停止记录资金数据"""
        if self.balance_record_timer is not None:
            self.root.after_cancel(self.balance_record_timer)
            self.balance_record_timer = None

    def initialize_multi_agent_system(self):
        """初始化多智能体量化交易系统"""
        print("初始化多智能体量化交易系统...")
        
        # 初始化FinGPT舆情分析器
        if FinGPTSentimentAnalyzer is not None and self.use_sentiment_analysis:
            try:
                print("  正在初始化FinGPT舆情分析器...")
                self.fingpt_analyzer = FinGPTSentimentAnalyzer(
                    use_local_model=True,
                    use_qwen_preprocessing=True
                )
                print("  ✓ FinGPT舆情分析器初始化完成")
            except Exception as e:
                print(f"  ✗ FinGPT舆情分析器初始化失败: {e}")
                self.fingpt_analyzer = None
        else:
            print("  ⚠ FinGPT模块不可用或已禁用")
            self.fingpt_analyzer = None
        
        # 初始化策略协调器
        if StrategyCoordinator is not None:
            try:
                print("  正在初始化策略协调器...")
                self.strategy_coordinator = StrategyCoordinator(
                    kronos_model_name="custom:custom_model",
                    use_fingpt=(self.fingpt_analyzer is not None),
                    symbol="BTC"
                )
                print("  ✓ 策略协调器初始化完成")
                
                # 应用等待的参数（如果有）
                if hasattr(self, "_pending_coordinator_params") and self._pending_coordinator_params:
                    print("  应用之前优化的协调器参数...")
                    self.strategy_coordinator.update_config(self._pending_coordinator_params)
                    print("  ✓ 协调器参数已应用")
            except Exception as e:
                print(f"  ✗ 策略协调器初始化失败: {e}")
                self.strategy_coordinator = None
        else:
            print("  ⚠ 策略协调器模块不可用")
            self.strategy_coordinator = None
        
        print("多智能体量化交易系统初始化完成")
        
        # 初始化BTC新闻爬虫
        if BTCNewsCrawler is not None:
            try:
                print("  正在初始化BTC新闻爬虫...")
                self.news_crawler = BTCNewsCrawler()
                print("  ✓ BTC新闻爬虫初始化完成")
            except Exception as e:
                print(f"  ✗ BTC新闻爬虫初始化失败: {e}")
                self.news_crawler = None
        
        # 更新GUI状态显示（使用after确保GUI已创建）
        self.root.after(100, self.update_multi_agent_status)
    
    def update_multi_agent_status(self):
        """更新多智能体系统状态显示"""
        # 更新FinGPT状态
        if self.fingpt_analyzer is not None:
            self.fingpt_status_var.set("运行中")
            self.fingpt_status_label.config(foreground="#27ae60")  # 绿色
        else:
            self.fingpt_status_var.set("未启用")
            self.fingpt_status_label.config(foreground="#f39c12")  # 橙色
        
        # 更新策略协调器状态
        if self.strategy_coordinator is not None:
            self.coordinator_status_var.set("运行中")
            self.coordinator_status_label.config(foreground="#27ae60")  # 绿色
        else:
            self.coordinator_status_var.set("未启用")
            self.coordinator_status_label.config(foreground="#f39c12")  # 橙色

    def refresh_model_list(self):
        """刷新模型列表"""
        model_list = ["kronos-small", "kronos-mini", "kronos-base"]
        seen_models = set(model_list)

        # 扫描训练好的模型目录
        possible_dirs = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Kronos",
                "finetune_csv",
                "finetuned",
            ),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Kronos",
                "finetune_csv",
                "output",
            ),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Kronos",
                "finetune_csv",
                "Kronos",
                "finetune_csv",
                "finetuned",
            ),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Kronos", "finetune_csv"
            ),
        ]

        for train_output_dir in possible_dirs:
            if not os.path.exists(train_output_dir):
                continue
            try:
                for exp_name in os.listdir(train_output_dir):
                    exp_path = os.path.join(train_output_dir, exp_name)
                    if os.path.isdir(exp_path):
                        # 检查是否有训练好的模型
                        tokenizer_path = os.path.join(
                            exp_path, "tokenizer", "best_model"
                        )
                        basemodel_path = os.path.join(
                            exp_path, "basemodel", "best_model"
                        )
                        # 也检查嵌套目录结构
                        tokenizer_path_alt = os.path.join(
                            exp_path, "tokenizer", "best_model"
                        )
                        basemodel_path_alt = os.path.join(
                            exp_path, "basemodel", "best_model"
                        )
                        if (
                            os.path.exists(tokenizer_path)
                            and os.path.exists(basemodel_path)
                        ) or (
                            os.path.exists(tokenizer_path_alt)
                            and os.path.exists(basemodel_path_alt)
                        ):
                            model_name = f"custom:{exp_name}"
                            if model_name not in seen_models:
                                model_list.append(model_name)
                                seen_models.add(model_name)
            except Exception as e:
                print(f"扫描训练模型失败: {e}")

        # 也可以检查用户自定义的模型目录
        custom_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "custom_models"
        )
        if os.path.exists(custom_model_dir):
            try:
                for model_name in os.listdir(custom_model_dir):
                    model_path = os.path.join(custom_model_dir, model_name)
                    if os.path.isdir(model_path):
                        full_model_name = f"custom:{model_name}"
                        if full_model_name not in seen_models:
                            model_list.append(full_model_name)
                            seen_models.add(full_model_name)
            except Exception as e:
                print(f"扫描自定义模型失败: {e}")

        self.model_combo["values"] = tuple(model_list)

    def create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure(
            "TLabel", background="#f0f0f0", foreground="#333333", font=LARGE_FONT
        )
        style.configure("Title.TLabel", font=TITLE_FONT, foreground="#0066cc")
        style.configure(
            "TCheckbutton", background="#f0f0f0", foreground="#333333", font=LARGE_FONT
        )
        style.configure("TButton", font=BOLD_FONT)
        style.map("TButton", background=[("active", "#0066cc")])
        style.configure(
            "green.Horizontal.TProgressbar",
            troughbackground="#e0e0e0",
            background="#00cc00",
            thickness=20,
        )
        style.configure('Accent.TButton', 
                       font=('微软雅黑', 11, 'bold'),
                       padding=10,
                       foreground='white',
                       background='#3498db',
                       bordercolor='#2980b9',
                       focuscolor='none')
        style.map('Accent.TButton',
                 background=[('active', '#2980b9')],
                 relief=[('pressed', 'sunken')])

    def refresh_ip_address(self):
        """刷新获取当前IP地址"""
        try:
            self.ip_address_var.set("获取中...")
            import socket
            import requests
            
            # 方法1: 使用socket获取本地IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "127.0.0.1"
            
            # 方法2: 使用API获取公网IP（可选）
            public_ip = "获取中..."
            try:
                response = requests.get("https://api.ipify.org?format=json", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    public_ip = data.get("ip", "未知")
            except:
                public_ip = "获取失败"
            
            # 显示本地IP和公网IP
            self.ip_address_var.set(f"本地:{local_ip} | 公网:{public_ip}")
        except Exception as e:
            self.ip_address_var.set(f"获取失败: {str(e)}")

    def create_widgets(self):
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_container, style="TFrame", width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_container, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 右侧分为上下两部分：上面是系统监控，下面是终端
        right_top_frame = ttk.Frame(right_frame, style="TFrame")
        right_top_frame.pack(fill=tk.X, padx=5, pady=(5, 10))

        right_bottom_frame = ttk.Frame(right_frame, style="TFrame")
        right_bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.create_api_section(left_frame)
        self.create_config_section(left_frame)
        self.create_control_section(left_frame)
        self.create_system_monitor_section(right_top_frame)
        self.create_terminal_section(right_bottom_frame)

    def create_system_monitor_section(self, parent):
        """创建系统监控面板"""
        # CPU
        cpu_frame = ttk.Frame(parent)
        cpu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(cpu_frame, text="CPU", font=("微软雅黑", 10, "bold")).pack()
        self.cpu_var = tk.StringVar(value="0%")
        self.cpu_label = ttk.Label(
            cpu_frame,
            textvariable=self.cpu_var,
            font=("微软雅黑", 16, "bold"),
            foreground="#007bff",
        )
        self.cpu_label.pack()
        self.cpu_progress = ttk.Progressbar(cpu_frame, mode="determinate", length=80)
        self.cpu_progress.pack(fill=tk.X, pady=(0, 5))

        self.cpu_temp_var = tk.StringVar(value="--°C")
        self.cpu_temp_label = ttk.Label(
            cpu_frame, textvariable=self.cpu_temp_var, font=("微软雅黑", 9)
        )
        self.cpu_temp_label.pack()

        # 内存
        mem_frame = ttk.Frame(parent)
        mem_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(mem_frame, text="内存", font=("微软雅黑", 10, "bold")).pack()
        self.mem_var = tk.StringVar(value="0%")
        self.mem_label = ttk.Label(
            mem_frame,
            textvariable=self.mem_var,
            font=("微软雅黑", 16, "bold"),
            foreground="#ff9500",
        )
        self.mem_label.pack()
        self.mem_progress = ttk.Progressbar(mem_frame, mode="determinate", length=80)
        self.mem_progress.pack(fill=tk.X, pady=(0, 5))

        self.mem_detail_var = tk.StringVar(value="")
        self.mem_detail_label = ttk.Label(
            mem_frame, textvariable=self.mem_detail_var, font=("微软雅黑", 9)
        )
        self.mem_detail_label.pack()

        # GPU
        gpu_frame = ttk.Frame(parent)
        gpu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(gpu_frame, text="GPU", font=("微软雅黑", 10, "bold")).pack()
        self.gpu_var = tk.StringVar(value="检测中...")
        self.gpu_label = ttk.Label(
            gpu_frame,
            textvariable=self.gpu_var,
            font=("微软雅黑", 16, "bold"),
            foreground="#28a745",
        )
        self.gpu_label.pack()
        self.gpu_progress = ttk.Progressbar(gpu_frame, mode="determinate", length=80)
        self.gpu_progress.pack(fill=tk.X, pady=(0, 5))

        self.gpu_temp_var = tk.StringVar(value="--°C")
        self.gpu_temp_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_temp_var, font=("微软雅黑", 9)
        )
        self.gpu_temp_label.pack()

        self.gpu_3d_var = tk.StringVar(value="3D: --%")
        self.gpu_3d_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_3d_var, font=("微软雅黑", 9)
        )
        self.gpu_3d_label.pack()

        self.gpu_detail_var = tk.StringVar(value="")
        self.gpu_detail_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_detail_var, font=("微软雅黑", 8)
        )
        self.gpu_detail_label.pack()

        # 磁盘
        disk_frame = ttk.Frame(parent)
        disk_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(disk_frame, text="磁盘", font=("微软雅黑", 10, "bold")).pack()
        self.disk_var = tk.StringVar(value="0%")
        self.disk_label = ttk.Label(
            disk_frame,
            textvariable=self.disk_var,
            font=("微软雅黑", 16, "bold"),
            foreground="#6c757d",
        )
        self.disk_label.pack()
        self.disk_progress = ttk.Progressbar(disk_frame, mode="determinate", length=80)
        self.disk_progress.pack(fill=tk.X, pady=(0, 5))

        self.disk_detail_var = tk.StringVar(value="")
        self.disk_detail_label = ttk.Label(
            disk_frame, textvariable=self.disk_detail_var, font=("微软雅黑", 9)
        )
        self.disk_detail_label.pack()

        # 启动定时更新
        self._update_system_info()
        self._system_timer = self.root.after(2000, self._update_system_info)

    def _update_system_info(self):
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.3)
            self.cpu_var.set(f"{cpu_percent:.0f}%")
            self.cpu_progress["value"] = cpu_percent

            # CPU温度
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries and hasattr(entries[0], "current"):
                            self.cpu_temp_var.set(f"{entries[0].current:.0f}°C")
                            break
            except:
                self.cpu_temp_var.set("--°C")

            # 内存
            mem = psutil.virtual_memory()
            self.mem_var.set(f"{mem.percent:.0f}%")
            self.mem_progress["value"] = mem.percent
            self.mem_detail_var.set(f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB")

            # GPU - 使用pynvml直接检测
            if not self._nvml_initialized:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        self._gpu_available = True
                        self._nvml_initialized = True
                        print("GPU重新初始化成功")
                    else:
                        self._gpu_available = False
                        self._gpu_handle = None
                except Exception as e:
                    print(f"GPU初始化失败: {e}")
                    self._nvml_initialized = False
                    self._gpu_available = False
                    self._gpu_handle = None

            if self._gpu_available and self._gpu_handle:
                try:
                    # GPU温度
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        self.gpu_temp_var.set(f"{temp}°C")
                    except Exception as e:
                        print(f"获取温度失败: {e}")
                        self.gpu_temp_var.set("--°C")

                    # GPU 3D使用率
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                        self.gpu_3d_var.set(f"3D: {util.gpu}%")
                    except Exception as e:
                        print(f"获取使用率失败: {e}")
                        self.gpu_3d_var.set("3D: --%")

                    # GPU显存
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        gpu_mem_used = mem_info.used / 1024**3
                        gpu_mem_total = mem_info.total / 1024**3
                        gpu_util = (
                            (gpu_mem_used / gpu_mem_total) * 100
                            if gpu_mem_total > 0
                            else 0
                        )
                        self.gpu_var.set(f"{gpu_util:.0f}%")
                        self.gpu_progress["value"] = gpu_util
                        self.gpu_detail_var.set(
                            f"{gpu_mem_used:.1f}/{gpu_mem_total:.0f}GB"
                        )
                    except Exception as e:
                        print(f"获取显存失败: {e}")
                        # 如果获取显存失败，显示3D使用率
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(
                                self._gpu_handle
                            )
                            self.gpu_var.set(f"{util.gpu}%")
                            self.gpu_progress["value"] = util.gpu
                        except Exception as e2:
                            print(f"获取3D使用率失败: {e2}")
                            self.gpu_var.set("--%")
                        self.gpu_detail_var.set("")
                except Exception as e:
                    print(f"GPU检测错误: {e}")
                    self.gpu_temp_var.set("--°C")
                    self.gpu_3d_var.set("3D: --%")
                    self.gpu_var.set("无GPU")
                    self.gpu_detail_var.set("")
                    # 重置状态，下次重新初始化
                    self._nvml_initialized = False
                    self._gpu_available = False
                    self._gpu_handle = None
            else:
                self.gpu_temp_var.set("--°C")
                self.gpu_3d_var.set("3D: --%")
                self.gpu_var.set("无GPU")
                self.gpu_detail_var.set("")
                # 尝试重新初始化
                if not self._nvml_initialized:
                    try:
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        if device_count > 0:
                            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            self._gpu_available = True
                            self._nvml_initialized = True
                            print("GPU重新初始化成功")
                        else:
                            self._gpu_available = False
                            self._gpu_handle = None
                    except Exception as e:
                        print(f"GPU初始化失败: {e}")
                        self._nvml_initialized = False
                        self._gpu_available = False
                        self._gpu_handle = None

            # 磁盘
            try:
                disk = psutil.disk_usage("C:")
                self.disk_var.set(f"{disk.percent:.0f}%")
                self.disk_progress["value"] = disk.percent
                self.disk_detail_var.set(
                    f"{disk.used/1024**3:.0f}/{disk.total/1024**3:.0f}GB"
                )
            except:
                self.disk_var.set("--%")
                self.disk_detail_var.set("")

        except Exception as e:
            print(f"系统监控更新错误: {e}")

        self._system_timer = self.root.after(2000, self._update_system_info)

    def create_api_section(self, parent):
        api_frame = ttk.LabelFrame(parent, text="API 配置", style="TFrame", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))

        api_grid = ttk.Frame(api_frame)
        api_grid.pack(fill=tk.X, pady=(0, 5))

        api_col1 = ttk.Frame(api_grid)
        api_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(api_col1, text="币安 API Key:").pack(anchor=tk.W)
        self.api_key_entry = ttk.Entry(api_col1, width=20)
        self.api_key_entry.pack(fill=tk.X, pady=(0, 0))

        api_col2 = ttk.Frame(api_grid)
        api_col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(api_col2, text="币安 Secret Key:").pack(anchor=tk.W)
        self.secret_key_entry = ttk.Entry(api_col2, width=20)
        self.secret_key_entry.pack(fill=tk.X, pady=(0, 0))

        self.show_api_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            api_frame,
            text="显示API密钥",
            variable=self.show_api_var,
            command=self.toggle_api_visibility,
        ).pack(anchor=tk.W)

    def create_config_section(self, parent):
        config_frame = ttk.LabelFrame(
            parent, text="交易配置", style="TFrame", padding=10
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # 交易对 - 第一行
        pair_frame = ttk.Frame(config_frame)
        pair_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pair_frame, text="交易对:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Label(pair_frame, text="BTCUSDT", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT)
        
        # IP地址 - 第二行
        ip_frame = ttk.Frame(config_frame)
        ip_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ip_frame, text="IP地址:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.ip_address_var = tk.StringVar(value="获取中...")
        self.ip_address_label = ttk.Label(ip_frame, textvariable=self.ip_address_var, font=("微软雅黑", 10))
        self.ip_address_label.pack(side=tk.LEFT)
        # 添加获取IP地址按钮
        ttk.Button(ip_frame, text="刷新", command=self.refresh_ip_address).pack(side=tk.LEFT, padx=(10, 0))

        # 交易策略和Kronos模型 - 一行两列
        strategy_model_grid = ttk.Frame(config_frame)
        strategy_model_grid.pack(fill=tk.X, pady=(0, 5))

        strategy_col = ttk.Frame(strategy_model_grid)
        strategy_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(strategy_col, text="交易策略:").pack(anchor=tk.W)
        self.strategy_var = tk.StringVar(value="自动策略")
        strategy_combo = ttk.Combobox(
            strategy_col, textvariable=self.strategy_var, state="readonly", width=15
        )
        strategy_combo["values"] = (
            "趋势爆发",
            "震荡套利",
            "消息突破",
            "自动策略",
            "时间策略",
        )
        strategy_combo.pack(fill=tk.X, pady=(0, 0))
        strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_changed)

        model_col = ttk.Frame(strategy_model_grid)
        model_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(model_col, text="Kronos模型:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="custom:custom_model")
        self.model_combo = ttk.Combobox(
            model_col, textvariable=self.model_var, state="readonly", width=15
        )
        self.refresh_model_list()
        self.model_combo.pack(fill=tk.X, pady=(0, 0))
        
        # 模型切换时重置Kronos predictor
        def on_model_change(*args):
            self.kronos_predictor = None
            self.kronos_model = None
            self.kronos_tokenizer = None
            self.kronos_model_name = self.model_var.get()
        
        self.model_var.trace_add("write", on_model_change)

        # 刷新模型按钮
        refresh_model_frame = ttk.Frame(config_frame)
        refresh_model_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            refresh_model_frame, text="刷新模型列表", command=self.refresh_model_list
        ).pack(side=tk.RIGHT)

        # 参数 - 5行两列，共10个参数
        params_grid = ttk.Frame(config_frame)
        params_grid.pack(fill=tk.X, pady=(0, 0))

        # 行1: 分析周期 + 杠杆倍数
        row1 = ttk.Frame(params_grid)
        row1.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row1)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="分析周期:").pack(anchor=tk.W)
        self.timeframe_var = tk.StringVar(value="5m")
        timeframe_combo = ttk.Combobox(
            col1, textvariable=self.timeframe_var, state="readonly", width=15
        )
        timeframe_combo["values"] = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        timeframe_combo.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row1)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="杠杆倍数:").pack(anchor=tk.W)
        self.leverage_var = tk.StringVar(value="10")
        leverage_combo = ttk.Combobox(
            col2, textvariable=self.leverage_var, state="readonly", width=15
        )
        leverage_combo["values"] = ("1", "2", "3", "5", "10", "20", "25", "50", "75", "100")
        leverage_combo.pack(fill=tk.X, pady=(0, 0))

        # 行2: 最小仓位 + 交易间隔
        row2 = ttk.Frame(params_grid)
        row2.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row2)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="最小仓位 (USDT):").pack(anchor=tk.W)
        self.min_position_var = tk.StringVar(value="100")
        min_pos_entry = ttk.Entry(
            col1, textvariable=self.min_position_var, width=18
        )
        min_pos_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row2)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="交易间隔 (秒):").pack(anchor=tk.W)
        self.interval_var = tk.StringVar(value="120")
        interval_combo = ttk.Combobox(
            col2, textvariable=self.interval_var, state="readonly", width=15
        )
        interval_combo["values"] = ("30", "60", "120", "180", "300", "600")
        interval_combo.pack(fill=tk.X, pady=(0, 0))

        # 行3: 趋势阈值 + AI最小趋势强度
        row3 = ttk.Frame(params_grid)
        row3.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row3)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="趋势阈值:").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="0.008")
        threshold_entry = ttk.Entry(
            col1, textvariable=self.threshold_var, width=18
        )
        threshold_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row3)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="AI最小趋势强度:").pack(anchor=tk.W)
        self.ai_min_trend_var = tk.StringVar(value="0.010")
        ai_min_trend_entry = ttk.Entry(
            col2, textvariable=self.ai_min_trend_var, width=18
        )
        ai_min_trend_entry.pack(fill=tk.X, pady=(0, 0))

        # 行4: AI最小预测偏离度 + 最大资金费率
        row4 = ttk.Frame(params_grid)
        row4.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row4)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="AI最小预测偏离度:").pack(anchor=tk.W)
        self.ai_min_deviation_var = tk.StringVar(value="0.008")
        ai_min_deviation_entry = ttk.Entry(
            col1, textvariable=self.ai_min_deviation_var, width=18
        )
        ai_min_deviation_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row4)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="最大资金费率 (%):").pack(anchor=tk.W)
        self.max_funding_var = tk.StringVar(value="1.0")
        max_funding_entry = ttk.Entry(
            col2, textvariable=self.max_funding_var, width=18
        )
        max_funding_entry.pack(fill=tk.X, pady=(0, 0))

        # 行5: 最小资金费率 + 亏损触发优化
        row5 = ttk.Frame(params_grid)
        row5.pack(fill=tk.X, pady=(0, 0))
        col1 = ttk.Frame(row5)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="最小资金费率 (%):").pack(anchor=tk.W)
        self.min_funding_var = tk.StringVar(value="-1.0")
        min_funding_entry = ttk.Entry(
            col1, textvariable=self.min_funding_var, width=18
        )
        min_funding_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row5)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="亏损触发优化 (%):").pack(anchor=tk.W)
        self.loss_trigger_var = tk.StringVar(value="10.0")
        loss_trigger_entry = ttk.Entry(
            col2, textvariable=self.loss_trigger_var, width=18
        )
        loss_trigger_entry.pack(fill=tk.X, pady=(0, 0))

    def create_control_section(self, parent):
        control_frame = ttk.Frame(parent, style="TFrame", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))

        self.start_button = ttk.Button(
            button_frame, text="启动交易", command=self.start_trading
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        self.stop_button = ttk.Button(
            button_frame, text="停止交易", command=self.stop_trading, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        # 资金信息显示
        fund_frame = ttk.LabelFrame(control_frame, text="合约资金", padding=8)
        fund_frame.pack(fill=tk.X, pady=(10, 0))

        # 设置字体
        label_font = ("微软雅黑", 10)
        value_font = ("微软雅黑", 11, "bold")

        # 初始资金 (第1行第1列)
        ttk.Label(fund_frame, text="初始资金:", font=label_font).grid(
            row=0, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.initial_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="blue"
        )
        self.initial_balance_label.grid(row=0, column=1, sticky=tk.E, pady=4, padx=5)

        # 当前资金 (第1行第2列)
        ttk.Label(fund_frame, text="当前资金:", font=label_font).grid(
            row=0, column=2, sticky=tk.W, pady=4, padx=5
        )
        self.current_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="green"
        )
        self.current_balance_label.grid(row=0, column=3, sticky=tk.E, pady=4, padx=5)

        # 盈亏金额 (第2行第1列)
        ttk.Label(fund_frame, text="盈亏金额:", font=label_font).grid(
            row=1, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="red"
        )
        self.pnl_label.grid(row=1, column=1, sticky=tk.E, pady=4, padx=5)

        # 盈亏比例 (第2行第2列)
        ttk.Label(fund_frame, text="盈亏比例:", font=label_font).grid(
            row=1, column=2, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_pct_label = ttk.Label(
            fund_frame, text="0.00%", font=value_font, foreground="red"
        )
        self.pnl_pct_label.grid(row=1, column=3, sticky=tk.E, pady=4, padx=5)

        # 初始化资金变量
        self.initial_futures_balance = 0.0
        
        # 资金数据记录
        self.balance_data_file = os.path.join(os.path.dirname(__file__), "balance_data.csv")
        self.balance_data = []
        self.balance_record_timer = None
        self._load_balance_data()
        
        # 程序启动后立即开始记录资金数据
        self.root.after(1000, self._start_balance_recording)

        # Kronos预测可视化图表
        prediction_chart_frame = ttk.LabelFrame(control_frame, text="📊 Kronos预测走势", padding=8)
        prediction_chart_frame.pack(fill=tk.X, pady=(10, 0))

        # 导入matplotlib
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # 存储Kronos分析结果的变量
            self.kronos_analysis_data = {
                'trend_direction': None,
                'trend_strength': 0,
                'pred_change': 0,
                'threshold': 0.008,
                'timestamp': None
            }
            
            # 创建图表
            self.prediction_fig = plt.Figure(figsize=(5.5, 3.0), dpi=100)
            self.prediction_ax = self.prediction_fig.add_subplot(111)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 初始绘制空白图表
            self._draw_prediction_chart()
            
            # 嵌入到Tkinter
            self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, master=prediction_chart_frame)
            self.prediction_canvas.draw()
            self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except ImportError as e:
            ttk.Label(prediction_chart_frame, text=f"图表库未安装: {e}", foreground="red").pack()
            self.prediction_fig = None
            self.prediction_ax = None
            self.prediction_canvas = None
            self.kronos_analysis_data = None

    def create_terminal_section(self, parent):
        # 创建笔记本控件（标签页）
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 交易日志标签页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="交易日志")
        self._create_log_tab(log_frame)

        # AI实盘策略标签页（合并AI策略中心 + 实盘监控）
        ai_trading_frame = ttk.Frame(notebook)
        notebook.add(ai_trading_frame, text="AI实盘策略")
        self._create_ai_trading_tab(ai_trading_frame)
        
        # 策略回测标签页
        backtest_frame = ttk.Frame(notebook)
        notebook.add(backtest_frame, text="策略回测")
        self._create_main_backtest_tab(backtest_frame)
        
        # BTC新闻标签页
        news_frame = ttk.Frame(notebook)
        notebook.add(news_frame, text="BTC新闻")
        self._create_news_tab(news_frame)

        # 可视化预测标签页
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="可视化预测")
        self._create_visualization_tab(viz_frame)
        
        # 资金曲线标签页
        balance_chart_frame = ttk.Frame(notebook)
        notebook.add(balance_chart_frame, text="资金曲线")
        self._create_balance_chart_tab(balance_chart_frame)

        # 训练Kronos模型标签页
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="训练Kronos模型")
        self._create_training_tab(train_frame)

        # 训练文件管理标签页
        training_manager_frame = ttk.Frame(notebook)
        notebook.add(training_manager_frame, text="训练文件管理")
        self._create_training_manager_tab(training_manager_frame)
        
        # 帮助标签页
        help_frame = ttk.Frame(notebook)
        notebook.add(help_frame, text="帮助")
        self._create_help_tab(help_frame)

    def _create_log_tab(self, parent):
        # 交易间隔进度条框架
        interval_frame = ttk.Frame(parent)
        interval_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 交易间隔标签
        self.interval_label = ttk.Label(
            interval_frame, 
            text="交易间隔: 空闲", 
            font=("微软雅黑", 10),
            style="TLabel"
        )
        self.interval_label.pack(anchor=tk.W, pady=(0, 3))
        
        # 进度条 - 显示交易间隔倒计时
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            interval_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=200,
            style="green.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # 剩余时间标签
        self.remaining_time_var = tk.StringVar(value="-- 秒")
        self.remaining_time_label = ttk.Label(
            interval_frame,
            textvariable=self.remaining_time_var,
            font=("微软雅黑", 9),
            foreground="#7f8c8d"
        )
        self.remaining_time_label.pack(anchor=tk.E)
        
        # 状态标签
        self.status_label = ttk.Label(parent, text="就绪", style="TLabel")
        self.status_label.pack(anchor=tk.W, pady=(0, 5))

        # 终端输出
        self.terminal = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#333333",
            font=("Consolas", 12),
            insertbackground="#333333",
        )
        self.terminal.pack(fill=tk.BOTH, expand=True)

    def _create_visualization_tab(self, parent):
        # 可视化控制按钮框架
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 生成可视化按钮
        self.viz_button = ttk.Button(
            control_frame,
            text="生成Kronos预测图表",
            command=self.generate_visualization,
        )
        self.viz_button.pack(side=tk.LEFT, padx=5)

        # 模式选择
        ttk.Label(control_frame, text="模式:").pack(side=tk.LEFT, padx=(20, 5))
        self.viz_mode_var = tk.StringVar(value="预测未来")
        mode_combo = ttk.Combobox(
            control_frame, textvariable=self.viz_mode_var, state="readonly", width=12
        )
        mode_combo["values"] = ("预测未来", "回测过去")
        mode_combo.pack(side=tk.LEFT, padx=5)

        # 配置选项
        ttk.Label(control_frame, text="预测长度:").pack(side=tk.LEFT, padx=(20, 5))
        self.pred_len_var = tk.StringVar(value="120")
        pred_len_combo = ttk.Combobox(
            control_frame, textvariable=self.pred_len_var, state="readonly", width=10
        )
        pred_len_combo["values"] = (
            "10",
            "15",
            "20",
            "30",
            "45",
            "60",
            "120",
            "180",
            "240",
            "300",
        )
        pred_len_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="回看周期:").pack(side=tk.LEFT, padx=(20, 5))
        self.lookback_var = tk.StringVar(value="512")
        lookback_combo = ttk.Combobox(
            control_frame, textvariable=self.lookback_var, state="readonly", width=10
        )
        lookback_combo["values"] = ("50", "100", "200", "300", "400", "512", "600")
        lookback_combo.pack(side=tk.LEFT, padx=5)

        # 图表画布框架
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 创建matplotlib图形
        self.viz_figure = Figure(figsize=(8, 6), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, master=canvas_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态标签
        self.viz_status_label = ttk.Label(
            parent, text="就绪，点击按钮生成可视化", style="TLabel"
        )
        self.viz_status_label.pack(anchor=tk.W, pady=(5, 0))

    def generate_visualization(self):
        """生成Kronos预测可视化图表"""
        try:
            # 禁用按钮防止重复点击
            self.viz_button.config(state=tk.DISABLED)
            self.viz_status_label.config(text="正在生成可视化...")
            self.viz_canvas.get_tk_widget().update()

            # 添加DatetimeIndex兼容性（Kronos模型需要）
            pd.DatetimeIndex.dt = property(lambda self: self)

            # 获取配置参数
            lookback = int(self.lookback_var.get())
            pred_len = int(self.pred_len_var.get())
            viz_mode = self.viz_mode_var.get()

            # 更新状态
            if viz_mode == "预测未来":
                self.viz_status_label.config(
                    text=f"[预测未来] 正在获取市场数据... (回看{lookback}周期)"
                )
                data_len = lookback
            else:
                self.viz_status_label.config(
                    text=f"[回测过去] 正在获取市场数据... (回看{lookback}+{pred_len}周期)"
                )
                data_len = lookback + pred_len

            # 获取K线数据
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()

            # 使用binance_api获取数据
            from binance_api import BinanceAPI

            binance = BinanceAPI()
            klines_df = binance.get_recent_klines(symbol, timeframe, data_len)

            if klines_df is None or len(klines_df) < lookback:
                self.viz_status_label.config(text="获取数据失败或数据不足")
                self.viz_button.config(state=tk.NORMAL)
                return

            # get_recent_klines返回的是已经处理好的DataFrame
            # 检查列名
            self.log(f"K线数据列: {list(klines_df.columns)}")
            self.log(f"K线数据形状: {klines_df.shape}")

            # 重命名列以匹配Kronos期望的格式
            # get_klines返回的列是: ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            if "timestamps" in klines_df.columns:
                df = klines_df.rename(columns={"timestamps": "timestamp"})
            else:
                df = klines_df.copy()

            # 确保有timestamp列
            if "timestamp" not in df.columns:
                self.viz_status_label.config(text="K线数据缺少timestamp列")
                self.viz_button.config(state=tk.NORMAL)
                return

            # 只保留需要的列
            needed_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in needed_cols if col not in df.columns]
            if missing_cols:
                self.log(f"缺少列: {missing_cols}")
                self.viz_status_label.config(text=f"K线数据缺少列: {missing_cols}")
                self.viz_button.config(state=tk.NORMAL)
                return

            df = df[needed_cols].copy()

            # 确保timestamp是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                self.log("转换timestamp列为datetime类型")
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            self.log(f"处理后的DataFrame形状: {df.shape}")
            self.log(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

            # 更新状态
            self.viz_status_label.config(text="正在加载Kronos模型...")
            self.viz_canvas.get_tk_widget().update()

            # 加载Kronos模型（使用缓存）
            # 在PyInstaller环境中正确设置路径
            if getattr(sys, 'frozen', False):
                # 如果是exe，Kronos目录在sys._MEIPASS下
                kronos_path = os.path.join(sys._MEIPASS, "Kronos")
            else:
                kronos_path = os.path.join(os.path.dirname(__file__), "Kronos")
            sys.path.append(kronos_path)
            from Kronos.model import Kronos, KronosTokenizer, KronosPredictor

            if self.kronos_tokenizer is None or self.kronos_model is None:
                self.viz_status_label.config(text="正在加载Kronos模型...")
                self.viz_canvas.get_tk_widget().update()

                # 获取模型目录路径
                if getattr(sys, 'frozen', False):
                    base_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                
                # 检查 _internal/models 目录（打包环境），如果不存在再检查 models 目录
                tokenizer_path = os.path.join(base_dir, "_internal", "models", "kronos-tokenizer-base")
                if not os.path.exists(tokenizer_path):
                    tokenizer_path = os.path.join(base_dir, "models", "kronos-tokenizer-base")
                
                model_path = os.path.join(base_dir, "_internal", "models", "kronos-small")
                if not os.path.exists(model_path):
                    model_path = os.path.join(base_dir, "models", "kronos-small")
                
                self.log(f"Tokenizer路径: {tokenizer_path}")
                self.log(f"Model路径: {model_path}")
                
                if os.path.exists(tokenizer_path) and os.path.exists(model_path):
                    self.viz_status_label.config(text="从本地加载Kronos模型...")
                    self.log("从本地目录加载Kronos模型")
                else:
                    self.viz_status_label.config(text="模型文件缺失")
                    self.log("错误: 模型文件不存在，请确保models目录完整")
                    messagebox.showerror(
                        "模型缺失", 
                        "未找到Kronos模型文件！\n\n"
                        "请确保models目录包含以下内容：\n"
                        "- models/kronos-small/\n"
                        "- models/kronos-tokenizer-base/"
                    )
                    return

                try:
                    import json
                    from safetensors.torch import load_file
                    
                    self.log("手动加载Kronos Tokenizer...")
                    tokenizer_config_path = os.path.join(tokenizer_path, "config.json")
                    tokenizer_weights_path = os.path.join(tokenizer_path, "model.safetensors")
                    
                    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                        tokenizer_config = json.load(f)
                    
                    self.kronos_tokenizer = KronosTokenizer(**tokenizer_config)
                    tokenizer_state_dict = load_file(tokenizer_weights_path)
                    self.kronos_tokenizer.load_state_dict(tokenizer_state_dict)
                    self.kronos_tokenizer.eval()
                    self.log("Tokenizer加载完成")
                    
                    self.log("手动加载Kronos Model...")
                    model_config_path = os.path.join(model_path, "config.json")
                    model_weights_path = os.path.join(model_path, "model.safetensors")
                    
                    with open(model_config_path, "r", encoding="utf-8") as f:
                        model_config = json.load(f)
                    
                    self.kronos_model = Kronos(**model_config)
                    model_state_dict = load_file(model_weights_path)
                    self.kronos_model.load_state_dict(model_state_dict)
                    self.kronos_model.eval()
                    self.log("Model加载完成")
                    
                except Exception as e:
                    self.log(f"手动加载失败: {e}")
                    self.log("尝试使用from_pretrained方法(仅本地)...")
                    try:
                        self.kronos_tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                        self.kronos_model = Kronos.from_pretrained(model_path, local_files_only=True)
                    except Exception as e2:
                        self.log(f"from_pretrained本地加载也失败: {e2}")
                        self.log("请检查模型文件是否完整，或尝试重新下载模型")
                        raise
                
                self.log("Kronos模型加载完成")

            # 定义Kronos的特征列表
            # 官方模型(kronos-*)用6个基础特征，自定义模型用27个特征
            OFFICIAL_MODEL_FEATURES = [
                "open", "high", "low", "close", "vol", "amt"
            ]
            CUSTOM_MODEL_FEATURES = [
                "open", "high", "low", "close", "vol", "amt",  # 基础数据
                "MA5", "MA10", "MA20",                           # 移动平均线
                "BIAS20",                                          # 乖离率
                "ATR14", "AMPLITUDE",                             # 波动性指标
                "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",         # 成交量指标
                "RSI14", "RSI7",                                  # 动量指标
                "MACD", "MACD_HIST",                              # MACD指标
                "PRICE_SLOPE5", "PRICE_SLOPE10",                  # 趋势指标
                "HIGH5", "LOW5", "HIGH10", "LOW10",              # 极值指标
                "VOL_BREAKOUT", "VOL_SHRINK"                       # 成交量突破
            ]
            
            # 根据模型类型选择特征列表
            # 直接从model_var获取当前选择的模型，确保第一次加载就正确
            model_name = self.model_var.get()
            if model_name.startswith("kronos-"):
                # 官方模型
                FEATURE_LIST = OFFICIAL_MODEL_FEATURES
            else:
                # 自定义模型
                FEATURE_LIST = CUSTOM_MODEL_FEATURES
            
            # 更新kronos_model_name，确保下次切换时正确
            self.kronos_model_name = model_name

            if self.kronos_predictor is None:
                self.kronos_predictor = KronosPredictor(
                    self.kronos_model, self.kronos_tokenizer, max_context=512,
                    feature_list=FEATURE_LIST
                )

            predictor = self.kronos_predictor

            # 更新状态
            self.viz_status_label.config(text="正在生成预测...")
            self.viz_canvas.get_tk_widget().update()

            # 准备数据 - Kronos模型需要正确的列名
            x_df = df.iloc[:lookback].copy()
            
            # 确保有必要的列（兼容不同列名）
            # 处理volume/vol列
            if "volume" in x_df.columns and "vol" not in x_df.columns:
                x_df["vol"] = x_df["volume"]
            elif "vol" in x_df.columns and "volume" not in x_df.columns:
                x_df["volume"] = x_df["vol"]
            elif "vol" not in x_df.columns and "volume" not in x_df.columns:
                # 如果都没有，创建默认值
                x_df["vol"] = x_df["close"] * 100
                x_df["volume"] = x_df["vol"]
            
            # 处理amount/amt列
            if "amount" in x_df.columns and "amt" not in x_df.columns:
                x_df["amt"] = x_df["amount"]
            elif "amt" in x_df.columns and "amount" not in x_df.columns:
                x_df["amount"] = x_df["amt"]
            elif "amt" not in x_df.columns and "amount" not in x_df.columns:
                # 如果都没有，用成交量*平均价格估算
                x_df["amt"] = x_df["vol"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
                x_df["amount"] = x_df["amt"]

            # 转换时间戳为DatetimeIndex（Kronos模型要求）
            x_timestamp = pd.DatetimeIndex(df.iloc[:lookback]["timestamp"])

            # 确定y_timestamp
            if viz_mode == "预测未来":
                # 预测未来模式：直接创建未来时间戳
                last_timestamp = x_timestamp[-1]
                freq = self._get_timeframe_freq(timeframe)
                y_timestamp = pd.date_range(
                    start=last_timestamp + pd.Timedelta(freq),
                    periods=pred_len,
                    freq=freq,
                )
                self.log(f"[预测未来] 预测未来 {pred_len} 个周期")
            else:
                # 回测过去模式：如果有足够的历史数据则使用历史数据，否则创建未来时间戳
                available_future_data = max(0, len(df) - lookback)
                if available_future_data >= pred_len:
                    # 有足够的历史数据作为未来数据
                    y_timestamp = pd.DatetimeIndex(
                        df.iloc[lookback : lookback + pred_len]["timestamp"]
                    )
                    self.log(
                        f"[回测过去] 使用历史数据验证，未来数据长度: {len(y_timestamp)}"
                    )
                elif available_future_data > 0:
                    # 部分历史数据 + 部分未来时间戳
                    historical_part = pd.DatetimeIndex(
                        df.iloc[lookback : lookback + available_future_data][
                            "timestamp"
                        ]
                    )
                    last_timestamp = (
                        historical_part[-1]
                        if len(historical_part) > 0
                        else x_timestamp[-1]
                    )

                    # 确定频率
                    freq = self._get_timeframe_freq(timeframe)
                    future_periods = pred_len - available_future_data
                    future_part = pd.date_range(
                        start=last_timestamp + pd.Timedelta(freq),
                        periods=future_periods,
                        freq=freq,
                    )
                    # 合并历史数据和未来时间戳
                    y_timestamp = pd.DatetimeIndex(
                        list(historical_part) + list(future_part)
                    )
                    self.log(
                        f"[回测过去] 混合模式: 历史{len(historical_part)} + 未来{len(future_part)}"
                    )
                else:
                    # 完全没有历史数据，全部创建未来时间戳
                    last_timestamp = x_timestamp[-1]
                    freq = self._get_timeframe_freq(timeframe)
                    y_timestamp = pd.date_range(
                        start=last_timestamp + pd.Timedelta(freq),
                        periods=pred_len,
                        freq=freq,
                    )
                    self.log(f"[回测过去] 无未来历史数据，使用纯预测")

            # 基本调试信息
            self.log(f"时间戳信息: x长度={len(x_timestamp)}, y长度={len(y_timestamp)}")
            self.log(
                f"时间范围: x[{x_timestamp[0]}]到[{x_timestamp[-1]}], y[{y_timestamp[0]}]到[{y_timestamp[-1]}]"
            )

            # 生成预测前先计算技术指标特征
            x_df_with_features = self._calculate_kronos_features(x_df, FEATURE_LIST)

            # 选择Kronos需要的列
            kronos_columns = FEATURE_LIST

            # 生成预测
            pred_df = predictor.predict(
                df=x_df_with_features[kronos_columns],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

            # 更新状态
            self.viz_status_label.config(text="正在绘制图表...")
            self.viz_canvas.get_tk_widget().update()

            # 清空之前的图表
            self.viz_figure.clear()

            # 创建子图
            ax1 = self.viz_figure.add_subplot(2, 1, 1)
            ax2 = self.viz_figure.add_subplot(2, 1, 2, sharex=ax1)

            # 准备时间轴数据
            if viz_mode == "预测未来":
                # 预测未来模式：只显示历史数据 + 纯预测
                history_timestamps = pd.DatetimeIndex(df["timestamp"])
                history_prices = df["close"].values
                history_volumes = df["volume"].values

                # 预测数据时间戳（y_timestamp）
                pred_timestamps = y_timestamp[: len(pred_df)]  # 确保长度匹配
                pred_prices = pred_df["close"].values
                pred_volumes = pred_df["volume"].values

                # 绘制收盘价
                ax1.plot(
                    history_timestamps,
                    history_prices,
                    label="历史价格",
                    color="blue",
                    linewidth=1.5
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="预测未来",
                    color="red",
                    linewidth=1.5,
                    linestyle="--"
                )
                ax1.set_ylabel("收盘价 (USDT)", fontsize=12)
                ax1.legend(loc="upper left", fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_title(
                    f"Kronos预测未来 - {symbol} ({timeframe})",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                # 回测过去模式：历史数据（包含未来验证数据）
                history_start = max(0, len(df) - lookback - pred_len)
                history_timestamps = pd.DatetimeIndex(
                    df["timestamp"].iloc[history_start:]
                )
                history_prices = df["close"].iloc[history_start:].values
                history_volumes = df["volume"].iloc[history_start:].values

                # 预测数据时间戳（y_timestamp）
                pred_timestamps = y_timestamp[: len(pred_df)]  # 确保长度匹配
                pred_prices = pred_df["close"].values
                pred_volumes = pred_df["volume"].values

                # 绘制收盘价
                ax1.plot(
                    history_timestamps,
                    history_prices,
                    label="实际价格",
                    color="blue",
                    linewidth=1.5
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="预测价格",
                    color="red",
                    linewidth=1.5,
                    linestyle="--"
                )
                ax1.set_ylabel("收盘价 (USDT)", fontsize=12)
                ax1.legend(loc="upper left", fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_title(
                    f"Kronos回测过去 - {symbol} ({timeframe})",
                    fontsize=14,
                    fontweight="bold",
                )

            # 绘制成交量
            if viz_mode == "预测未来":
                ax2.plot(
                    history_timestamps,
                    history_volumes,
                    label="历史成交量",
                    color="blue",
                    linewidth=1.5,
                )
                ax2.plot(
                    pred_timestamps,
                    pred_volumes,
                    label="预测成交量",
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                )
            else:
                ax2.plot(
                    history_timestamps,
                    history_volumes,
                    label="实际成交量",
                    color="blue",
                    linewidth=1.5,
                )
                ax2.plot(
                    pred_timestamps,
                    pred_volumes,
                    label="预测成交量",
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                )
            ax2.set_ylabel("成交量", fontsize=12)
            ax2.set_xlabel("时间", fontsize=12)
            ax2.legend(loc="upper left", fontsize=10)
            ax2.grid(True, alpha=0.3)

            # 调整布局
            self.viz_figure.tight_layout()

            # 更新画布
            self.viz_canvas.draw()

            # 恢复按钮状态
            self.viz_button.config(state=tk.NORMAL)
            if viz_mode == "预测未来":
                self.viz_status_label.config(
                    text=f"[预测未来] 可视化完成！预测长度: {pred_len}周期"
                )
                # 更新左侧小图表
                self._update_small_prediction_chart(history_timestamps, history_prices, 
                                                    pred_timestamps, pred_prices, symbol, timeframe)
            else:
                self.viz_status_label.config(
                    text=f"[回测过去] 可视化完成！预测长度: {pred_len}周期"
                )

        except Exception as e:
            # 错误处理
            self.viz_status_label.config(text=f"错误: {str(e)}")
            self.viz_button.config(state=tk.NORMAL)
            import traceback

            traceback.print_exc()
            # 将错误信息输出到日志
            self.log(f"可视化生成错误: {str(e)}")

    def _draw_prediction_chart(self):
        """绘制Kronos预测K线走势图表"""
        try:
            if self.prediction_ax is None:
                return

            # 清空图表
            self.prediction_ax.clear()

            data = self.kronos_analysis_data

            if data.get('history_prices') is None or len(data.get('history_prices', [])) == 0:
                # 没有分析数据时显示等待状态
                self.prediction_ax.text(0.5, 0.5, '等待Kronos分析...', 
                                        ha='center', va='center', 
                                        fontsize=14, color='gray', 
                                        transform=self.prediction_ax.transAxes)
                self.prediction_ax.set_title('Kronos预测走势', fontsize=11, fontweight='bold')
                self.prediction_ax.axis('off')
            else:
                # 有分析数据，绘制历史和预测价格
                self.prediction_ax.set_facecolor('#f8f9fa')
                
                history_timestamps = data.get('history_timestamps', [])
                history_prices = data.get('history_prices', [])
                pred_timestamps = data.get('pred_timestamps', [])
                pred_prices = data.get('pred_prices', [])
                trend_direction = data.get('trend_direction', 'NEUTRAL')
                
                # 绘制历史价格
                self.prediction_ax.plot(history_timestamps, history_prices, 
                                        label='历史', color='#3498db', linewidth=1.0)
                
                # 绘制预测价格
                self.prediction_ax.plot(pred_timestamps, pred_prices, 
                                        label='预测', color='#e74c3c', linewidth=1.0)
                
                # 设置标题和标签
                direction_text = '↑ 上涨' if trend_direction == 'LONG' else '↓ 下跌'
                self.prediction_ax.set_title(f'Kronos预测走势 ({direction_text})', fontsize=11, fontweight='bold')
                
                # 隐藏X/Y轴的数字和标签
                self.prediction_ax.set_xticks([])
                self.prediction_ax.set_yticks([])
                
                # 隐藏所有边框
                self.prediction_ax.spines['top'].set_visible(False)
                self.prediction_ax.spines['right'].set_visible(False)
                self.prediction_ax.spines['left'].set_visible(False)
                self.prediction_ax.spines['bottom'].set_visible(False)
                
                # 只显示网格
                self.prediction_ax.grid(True, alpha=0.3)
                
            # 更新画布（使用更大的tight_layout来最大化图表显示面积）
            self.prediction_fig.tight_layout(pad=1.0)
            if hasattr(self, 'prediction_canvas') and self.prediction_canvas:
                self.prediction_canvas.draw()

        except Exception as e:
            print(f"绘制预测图表失败: {e}")
            import traceback
            traceback.print_exc()

    def update_kronos_analysis(self, history_timestamps=None, history_prices=None, 
                               pred_timestamps=None, pred_prices=None,
                               trend_direction='NEUTRAL', trend_strength=0, 
                               pred_change=0, threshold=0.008):
        """更新Kronos分析数据并刷新图表
        
        Args:
            history_timestamps: 历史时间戳数组
            history_prices: 历史价格数组
            pred_timestamps: 预测时间戳数组
            pred_prices: 预测价格数组
            trend_direction: 'LONG' 或 'SHORT'
            trend_strength: 趋势强度值
            pred_change: 预测变化值 (小数)
            threshold: 阈值
        """
        try:
            if not hasattr(self, 'kronos_analysis_data') or self.kronos_analysis_data is None:
                return

            from datetime import datetime
            
            self.kronos_analysis_data = {
                'history_timestamps': history_timestamps,
                'history_prices': history_prices,
                'pred_timestamps': pred_timestamps,
                'pred_prices': pred_prices,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'pred_change': pred_change,
                'threshold': threshold,
                'timestamp': datetime.now()
            }
            
            # 刷新图表
            self._draw_prediction_chart()
            
        except Exception as e:
            print(f"更新Kronos分析数据失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_small_prediction_chart(self, history_timestamps, history_prices, 
                                       pred_timestamps, pred_prices, symbol, timeframe):
        """更新左侧的Kronos预测小图表（保留旧方法兼容性）"""
        pass

    def _get_timeframe_freq(self, timeframe):
        """将时间周期字符串转换为pandas频率字符串"""
        if timeframe == "1m":
            return "1T"
        elif timeframe == "3m":
            return "3T"
        elif timeframe == "5m":
            return "5T"
        elif timeframe == "15m":
            return "15T"
        elif timeframe == "30m":
            return "30T"
        elif timeframe == "1h":
            return "1H"
        elif timeframe == "4h":
            return "4H"
        elif timeframe == "1d":
            return "1D"
        else:
            return "5T"  # 默认5分钟

    def _calculate_kronos_features(self, df, feature_list):
        """计算Kronos预测所需的特征 - 根据feature_list决定计算哪些"""
        import numpy as np
        
        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # 处理成交量列 - 兼容volume/vol
        if "volume" in df.columns and "vol" not in df.columns:
            df["vol"] = df["volume"]
        elif "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]
        elif "vol" not in df.columns and "volume" not in df.columns:
            df["vol"] = close * 100
            df["volume"] = df["vol"]
        
        # 处理成交额列 - 兼容不同数据源
        if "amount" in df.columns:
            amount = df["amount"].values
        elif "amt" in df.columns:
            amount = df["amt"].values
        elif "volume" in df.columns:
            amount = df["volume"].values
        elif "quote_asset_volume" in df.columns:
            amount = df["quote_asset_volume"].values
        else:
            amount = close * 100
        
        # 确保amt列存在
        if "amt" not in df.columns:
            df["amt"] = amount
        if "amount" not in df.columns:
            df["amount"] = amount
        
        # 检查是否需要计算技术指标
        needs_tech_indicators = any(
            col in feature_list 
            for col in ["MA5", "MA10", "MA20", "BIAS20", "ATR14", "AMPLITUDE", 
                       "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO", "RSI14", "RSI7", 
                       "MACD", "MACD_HIST", "PRICE_SLOPE5", "PRICE_SLOPE10", 
                       "HIGH5", "LOW5", "HIGH10", "LOW10", "VOL_BREAKOUT", "VOL_SHRINK"]
        )
        
        if needs_tech_indicators:
            df["MA5"] = pd.Series(close).rolling(window=5).mean().values
            df["MA10"] = pd.Series(close).rolling(window=10).mean().values
            df["MA20"] = pd.Series(close).rolling(window=20).mean().values

            df["BIAS20"] = (close / df["MA20"] - 1) * 100

            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = 0
            df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values

            df["AMPLITUDE"] = (high - low) / close * 100

            df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
            df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values

            df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]

            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain14 = gain.rolling(window=14).mean()
            avg_loss14 = loss.rolling(window=14).mean()
            rs14 = avg_gain14 / avg_loss14
            df["RSI14"] = (100 - (100 / (1 + rs14))).values

            avg_gain7 = gain.rolling(window=7).mean()
            avg_loss7 = loss.rolling(window=7).mean()
            rs7 = avg_gain7 / avg_loss7
            df["RSI7"] = (100 - (100 / (1 + rs7))).values

            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

            df["PRICE_SLOPE5"] = (
                pd.Series(close)
                .rolling(window=5)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True,
                )
                .values
            )
            df["PRICE_SLOPE10"] = (
                pd.Series(close)
                .rolling(window=10)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True,
                )
                .values
            )

            df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
            df["LOW5"] = pd.Series(low).rolling(window=5).min().values
            df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
            df["LOW10"] = pd.Series(low).rolling(window=10).min().values

            df["VOL_BREAKOUT"] = (
                amount > df["AMOUNT_MA5"] * 1.5
            ).astype(int).values
            df["VOL_SHRINK"] = (
                amount < df["AMOUNT_MA5"] * 0.5
            ).astype(int).values
        
        # 清理 NaN 值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

    def load_settings(self):
        try:
            from dotenv import load_dotenv

            # 策略名称映射
            strategy_name_map = {
                "trend": "趋势爆发",
                "range": "震荡套利",
                "breakout": "消息突破",
                "auto": "自动策略",
                "time": "时间策略",
            }

            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

            # 使用UTF-8编码加载，避免Windows编码问题
            load_dotenv(env_path, encoding="utf-8")

            api_key = os.getenv("BINANCE_API_KEY", "")
            secret_key = os.getenv("BINANCE_API_SECRET", "")
            os.getenv("THRESHOLD", "0.008")
            leverage = os.getenv("LEVERAGE", "10")
            os.getenv("CHECK_INTERVAL", "120")
            os.getenv("TIMEFRAME", "5m")
            symbol = os.getenv("SYMBOL", "BTCUSDT")
            strategy_code = os.getenv("STRATEGY", "auto")
            model = os.getenv("MODEL", "kronos-small")
            min_position = os.getenv("MIN_POSITION", "100")
            os.getenv("AI_MIN_TREND", "0.010")
            os.getenv("AI_MIN_DEVIATION", "0.008")
            os.getenv("MAX_FUNDING", "1.0")
            os.getenv("MIN_FUNDING", "-1.0")

            # 转换策略代码为显示名称
            strategy = strategy_name_map.get(strategy_code, "趋势爆发")

            if api_key:
                self.api_key_entry.insert(0, api_key)
            if secret_key:
                self.secret_key_entry.insert(0, secret_key)
            self.leverage_var.set(leverage)
            self.symbol_var.set(symbol)
            self.strategy_var.set(strategy)
            self.model_var.set(model)
            self.min_position_var.set(min_position)

            # 根据策略加载默认参数（忽略.env中的策略参数，使用内置的稳健参数）
            strategy_code_map = {
                "趋势爆发": "trend",
                "震荡套利": "range",
                "消息突破": "breakout",
                "自动策略": "auto",
                "时间策略": "time",
            }
            strategy_key = strategy_code_map.get(strategy, "auto")
            self.reset_to_strategy_defaults(strategy_key)
            
            # 加载确认次数设置
            self._load_confirm_counts_settings()
        except Exception as e:
            print(f"加载配置失败: {e}")

    def save_settings(self):
        try:
            # 策略名称映射
            strategy_code_map = {
                "趋势爆发": "trend",
                "震荡套利": "range",
                "消息突破": "breakout",
            }

            api_key = self.api_key_entry.get().strip()
            secret_key = self.secret_key_entry.get().strip()
            threshold = self.threshold_var.get()
            leverage = self.leverage_var.get()
            interval = self.interval_var.get()
            timeframe = self.timeframe_var.get()
            symbol = self.symbol_var.get()
            strategy_display = self.strategy_var.get()
            model = self.model_var.get()
            min_position = self.min_position_var.get()
            ai_min_trend = self.ai_min_trend_var.get()
            ai_min_deviation = self.ai_min_deviation_var.get()
            max_funding = self.max_funding_var.get()
            min_funding = self.min_funding_var.get()

            # 转换为策略代码（避免中文编码问题）
            strategy_code = strategy_code_map.get(strategy_display, "trend")

            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

            env_vars = {}
            if os.path.exists(env_path):
                try:
                    # 使用UTF-8编码读取
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if "=" in line and not line.startswith("#"):
                                key, val = line.split("=", 1)
                                env_vars[key] = val
                except UnicodeDecodeError:
                    try:
                        # 如果UTF-8失败，尝试GBK（Windows默认）
                        with open(env_path, "r", encoding="gbk") as f:
                            for line in f:
                                line = line.strip()
                                if "=" in line and not line.startswith("#"):
                                    key, val = line.split("=", 1)
                                    env_vars[key] = val
                    except Exception as e:
                        print(f"读取旧配置文件失败: {e}")

            env_vars["BINANCE_API_KEY"] = api_key
            env_vars["BINANCE_API_SECRET"] = secret_key
            env_vars["THRESHOLD"] = threshold
            env_vars["LEVERAGE"] = leverage
            env_vars["CHECK_INTERVAL"] = interval
            env_vars["TIMEFRAME"] = timeframe
            env_vars["SYMBOL"] = symbol
            env_vars["STRATEGY"] = strategy_code
            env_vars["MODEL"] = model
            env_vars["MIN_POSITION"] = min_position
            env_vars["AI_MIN_TREND"] = ai_min_trend
            env_vars["AI_MIN_DEVIATION"] = ai_min_deviation
            env_vars["MAX_FUNDING"] = max_funding
            env_vars["MIN_FUNDING"] = min_funding

            # 使用UTF-8编码保存
            with open(env_path, "w", encoding="utf-8") as f:
                for key, val in env_vars.items():
                    f.write(f"{key}={val}\n")

            self.log(
                f"参数已保存: 阈值={threshold}, 杠杆={leverage}x, 周期={timeframe}, 间隔={interval}秒, AI趋势={ai_min_trend}, AI偏离={ai_min_deviation}, 资金费率=[{min_funding}%, {max_funding}%]"
            )
        except Exception as e:
            self.log(f"保存配置失败: {e}")

    def on_strategy_changed(self, event=None):
        strategy_map = {
            "趋势爆发": "trend",
            "震荡套利": "range",
            "消息突破": "breakout",
            "自动策略": "auto",
            "时间策略": "time",
        }
        strategy_key = strategy_map.get(self.strategy_var.get(), "trend")
        self.reset_to_strategy_defaults(strategy_key)

    def reset_to_strategy_defaults(self, strategy_key):
        strategy_params = {
            "trend": {
                "name": "趋势爆发",
                "threshold": "0.008",
                "ai_min_trend": "0.010",
                "ai_min_deviation": "0.008",
                "max_funding": "1.0",
                "min_funding": "-1.0",
                "timeframe": "5m",
                "interval": "120",
            },
            "range": {
                "name": "震荡套利",
                "threshold": "0.005",
                "ai_min_trend": "0.003",
                "ai_min_deviation": "0.005",
                "max_funding": "2.0",
                "min_funding": "-2.0",
                "timeframe": "5m",
                "interval": "60",
            },
            "breakout": {
                "name": "消息突破",
                "threshold": "0.015",
                "ai_min_trend": "0.020",
                "ai_min_deviation": "0.015",
                "max_funding": "3.0",
                "min_funding": "-3.0",
                "timeframe": "5m",
                "interval": "180",
            },
            "auto": {
                "name": "自动策略",
                "threshold": "0.0047",
                "ai_min_trend": "0.0047",
                "ai_min_deviation": "0.005",
                "max_funding": "3.0",
                "min_funding": "-3.0",
                "timeframe": "5m",
                "interval": "180",
            },
            "time": {
                "name": "时间策略",
                "threshold": "0.006",
                "ai_min_trend": "0.006",
                "ai_min_deviation": "0.006",
                "max_funding": "3.5",
                "min_funding": "-3.5",
                "timeframe": "5m",
                "interval": "180",
            },
        }

        params = strategy_params.get(strategy_key, strategy_params["trend"])

        self.threshold_var.set(params["threshold"])
        self.ai_min_trend_var.set(params["ai_min_trend"])
        self.ai_min_deviation_var.set(params["ai_min_deviation"])
        self.max_funding_var.set(params["max_funding"])
        self.min_funding_var.set(params["min_funding"])
        self.timeframe_var.set(params["timeframe"])
        self.interval_var.set(params["interval"])

        self.log(f"已切换策略: {params['name']}")
        self.log(
            f"参数已自动加载 - 阈值: {params['threshold']}, AI趋势: {params['ai_min_trend']}, AI偏离: {params['ai_min_deviation']}"
        )
        self.log(
            f"资金费率: [{params['min_funding']}%, {params['max_funding']}%], 周期: {params['timeframe']}, 间隔: {params['interval']}秒"
        )

    def toggle_api_visibility(self):
        show = "*" if not self.show_api_var.get() else ""
        self.api_key_entry.config(show=show)
        self.secret_key_entry.config(show=show)

    def log(self, message):
        self.queue.put(("log", message))
        # 同时写入交易日志文件
        if hasattr(self, "trade_logger"):
            self.trade_logger.info(message)

    def start_interval_timer(self, interval_seconds):
        """启动交易间隔计时器
        
        Args:
            interval_seconds: 交易间隔秒数
        """
        # 停止现有计时器
        self.stop_interval_timer()
        
        # 设置间隔参数
        self.interval_seconds = interval_seconds
        self.remaining_seconds = interval_seconds
        self.is_interval_running = True
        
        # 更新UI显示
        self.queue.put(("progress", 0))  # 初始进度0%（递增进度条）
        self.queue.put(("status", f"交易间隔: {interval_seconds}秒"))
        
        # 更新间隔标签
        if hasattr(self, 'interval_label'):
            self.interval_label.config(text=f"交易间隔: {interval_seconds}秒")
            self.remaining_time_var.set(f"{interval_seconds} 秒")
        
        # 启动计时器
        self._update_interval_timer()

    def stop_interval_timer(self):
        """停止交易间隔计时器"""
        self.is_interval_running = False
        if self.interval_timer:
            self.root.after_cancel(self.interval_timer)
            self.interval_timer = None
        
        # 重置UI显示
        self.queue.put(("progress", 0))
        self.queue.put(("status", "空闲"))
        
        if hasattr(self, 'interval_label'):
            self.interval_label.config(text="交易间隔: 空闲")
            self.remaining_time_var.set("-- 秒")

    def _update_interval_timer(self):
        """更新间隔计时器（内部方法）"""
        if not self.is_interval_running:
            return
        
        # 减少剩余时间
        self.remaining_seconds -= 1
        
        # 计算进度百分比（递增：从0%到100%）
        if self.interval_seconds > 0:
            # 已过时间 = 总间隔 - 剩余时间
            elapsed_seconds = self.interval_seconds - self.remaining_seconds
            progress_percent = (elapsed_seconds / self.interval_seconds) * 100
        else:
            progress_percent = 0
        
        # 更新UI
        self.queue.put(("progress", progress_percent))
        self.queue.put(("status", f"下次分析: {self.remaining_seconds}秒"))
        
        # 更新间隔标签和剩余时间
        if hasattr(self, 'interval_label'):
            self.interval_label.config(text=f"交易间隔: {self.interval_seconds}秒")
            self.remaining_time_var.set(f"{self.remaining_seconds} 秒")
        
        # 如果时间到，重启计时器
        if self.remaining_seconds > 0:
            # 每1000ms（1秒）更新一次
            self.interval_timer = self.root.after(1000, self._update_interval_timer)
        else:
            # 时间到，输出消息并准备重置
            self.log(f"交易间隔结束，开始新一轮分析...")
            # 进度保持100%一秒，然后重置
            self.root.after(1000, self._reset_interval_timer)

    def _reset_interval_timer(self):
        """重置间隔计时器"""
        if not self.is_interval_running:
            return
        
        # 重置剩余时间
        self.remaining_seconds = self.interval_seconds
        
        # 重置进度为0%
        self.queue.put(("progress", 0))
        
        # 继续计时器
        if self.is_interval_running:
            self.interval_timer = self.root.after(1000, self._update_interval_timer)

    def set_progress(self, value):
        self.queue.put(("progress", value))
        # 调试日志
        # self.queue.put(("log", f"[调试] 设置进度: {value}"))

    def set_status(self, status):
        self.queue.put(("status", status))

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()

                if msg_type == "log":
                    # 显示在终端中
                    self.terminal.insert(tk.END, f"[{self.get_time()}] {data}\n")
                    # 只有当滚动条在最底部时才自动跟随
                    scroll_position = self.terminal.yview()
                    if scroll_position[1] >= 0.99:
                        self.terminal.see(tk.END)
                    
                    # 检测是否是多智能体系统的日志，如果是，也显示在AI策略中心和实盘监控日志中
                    multi_agent_keywords = [
                        "[多智能体系统]", "[FinGPT]", "[策略协调器]", 
                        "Kronos分析", "舆情分析", "信号过滤", "协调器",
                        "市场情绪", "风险等级", "交易建议", "CoinGecko", "币安"
                    ]
                    
                    exclude_keywords = [
                        "[Qwen新闻处理器]", "[社交媒体情绪]", "qwen_news_processor", 
                        "social_sentiment"
                    ]
                    
                    is_multi_agent_log = any(keyword in data for keyword in multi_agent_keywords)
                    is_excluded_log = any(keyword in data for keyword in exclude_keywords)
                    
                    if is_multi_agent_log and not is_excluded_log:
                        # 确定日志级别
                        level = "INFO"
                        if any(keyword in data for keyword in ["✓", "成功", "完成"]):
                            level = "SUCCESS"
                        elif any(keyword in data for keyword in ["⚠", "警告", "过滤", "风险"]):
                            level = "WARNING"
                        elif any(keyword in data for keyword in ["✗", "错误", "失败"]):
                            level = "ERROR"
                        
                        # 记录到AI策略中心和实盘监控日志
                        self._log_live_message(data, level)
                elif msg_type == "progress":
                    self.progress_var.set(data)
                    # 调试日志
                    # self.status_label.config(text=f"进度: {data}%")
                elif msg_type == "status":
                    self.status_label.config(text=data)
                elif msg_type == "kronos_analysis":
                    # 更新Kronos趋势分析图表
                    self.update_kronos_analysis(**data)

        except queue.Empty:
            pass

        self.root.after(100, self.process_queue)

    def start_trading(self):
        self.save_settings()

        api_key = self.api_key_entry.get().strip()
        secret_key = self.secret_key_entry.get().strip()

        if not api_key or not secret_key:
            try:
                from dotenv import load_dotenv

                env_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), ".env"
                )
                load_dotenv(env_path, encoding="utf-8")
                api_key = os.getenv("BINANCE_API_KEY", "")
                secret_key = os.getenv("BINANCE_API_SECRET", "")
            except:
                pass

        if not api_key or not secret_key:
            messagebox.showerror("错误", "请输入API Key和Secret Key")
            return

        # 确保前一个线程已完全停止
        if self.trading_thread is not None and self.trading_thread.is_alive():
            self.log("等待前一个交易线程完全停止...")
            if self.stop_event is not None:
                self.stop_event.set()
            self.trading_thread.join(timeout=5)
            if self.trading_thread.is_alive():
                self.log("警告: 前一个线程未能在5秒内停止，强制启动新线程")

        os.environ["BINANCE_API_KEY"] = api_key
        os.environ["BINANCE_API_SECRET"] = secret_key

        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.trading_thread = threading.Thread(target=self.run_trading, daemon=True)
        self.trading_thread.start()
        
        self._start_balance_recording()

        self.log("=" * 60)
        self.log("交易系统启动")
        self.log("=" * 60)

    def stop_trading(self):
        self._stop_balance_recording()
        if self.stop_event is not None:
            self.stop_event.set()

        self.log("正在停止交易...")

        # 恢复默认参数
        if hasattr(self, 'strategy') and self.strategy is not None:
            self.log("恢复默认参数...")
            try:
                self.strategy.restore_default_parameters()
            except Exception as e:
                self.log(f"恢复默认参数时出错: {e}")

        # 等待线程真正结束（最多等待10秒）
        if self.trading_thread is not None and self.trading_thread.is_alive():
            self.log("等待交易线程完全停止...")
            try:
                self.trading_thread.join(timeout=10)
                if self.trading_thread.is_alive():
                    self.log("警告: 交易线程仍在运行，强制终止")
                    # 强制终止 - 设置标志
                    self._force_stop_trading = True
            except Exception as e:
                self.log(f"等待线程停止时出错: {e}")

        # 恢复标准输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        self.stop_interval_timer()
        self.stop_loss_check_timer()
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.set_status("已停止")
        self.log("交易系统已停止")

    def run_trading(self):
        self.set_progress(0)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector

        try:
            self.log("正在初始化交易系统...")
            self.set_status("加载模型...")

            from binance_api import BinanceAPI
            from professional_strategy import ProfessionalTradingStrategy

            symbol = self.symbol_var.get()
            model_name = self.model_var.get()
            timeframe = self.timeframe_var.get()
            leverage = int(self.leverage_var.get())
            interval = int(self.interval_var.get())
            min_position = float(self.min_position_var.get())
            ai_min_trend = float(self.ai_min_trend_var.get())
            ai_min_deviation = float(self.ai_min_deviation_var.get())
            max_funding = float(self.max_funding_var.get())
            min_funding = float(self.min_funding_var.get())

            strategy_map = {
                "趋势爆发": "trend",
                "震荡套利": "range",
                "消息突破": "breakout",
                "自动策略": "auto",
                "时间策略": "time",
            }
            strategy_type = strategy_map.get(self.strategy_var.get(), "trend")
            
            # 5套交易策略强制使用5m时间周期（因为模型训练数据是5m）
            is_professional_strategy = self.strategy_var.get() in strategy_map
            if is_professional_strategy:
                if timeframe != "5m":
                    self.log(f"⚠️ 检测到选择了{self.strategy_var.get()}策略，强制使用5m时间周期")
                    timeframe = "5m"
                    self.timeframe_var.set("5m")

            self.log(f"交易对: {symbol}")
            self.log(f"策略: {self.strategy_var.get()}")
            self.log(f"模型: {model_name}")
            self.log(f"周期: {timeframe}")
            self.log(f"杠杆: {leverage}x")
            self.log(f"最小仓位: ${min_position:.2f}")
            self.log(f"间隔: {interval}秒")
            self.log(f"阈值: {self.threshold_var.get()}")
            self.log(f"AI趋势最小: {ai_min_trend}")
            self.log(f"AI偏离最小: {ai_min_deviation}")
            self.log(f"资金费率: [{min_funding}%, {max_funding}%]")

            BinanceAPI()

            # 获取完整的AI策略配置
            ai_strategy_config = None
            if hasattr(self, "_get_ai_strategy_config_from_ui"):
                try:
                    ai_strategy_config = self._get_ai_strategy_config_from_ui()
                    self.log(f"✓ 已获取AI策略配置")
                except Exception as e:
                    self.log(f"⚠ 获取AI策略配置失败: {e}")

            # 定义线程安全的Kronos分析回调函数
            def analysis_callback(history_timestamps=None, history_prices=None, 
                                  pred_timestamps=None, pred_prices=None,
                                  trend_direction='NEUTRAL', trend_strength=0, 
                                  pred_change=0, threshold=0.008):
                self.queue.put(("kronos_analysis", {
                    'history_timestamps': history_timestamps,
                    'history_prices': history_prices,
                    'pred_timestamps': pred_timestamps,
                    'pred_prices': pred_prices,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'pred_change': pred_change,
                    'threshold': threshold
                }))
            
            self.log("正在创建策略...")
            self.strategy = ProfessionalTradingStrategy(
                symbol=symbol,
                leverage=leverage,
                interval=interval,
                model_name=model_name,
                timeframe=timeframe,
                threshold=float(self.threshold_var.get()),
                strategy_type=strategy_type,
                min_position=min_position,
                ai_min_trend=ai_min_trend,
                ai_min_deviation=ai_min_deviation,
                max_funding=max_funding,
                min_funding=min_funding,
                analysis_callback=analysis_callback,
                strategy_config=ai_strategy_config
            )

            # 设置确认次数和新参数 - 从AI策略配置面板读取
            try:
                # 从AI策略配置面板读取最新参数
                if hasattr(self, "ai_strategy_config_vars"):
                    entry_count = int(self.ai_strategy_config_vars.get("strategy.entry_confirm_count", self.entry_confirm_count_var).get())
                    reverse_count = int(self.ai_strategy_config_vars.get("strategy.reverse_confirm_count", self.reverse_confirm_count_var).get())
                    consecutive_pred = int(self.ai_strategy_config_vars.get("strategy.require_consecutive_prediction", self.require_consecutive_prediction_var).get())
                    post_entry_hours = float(self.ai_strategy_config_vars.get("strategy.post_entry_hours", self.post_entry_hours_var).get())
                    take_profit_min_pct = float(self.ai_strategy_config_vars.get("strategy.take_profit_min_pct", self.take_profit_min_pct_var).get())
                else:
                    # 降级到旧变量
                    entry_count = int(self.entry_confirm_count_var.get())
                    reverse_count = int(self.reverse_confirm_count_var.get())
                    consecutive_pred = int(self.require_consecutive_prediction_var.get())
                    post_entry_hours = float(self.post_entry_hours_var.get())
                    take_profit_min_pct = float(self.take_profit_min_pct_var.get())
                
                # 验证范围
                if 1 <= entry_count <= 10:
                    self.strategy.entry_confirm_count = entry_count
                if 1 <= reverse_count <= 10:
                    self.strategy.reverse_confirm_count = reverse_count
                if 1 <= consecutive_pred <= 10:
                    self.strategy.require_consecutive_prediction = consecutive_pred
                if 0.5 <= post_entry_hours <= 24:
                    self.strategy.post_entry_hours = post_entry_hours
                if 0.1 <= take_profit_min_pct <= 10:
                    self.strategy.take_profit_min_pct = take_profit_min_pct
                    
                self.log(f"参数设置: 开仓{entry_count}次, 平仓{reverse_count}次, 连续预测{consecutive_pred}次, 开仓后计时{post_entry_hours}小时, 最小止盈{take_profit_min_pct}%")
            except Exception as e:
                self.log(f"设置参数失败: {e}, 使用默认值")

            # 设置性能监控器的策略实例（如果存在）
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                try:
                    self.performance_monitor.set_strategy_instance(self.strategy)
                    self.log("性能监控器已连接到策略实例")
                    
                    # 如果自动化优化已启用但性能监控未启动，现在启动它
                    if self.is_auto_optimization_enabled and not self.performance_monitor.is_monitoring:
                        self.performance_monitor.start_monitoring()
                        self.log("性能监控器已启动")
                        
                except Exception as e:
                    self.log(f"设置性能监控器策略实例失败: {e}")

            # 获取初始资金（合约仓位总资金，包含持仓盈亏）
            initial_balance = self.strategy.binance.get_total_balance()
            self.initial_futures_balance = initial_balance if initial_balance else 0.0
            self.optimization_baseline_balance = self.initial_futures_balance
            self.update_fund_display()
            self.log(f"初始合约资金(总资金): ${self.initial_futures_balance:.2f}")
            self.log(f"AI优化基准资金: ${self.optimization_baseline_balance:.2f}")
            
            # 最后交易时间保持为None，让策略自己管理

            # 启动资金更新定时器
            self.update_fund_timer()
            # 启动亏损检查定时器
            self.start_loss_check_timer()

            self.set_status("运行中...")
            self.start_interval_timer(interval)
            self.log("开始交易循环...")

            self.strategy.run_loop(
                interval_seconds=interval, stop_event=self.stop_event
            )

        except Exception as e:
            self.log(f"错误: {str(e)}")
            import traceback

            self.log(traceback.format_exc())
            self.stop_interval_timer()

        self.set_status("已停止")
        self.set_progress(0)
        self.is_running = False
        self.root.after(0, self.update_button_state)

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def update_button_state(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_fund_display(self):
        """更新资金显示"""
        if hasattr(self, "strategy") and self.strategy:
            current_balance = self.strategy.get_total_balance()
            if current_balance:
                self.current_balance_label.config(text=f"${current_balance:.2f}")

                # 计算盈亏
                if self.initial_futures_balance > 0:
                    pnl = current_balance - self.initial_futures_balance
                    pnl_text = f"${pnl:.2f}"
                    if pnl >= 0:
                        self.pnl_label.config(text=pnl_text, foreground="green")
                        self.pnl_pct_label.config(
                            text=f"+{pnl/self.initial_futures_balance*100:.2f}%",
                            foreground="green",
                        )
                    else:
                        self.pnl_label.config(text=pnl_text, foreground="red")
                        self.pnl_pct_label.config(
                            text=f"{pnl/self.initial_futures_balance*100:.2f}%",
                            foreground="red",
                        )

                self.initial_balance_label.config(
                    text=f"${self.initial_futures_balance:.2f}"
                )

    def update_fund_timer(self):
        """定时更新资金显示"""
        if hasattr(self, "is_running") and self.is_running:
            self.update_fund_display()
            # 每5秒更新一次
            self.root.after(5000, self.update_fund_timer)

    def start_loss_check_timer(self):
        """启动亏损检查定时器（每5分钟检查一次）"""
        if self.is_running:
            self.check_loss_and_optimize()
            self.loss_check_timer = self.root.after(300000, self.start_loss_check_timer)

    def stop_loss_check_timer(self):
        """停止亏损检查定时器"""
        if self.loss_check_timer is not None:
            try:
                self.root.after_cancel(self.loss_check_timer)
            except:
                pass
            self.loss_check_timer = None

    def check_loss_and_optimize(self):
        """检查亏损和无交易时间并在需要时触发AI优化"""
        if not self.is_running or not hasattr(self, "strategy"):
            return

        try:
            from datetime import datetime, timedelta
            now = datetime.now()
            optimize_triggered = False
            optimization_reason = ""
            optimization_details = None
            
            # ===============================
            # 1. 检查亏损
            # ===============================
            current_balance = self.strategy.get_total_balance()
            if current_balance and self.optimization_baseline_balance > 0:
                # 计算从优化基准资金以来的亏损
                loss_pct = ((self.optimization_baseline_balance - current_balance) / self.optimization_baseline_balance) * 100
                
                try:
                    loss_trigger = float(self.loss_trigger_var.get())
                except:
                    loss_trigger = 10.0

                self.log(f"[亏损检查] 基准资金: ${self.optimization_baseline_balance:.2f}, 当前: ${current_balance:.2f}, 亏损: {loss_pct:.2f}%")

                if loss_pct >= loss_trigger:
                    self.log(f"[亏损触发] 亏损超过{loss_trigger}%，启动AI策略优化...")
                    optimize_triggered = True
                    optimization_reason = "loss_trigger"
                    optimization_details = {
                        "loss_pct": loss_pct,
                        "trigger": "亏损触发"
                    }
                else:
                    self.log(f"[亏损检查] 亏损{loss_pct:.2f}% < 阈值{loss_trigger}%，无需优化")
            
            # ===============================
            # 2. 检查无交易时间（12小时）
            # ===============================
            # 检查是否有持仓 - 用于无交易优化
            has_position = False
            if hasattr(self.strategy, "current_position") and self.strategy.current_position:
                has_position = True
            
            if not has_position and self.strategy.last_trade_time:
                # 只有在没有持仓且有过交易记录时，才检查距离最后交易时间
                time_since_last_trade = now - self.strategy.last_trade_time
                hours_since_last_trade = time_since_last_trade.total_seconds() / 3600
                self.log(f"[无交易检查] 距最后交易: {hours_since_last_trade:.1f}小时 (阈值: 12小时)")
                
                if hours_since_last_trade >= 12.0:
                    self.log(f"[无交易触发] 12小时无交易，启动AI策略优化...")
                    optimize_triggered = True
                    optimization_reason = "no_trade"
                    optimization_details = {
                        "hours_since_last_trade": hours_since_last_trade,
                        "last_trade_time": self.strategy.last_trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "trigger": "12小时无交易"
                    }
            
            # ===============================
            # 3. 如果有优化触发，执行优化
            # ===============================
            if optimize_triggered:
                self.trigger_ai_optimization(
                    loss_pct=optimization_details.get("loss_pct", 0.0) if optimization_reason == "loss_trigger" else 0.0,
                    optimization_reason=optimization_reason,
                    optimization_details=optimization_details
                )

        except Exception as e:
            self.log(f"检查失败: {e}")
            import traceback
            traceback.print_exc()

    def trigger_ai_optimization(self, loss_pct: float = 0.0, optimization_reason: str = None, optimization_details: dict = None):
        """触发AI策略优化
        
        Args:
            loss_pct: 亏损百分比（如果是亏损触发）
            optimization_reason: 优化原因（"loss_trigger", "no_trade", "regular"）
            optimization_details: 优化详细信息字典
        """
        try:
            self.log("正在启动AI策略优化...")
            
            # 如果没有提供优化原因，根据 loss_pct 推断
            if optimization_reason is None:
                optimization_reason = "loss_trigger" if loss_pct > 0 else "regular"
            
            # 先检查是否有多智能体系统
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                # 使用多智能体系统进行优化
                self.log("使用多智能体系统优化参数...")
                # 这里可以添加具体的优化调用
            else:
                # 调用性能监控器的优化功能
                if hasattr(self, "performance_monitor") and self.performance_monitor:
                    self.log("使用性能监控器优化参数...")
                    self.performance_monitor.optimize_strategy()
            
            # 优化后，更新基准资金为当前资金
            current_balance = self.strategy.get_total_balance()
            if current_balance:
                self.optimization_baseline_balance = current_balance
                self.log(f"AI优化完成！新的基准资金: ${self.optimization_baseline_balance:.2f}")
                
        except Exception as e:
            self.log(f"AI优化触发失败: {e}")
            import traceback
            traceback.print_exc()

    def _create_system_tab(self, parent):
        """创建系统监控标签页"""
        # CPU监控框架
        cpu_frame = ttk.LabelFrame(parent, text="CPU", padding=10)
        cpu_frame.pack(fill=tk.X, padx=5, pady=5)

        self.cpu_var = tk.StringVar(value="0%")
        self.cpu_progress = ttk.Progressbar(
            cpu_frame,
            variable=self.cpu_var,
            maximum=100,
            length=200,
            style="blue.Horizontal.TProgressbar",
        )
        self.cpu_progress.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            cpu_frame, textvariable=self.cpu_var, font=("微软雅黑", 14, "bold")
        ).pack()

        self.cpu_count_label = ttk.Label(parent, text=f"核心数: {psutil.cpu_count()}")
        self.cpu_count_label.pack(pady=(0, 10))

        # 内存监控框架
        mem_frame = ttk.LabelFrame(parent, text="内存", padding=10)
        mem_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mem_var = tk.StringVar(value="0%")
        self.mem_progress = ttk.Progressbar(
            mem_frame,
            variable=self.mem_var,
            maximum=100,
            length=200,
            style="orange.Horizontal.TProgressbar",
        )
        self.mem_progress.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            mem_frame, textvariable=self.mem_var, font=("微软雅黑", 14, "bold")
        ).pack()

        self.mem_detail_label = ttk.Label(parent, text="")
        self.mem_detail_label.pack(pady=(0, 10))

        # GPU监控框架
        gpu_frame = ttk.LabelFrame(parent, text="GPU", padding=10)
        gpu_frame.pack(fill=tk.X, padx=5, pady=5)

        self.gpu_var = tk.StringVar(value="检测中...")
        ttk.Label(
            gpu_frame, textvariable=self.gpu_var, font=("微软雅黑", 14, "bold")
        ).pack()

        self.gpu_detail_label = ttk.Label(parent, text="")
        self.gpu_detail_label.pack(pady=(0, 10))

        # 磁盘监控框架
        disk_frame = ttk.LabelFrame(parent, text="磁盘", padding=10)
        disk_frame.pack(fill=tk.X, padx=5, pady=5)

        self.disk_var = tk.StringVar(value="0%")
        self.disk_progress = ttk.Progressbar(
            disk_frame,
            variable=self.disk_var,
            maximum=100,
            length=200,
            style="gray.Horizontal.TProgressbar",
        )
        self.disk_progress.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            disk_frame, textvariable=self.disk_var, font=("微软雅黑", 14, "bold")
        ).pack()

        self.disk_detail_label = ttk.Label(parent, text="")
        self.disk_detail_label.pack(pady=(0, 10))

        # 添加进度条样式
        style = ttk.Style()
        style.configure(
            "blue.Horizontal.TProgressbar",
            troughbackground="#e0e0e0",
            background="#007bff",
            thickness=20,
        )
        style.configure(
            "orange.Horizontal.TProgressbar",
            troughbackground="#e0e0e0",
            background="#ff9500",
            thickness=20,
        )
        style.configure(
            "gray.Horizontal.TProgressbar",
            troughbackground="#e0e0e0",
            background="#6c757d",
            thickness=20,
        )

        # 启动系统监控定时器
        self._update_system_info()
        self._system_timer = self.root.after(2000, self._update_system_info)

    def _update_system_info(self):
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=0.5)
            self.cpu_var.set(f"{cpu_percent:.1f}%")
            self.cpu_progress["value"] = cpu_percent

            # 内存信息
            mem = psutil.virtual_memory()
            self.mem_var.set(f"{mem.percent:.1f}%")
            self.mem_progress["value"] = mem.percent
            self.mem_detail_label.config(
                text=f"已用: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB"
            )

            # GPU信息
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    torch.cuda.memory_reserved() / 1024**3
                    gpu_mem_total = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    gpu_util = (
                        (gpu_mem_allocated / gpu_mem_total) * 100
                        if gpu_mem_total > 0
                        else 0
                    )
                    self.gpu_var.set(f"显存: {gpu_util:.1f}%")
                    self.gpu_detail_label.config(
                        text=f"已用: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.1f}GB"
                    )
                else:
                    self.gpu_var.set("无GPU")
                    self.gpu_detail_label.config(text="CUDA不可用")
            except ImportError:
                self.gpu_var.set("无PyTorch")
                self.gpu_detail_label.config(text="未安装PyTorch")
            except Exception as e:
                self.gpu_var.set("GPU错误")
                self.gpu_detail_label.config(text=str(e)[:30])

            # 磁盘信息
            disk = psutil.disk_usage("C:")
            self.disk_var.set(f"{disk.percent:.1f}%")
            self.disk_progress["value"] = disk.percent
            self.disk_detail_label.config(
                text=f"已用: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB"
            )

        except Exception as e:
            print(f"系统监控更新错误: {e}")

        # 每2秒更新一次
        self._system_timer = self.root.after(2000, self._update_system_info)

    def get_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def _create_training_tab(self, parent):
        """创建训练标签页"""
        # 训练配置框架
        config_frame = ttk.LabelFrame(
            parent, text="训练配置", style="TFrame", padding=10
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # ==================== 数据获取部分 ====================
        data_source_frame = ttk.LabelFrame(
            parent, text="数据获取", style="TFrame", padding=10
        )
        data_source_frame.pack(fill=tk.X, pady=(0, 10))

        # 交易对和周期
        pair_frame = ttk.Frame(data_source_frame)
        pair_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pair_frame, text="交易对:").pack(side=tk.LEFT, padx=(0, 5))
        self.train_symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Label(pair_frame, text="BTCUSDT", font=("TkDefaultFont", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(pair_frame, text="周期:").pack(side=tk.LEFT, padx=(0, 5))
        self.train_timeframe_var = tk.StringVar(value="5m")
        ttk.Combobox(
            pair_frame,
            textvariable=self.train_timeframe_var,
            values=[
                "1m",
                "3m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "4h",
                "6h",
                "8h",
                "12h",
                "1d",
            ],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT)

        # 日期范围
        date_frame = ttk.Frame(data_source_frame)
        date_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(date_frame, text="开始日期:").pack(side=tk.LEFT, padx=(0, 5))
        self.train_start_date_var = tk.StringVar(value="")
        ttk.Entry(date_frame, textvariable=self.train_start_date_var, width=12).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(date_frame, text="结束日期:").pack(side=tk.LEFT, padx=(0, 5))
        self.train_end_date_var = tk.StringVar(value="")
        ttk.Entry(date_frame, textvariable=self.train_end_date_var, width=12).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(date_frame, text="(格式: YYYY-MM-DD，留空获取全部)").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # 下载按钮和天数
        download_frame = ttk.Frame(data_source_frame)
        download_frame.pack(fill=tk.X, pady=(0, 5))
        self.download_data_button = ttk.Button(
            download_frame, text="下载币安数据", command=self.download_binance_data
        )
        self.download_data_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(download_frame, text="下载天数:").pack(side=tk.LEFT, padx=(20, 5))
        self.train_days_var = tk.StringVar(value="30")
        ttk.Entry(download_frame, textvariable=self.train_days_var, width=6).pack(
            side=tk.LEFT
        )
        ttk.Label(download_frame, text="天").pack(side=tk.LEFT)

        # ==================== 训练配置部分 ====================
        train_config_frame = ttk.LabelFrame(
            parent, text="模型训练", style="TFrame", padding=10
        )
        train_config_frame.pack(fill=tk.X, pady=(0, 10))

        # 模型选择和GPU选项
        model_frame = ttk.Frame(train_config_frame)
        model_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(model_frame, text="基模型:").pack(side=tk.LEFT, padx=(0, 10))
        self.train_base_model_var = tk.StringVar(value="Kronos-small")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.train_base_model_var,
            values=["Kronos-mini", "Kronos-small", "Kronos-base"],
            width=15,
            state="readonly",
        )
        model_combo.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="  设备:").pack(side=tk.LEFT, padx=(20, 5))
        self.train_device_var = tk.StringVar(value="GPU (CUDA)")
        device_combo = ttk.Combobox(
            model_frame,
            textvariable=self.train_device_var,
            values=["GPU (CUDA)", "CPU"],
            width=10,
            state="readonly",
        )
        device_combo.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="  (GPU更快，有Nvidia显卡选GPU)").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # 训练数据文件
        ttk.Label(train_config_frame, text="训练数据 (CSV):").pack(
            anchor=tk.W, pady=(5, 0)
        )
        data_frame = ttk.Frame(train_config_frame)
        data_frame.pack(fill=tk.X, pady=(0, 5))
        self.train_data_path_var = tk.StringVar(value="training_data/BTCUSDT_5m_with_indicators.csv")
        self.train_data_entry = ttk.Entry(
            data_frame, textvariable=self.train_data_path_var
        )
        self.train_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(data_frame, text="浏览...", command=self.browse_train_data).pack(
            side=tk.RIGHT
        )

        # 配置文件（可选）
        ttk.Label(train_config_frame, text="配置文件 (YAML, 可选):").pack(anchor=tk.W)
        config_file_frame = ttk.Frame(train_config_frame)
        config_file_frame.pack(fill=tk.X, pady=(0, 5))
        self.train_config_path_var = tk.StringVar(
            value="Kronos/finetune_csv/config.yaml"
        )
        self.train_config_entry = ttk.Entry(
            config_file_frame, textvariable=self.train_config_path_var
        )
        self.train_config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(
            config_file_frame, text="浏览...", command=self.browse_train_config
        ).pack(side=tk.RIGHT)

        # 训练参数
        ttk.Label(train_config_frame, text="训练参数:").pack(anchor=tk.W, pady=(5, 0))
        params_frame = ttk.Frame(train_config_frame)
        params_frame.pack(fill=tk.X)

        # 第一行参数
        ttk.Label(params_frame, text="回看窗口:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.train_lookback_var = tk.StringVar(value="256")
        ttk.Entry(params_frame, textvariable=self.train_lookback_var, width=8).grid(
            row=0, column=1, padx=(0, 10)
        )

        ttk.Label(params_frame, text="预测窗口:").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.train_predict_var = tk.StringVar(value="48")
        ttk.Entry(params_frame, textvariable=self.train_predict_var, width=8).grid(
            row=0, column=3, padx=(0, 10)
        )

        ttk.Label(params_frame, text="批次大小:").grid(
            row=0, column=4, sticky=tk.W, padx=(0, 5)
        )
        self.train_batch_var = tk.StringVar(value="64")
        ttk.Entry(params_frame, textvariable=self.train_batch_var, width=8).grid(
            row=0, column=5, padx=(0, 10)
        )

        ttk.Label(params_frame, text="学习率:").grid(
            row=0, column=6, sticky=tk.W, padx=(0, 5)
        )
        self.train_lr_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.train_lr_var, width=10).grid(
            row=0, column=7, padx=(0, 5)
        )

        # 第二行参数
        ttk.Label(params_frame, text="Tokenizer轮数:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_epochs_var = tk.StringVar(value="30")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_epochs_var, width=8
        ).grid(row=1, column=1, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="模型轮数:").grid(
            row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_basemodel_epochs_var = tk.StringVar(value="20")
        ttk.Entry(
            params_frame, textvariable=self.train_basemodel_epochs_var, width=8
        ).grid(row=1, column=3, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="Tokenizer学习率:").grid(
            row=1, column=4, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_lr_var = tk.StringVar(value="0.0002")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_lr_var, width=10
        ).grid(row=1, column=5, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="日志间隔:").grid(
            row=1, column=6, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_log_interval_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.train_log_interval_var, width=8).grid(
            row=1, column=7, padx=(0, 5), pady=(5, 0)
        )

        # 第三行参数
        ttk.Label(params_frame, text="训练集比例:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_ratio_var = tk.StringVar(value="0.9")
        ttk.Entry(params_frame, textvariable=self.train_ratio_var, width=8).grid(
            row=2, column=1, padx=(0, 10), pady=(5, 0)
        )

        ttk.Label(params_frame, text="验证集比例:").grid(
            row=2, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.val_ratio_var = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.val_ratio_var, width=8).grid(
            row=2, column=3, padx=(0, 10), pady=(5, 0)
        )

        ttk.Label(params_frame, text="数据加载线程:").grid(
            row=2, column=4, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_num_workers_var = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.train_num_workers_var, width=8).grid(
            row=2, column=5, padx=(0, 10), pady=(5, 0)
        )

        # 第四行参数 - 实验名称和保存路径
        ttk.Label(params_frame, text="实验名称:").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_exp_name_var = tk.StringVar(value="custom_model")
        ttk.Entry(params_frame, textvariable=self.train_exp_name_var, width=15).grid(
            row=3, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.W
        )

        ttk.Label(params_frame, text="保存路径:").grid(
            row=3, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_save_path_var = tk.StringVar(value="Kronos/finetune_csv/finetuned")
        path_frame = ttk.Frame(params_frame)
        path_frame.grid(
            row=3, column=3, columnspan=2, padx=(0, 5), pady=(5, 0), sticky=tk.W
        )
        ttk.Entry(path_frame, textvariable=self.train_save_path_var, width=18).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(
            path_frame, text="浏览", command=self.browse_save_path, width=5
        ).pack(side=tk.LEFT)

        # 第五行参数 - 模型子文件夹名称
        ttk.Label(params_frame, text="Tokenizer文件夹:").grid(
            row=4, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_folder_var = tk.StringVar(value="tokenizer")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_folder_var, width=12
        ).grid(row=4, column=1, padx=(0, 10), pady=(5, 0), sticky=tk.W)

        ttk.Label(params_frame, text="模型文件夹:").grid(
            row=4, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_basemodel_folder_var = tk.StringVar(value="basemodel")
        ttk.Entry(
            params_frame, textvariable=self.train_basemodel_folder_var, width=12
        ).grid(row=4, column=3, padx=(0, 10), pady=(5, 0), sticky=tk.W)

        ttk.Label(
            params_frame,
            text="(最终模型: {保存路径}/{实验名称}/{模型文件夹}/best_model)",
        ).grid(row=4, column=4, columnspan=4, sticky=tk.W, pady=(5, 0))

        # 复选框选项
        check_frame = ttk.Frame(train_config_frame)
        check_frame.pack(fill=tk.X, pady=(5, 0))
        self.train_tokenizer_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            check_frame, text="训练Tokenizer", variable=self.train_tokenizer_var
        ).pack(side=tk.LEFT, padx=10)
        self.train_basemodel_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            check_frame, text="训练基模型", variable=self.train_basemodel_check_var
        ).pack(side=tk.LEFT, padx=10)
        self.train_skip_existing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            check_frame, text="跳过已存在训练", variable=self.train_skip_existing_var
        ).pack(side=tk.LEFT, padx=10)

        # 训练控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.train_button = ttk.Button(
            control_frame, text="开始训练", command=self.start_training
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.stop_train_button = ttk.Button(
            control_frame,
            text="停止训练",
            command=self.stop_training,
            state=tk.DISABLED,
        )
        self.stop_train_button.pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=15, fill=tk.Y
        )

        self.save_config_button = ttk.Button(
            control_frame, text="保存配置", command=self.save_train_config
        )
        self.save_config_button.pack(side=tk.LEFT, padx=5)

        self.load_config_button = ttk.Button(
            control_frame, text="加载配置", command=self.load_train_config
        )
        self.load_config_button.pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=15, fill=tk.Y
        )

        self.help_button = ttk.Button(
            control_frame, text="帮助", command=self.show_train_help
        )
        self.help_button.pack(side=tk.LEFT, padx=5)

        # 训练进度条
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_bar = ttk.Progressbar(
            parent,
            variable=self.train_progress_var,
            maximum=100,
            mode="determinate",
            style="green.Horizontal.TProgressbar",
        )
        self.train_progress_bar.pack(fill=tk.X, pady=(0, 5))

        # 训练状态标签
        self.train_status_label = ttk.Label(parent, text="就绪", style="TLabel")
        self.train_status_label.pack(anchor=tk.W, pady=(0, 5))

        # 训练输出
        self.train_terminal = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#333333",
            font=("Consolas", 11),
            insertbackground="#333333",
        )
        self.train_terminal.pack(fill=tk.BOTH, expand=True)

        # 训练线程和事件
        self.train_thread = None
        self.train_stop_event = None

    def _create_training_manager_tab(self, parent):
        """创建训练文件管理标签页"""
        # 主框架
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="训练文件管理器",
            font=("微软雅黑", 18, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        # 说明文本
        description = (
            "此工具用于打包本机训练文件，或将其他机器的训练文件包解压到本机。\n"
            "打包文件包含：训练模型、配置文件、训练数据。\n"
            "解包时会覆盖本机现有文件，请先备份重要数据。"
        )
        desc_label = ttk.Label(
            main_frame,
            text=description,
            font=("微软雅黑", 10),
            foreground="#34495e",
            justify=tk.CENTER,
            wraplength=700,
        )
        desc_label.pack(pady=(0, 30))

        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(0, 30))

        # 打包按钮
        self.pack_button = ttk.Button(
            button_frame,
            text="📦 打包本机训练文件",
            command=self._pack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.pack_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 解包按钮
        self.unpack_button = ttk.Button(
            button_frame,
            text="📤 解包训练文件到本机",
            command=self._unpack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.unpack_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 状态区域
        status_frame = ttk.LabelFrame(main_frame, text="操作状态", padding="15")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # 状态文本
        self.training_manager_status_text = tk.Text(
            status_frame,
            height=12,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.training_manager_status_text.pack(fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(
            status_frame, orient=tk.VERTICAL, command=self.training_manager_status_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_manager_status_text.configure(yscrollcommand=scrollbar.set)

        # 底部信息
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X)

        # 当前配置信息
        self.training_config_info = tk.StringVar(value="正在加载配置...")
        config_label = ttk.Label(
            info_frame,
            textvariable=self.training_config_info,
            font=("微软雅黑", 9),
            foreground="#7f8c8d",
        )
        config_label.pack(side=tk.LEFT)

        # 版本信息
        version_label = ttk.Label(
            info_frame,
            text="版本 2.0 · 黑猫交易系统v2.0",
            font=("微软雅黑", 8),
            foreground="#95a5a6",
        )
        version_label.pack(side=tk.RIGHT)

        # 加载配置
        self._load_training_config()

    def _get_base_dir(self):
        """获取基础目录路径（兼容PyInstaller环境）"""
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def _load_training_config(self):
        """加载训练配置文件"""
        try:
            base_dir = self._get_base_dir()
            config_path = os.path.join(base_dir, "train_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.training_config = yaml.safe_load(f)

                exp_name = self.training_config.get("exp_name", "未知")
                save_path = self.training_config.get("save_path", "未知")
                data_path = self.training_config.get("data_path", "未知")

                info_text = f"实验名称: {exp_name} | 保存路径: {save_path} | 数据文件: {os.path.basename(data_path)}"
                self.training_config_info.set(info_text)
                self._log_training_manager_message("✓ 配置文件加载成功", "SUCCESS")
            else:
                self.training_config = {}
                self.training_config_info.set("未找到配置文件 train_config.yaml")
                self._log_training_manager_message("⚠️ 未找到配置文件", "WARNING")
        except Exception as e:
            self.training_config = {}
            self.training_config_info.set("配置文件加载失败")
            self._log_training_manager_message(f"❌ 配置文件加载失败: {e}", "ERROR")

    def _log_training_manager_message(self, message, level="INFO"):
        """记录消息到训练文件管理器状态区域"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.training_manager_status_text.insert(tk.END, formatted_msg)
        self.training_manager_status_text.see(tk.END)
        self.training_manager_status_text.update()

    def _pack_training_files(self):
        """打包本机训练文件"""
        try:
            # 检查配置文件
            if not hasattr(self, 'training_config') or not self.training_config:
                self._log_training_manager_message(
                    "❌ 请先确保配置文件 train_config.yaml 存在且有效", "ERROR"
                )
                return

            # 询问保存位置
            default_filename = f"kronos_training_{self.training_config.get('exp_name', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = filedialog.asksaveasfilename(
                title="保存训练文件包",
                defaultextension=".zip",
                initialfile=default_filename,
                filetypes=[("ZIP压缩包", "*.zip"), ("所有文件", "*.*")],
            )

            if not zip_path:
                self._log_training_manager_message("操作已取消", "INFO")
                return

            self._log_training_manager_message(
                f"开始打包训练文件到: {os.path.basename(zip_path)}", "INFO"
            )

            # 收集要打包的文件
            files_to_pack = []

            # 1. 模型文件
            exp_name = self.training_config.get("exp_name")
            save_path = self.training_config.get("save_path")
            if exp_name:
                # 查找模型文件的实际位置
                model_path = self._find_model_files(exp_name, save_path)
                if model_path and os.path.exists(model_path):
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, ".")
                            files_to_pack.append((full_path, rel_path))
                    self._log_training_manager_message(f"✓ 添加模型文件: {model_path}", "SUCCESS")
                else:
                    self._log_training_manager_message(
                        f"⚠️ 未找到模型文件，实验名称: {exp_name}", "WARNING"
                    )
                    # 尝试搜索所有可能的模型文件
                    self._log_training_manager_message("正在搜索模型文件...", "INFO")
                    model_files_found = self._search_all_model_files()
                    if model_files_found:
                        for file_path, rel_path in model_files_found:
                            files_to_pack.append((file_path, rel_path))
                        self._log_training_manager_message(
                            f"✓ 通过搜索找到 {len(model_files_found)} 个模型文件",
                            "SUCCESS",
                        )

            # 2. 配置文件
            base_dir = self._get_base_dir()
            config_files = ["train_config.yaml"]
            for config_file in config_files:
                full_config_path = os.path.join(base_dir, config_file)
                if os.path.exists(full_config_path):
                    files_to_pack.append((full_config_path, config_file))
                    self._log_training_manager_message(f"✓ 添加配置文件: {config_file}", "SUCCESS")
                else:
                    self._log_training_manager_message(f"⚠️ 配置文件不存在: {full_config_path}", "WARNING")

            # 3. 数据文件
            base_dir = self._get_base_dir()
            data_path = self.training_config.get("data_path")
            if data_path:
                # 尝试找到数据文件
                data_file_found = False

                # 检查绝对路径
                if os.path.exists(data_path):
                    # 如果是绝对路径，转换为相对路径
                    try:
                        rel_path = os.path.relpath(data_path, base_dir)
                        files_to_pack.append((data_path, rel_path))
                        self._log_training_manager_message(f"✓ 添加数据文件: {rel_path}", "SUCCESS")
                        data_file_found = True
                    except ValueError:
                        # 如果不在同一驱动器，使用基本名称
                        files_to_pack.append((data_path, os.path.basename(data_path)))
                        self._log_training_manager_message(
                            f"✓ 添加数据文件: {os.path.basename(data_path)}", "SUCCESS"
                        )
                        data_file_found = True

                # 如果没找到，尝试在training_data目录中查找
                if not data_file_found:
                    data_file_name = os.path.basename(data_path)
                    possible_dirs = [
                        "training_data",
                        "data",
                        "Kronos/finetune_csv/data",
                    ]

                    for data_dir in possible_dirs:
                        test_path = os.path.join(base_dir, data_dir, data_file_name)
                        if os.path.exists(test_path):
                            arc_path = os.path.join(data_dir, data_file_name)
                            files_to_pack.append((test_path, arc_path))
                            self._log_training_manager_message(f"✓ 找到数据文件: {arc_path}", "SUCCESS")
                            data_file_found = True
                            break

                if not data_file_found:
                    self._log_training_manager_message(f"⚠️ 数据文件不存在: {data_path}", "WARNING")
            else:
                self._log_training_manager_message("⚠️ 配置文件中未指定数据文件路径", "WARNING")

            # 创建ZIP文件
            if not files_to_pack:
                self._log_training_manager_message("❌ 没有找到可打包的文件", "ERROR")
                return

            import zipfile
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path, arcname in files_to_pack:
                    zipf.write(file_path, arcname)
                    self._log_training_manager_message(f"  压缩: {arcname}", "INFO")

            # 添加元数据
            metadata = {
                "打包时间": datetime.now().isoformat(),
                "实验名称": self.training_config.get("exp_name"),
                "文件数量": len(files_to_pack),
                "版本": "1.0",
            }

            with zipfile.ZipFile(zip_path, "a") as zipf:
                import json
                zipf.writestr(
                    "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False)
                )

            self._log_training_manager_message(
                f"✓ 打包完成! 共打包 {len(files_to_pack)} 个文件", "SUCCESS"
            )
            self._log_training_manager_message(f"文件保存到: {zip_path}", "INFO")

            messagebox.showinfo(
                "打包完成",
                f"训练文件打包成功!\n\n保存位置: {zip_path}\n文件数量: {len(files_to_pack)}",
            )

        except Exception as e:
            self._log_training_manager_message(f"❌ 打包失败: {e}", "ERROR")

    def _unpack_training_files(self):
        """解包训练文件到本机"""
        try:
            # 选择ZIP文件
            zip_path = filedialog.askopenfilename(
                title="选择训练文件包",
                filetypes=[("ZIP压缩包", "*.zip"), ("所有文件", "*.*")],
            )

            if not zip_path:
                self._log_training_manager_message("操作已取消", "INFO")
                return

            self._log_training_manager_message(f"开始解包文件: {os.path.basename(zip_path)}", "INFO")

            # 读取元数据
            metadata = None
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    if "metadata.json" in zipf.namelist():
                        with zipf.open("metadata.json") as f:
                            import json
                            metadata = json.load(f)
                            self._log_training_manager_message(
                                f"✓ 读取元数据: {metadata.get('实验名称', '未知')}",
                                "SUCCESS",
                            )
            except:
                self._log_training_manager_message("⚠️ 无法读取元数据", "WARNING")

            # 确认解包
            if metadata:
                confirm_msg = f"即将解包训练文件:\n\n实验名称: {metadata.get('实验名称', '未知')}\n打包时间: {metadata.get('打包时间', '未知')}\n文件数量: {metadata.get('文件数量', '未知')}\n\n解包将覆盖本机现有文件，是否继续？"
            else:
                confirm_msg = f"即将解包文件: {os.path.basename(zip_path)}\n\n解包将覆盖本机现有文件，是否继续？"

            if not messagebox.askyesno("确认解包", confirm_msg):
                self._log_training_manager_message("解包操作已取消", "INFO")
                return

            # 解包文件
            extracted_files = []
            skipped_files = []
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zipf:
                # 先获取文件列表
                file_list = [f for f in zipf.namelist() if f != "metadata.json"]

                for filename in file_list:
                    try:
                        # 验证文件路径
                        if not self._validate_zip_path(filename):
                            self._log_training_manager_message(
                                f"  ⚠️ 跳过不安全文件: {filename}", "WARNING"
                            )
                            skipped_files.append(filename)
                            continue

                        # 检查目标目录是否在项目范围内
                        target_path = os.path.join(".", filename)
                        target_dir = os.path.dirname(target_path)

                        # 确保目标目录存在
                        os.makedirs(target_dir, exist_ok=True)

                        # 提取文件
                        zipf.extract(filename, ".")
                        extracted_files.append(filename)
                        self._log_training_manager_message(f"  解压: {filename}", "INFO")
                    except Exception as e:
                        self._log_training_manager_message(f"  ⚠️ 解压失败 {filename}: {e}", "WARNING")

            if extracted_files:
                self._log_training_manager_message(
                    f"✓ 解包完成! 共解压 {len(extracted_files)} 个文件", "SUCCESS"
                )
            else:
                self._log_training_manager_message("⚠️ 没有解压任何文件", "WARNING")

            if skipped_files:
                self._log_training_manager_message(
                    f"⚠️ 跳过了 {len(skipped_files)} 个不安全文件", "WARNING"
                )

            # 重新加载配置
            self._load_training_config()

            messagebox.showinfo(
                "解包完成", f"训练文件解包成功!\n\n解压文件: {len(extracted_files)} 个"
            )

        except Exception as e:
            self._log_training_manager_message(f"❌ 解包失败: {e}", "ERROR")

    def _validate_zip_path(self, filename):
        """验证ZIP文件中的路径是否安全"""
        # 防止路径遍历攻击
        if ".." in filename or filename.startswith("/") or ":" in filename:
            return False

        # 防止绝对路径
        if os.path.isabs(filename):
            return False

        # 防止解压到系统目录
        normalized = os.path.normpath(filename)
        if (
            normalized.startswith("..")
            or normalized == "."
            or normalized.startswith("/")
        ):
            return False

        # 只允许特定扩展名的文件
        allowed_extensions = [
            ".safetensors",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".md",
            ".txt",
            ".py",
        ]
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions and file_ext != "":
            return False

        return True

    def _find_model_files(self, exp_name, save_path):
        """查找模型文件的实际位置"""
        possible_paths = []

        # 1. 配置文件中的路径
        if save_path and exp_name:
            possible_paths.append(os.path.join(save_path, exp_name))

        # 2. 常见的嵌套路径
        possible_paths.append(
            os.path.join(
                "Kronos",
                "finetune_csv",
                "Kronos",
                "finetune_csv",
                "finetuned",
                exp_name,
            )
        )
        possible_paths.append(
            os.path.join("Kronos", "finetune_csv", "finetuned", exp_name)
        )
        possible_paths.append(os.path.join("finetuned", exp_name))

        # 3. 搜索整个项目目录
        for root, dirs, files in os.walk("."):
            if exp_name in dirs:
                possible_paths.append(os.path.join(root, exp_name))

        # 检查路径是否存在
        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _search_all_model_files(self):
        """搜索所有模型文件（safetensors和config.json）"""
        model_files = []
        model_extensions = [".safetensors", ".json", ".md"]  # 模型相关文件扩展名

        # 搜索整个项目目录
        for root, dirs, files in os.walk("."):
            # 跳过一些目录
            if "__pycache__" in root or ".git" in root or "backup_" in root:
                continue

            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in model_extensions:
                    # 检查是否在模型相关目录中
                    if (
                        "model" in root.lower()
                        or "finetuned" in root.lower()
                        or "best_model" in root.lower()
                    ):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        model_files.append((full_path, rel_path))

        return model_files

    def browse_train_data(self):
        """浏览训练数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择训练数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
        )
        if file_path:
            self.train_data_path_var.set(file_path)
            self.log_train(f"已选择数据文件: {file_path}")

    def browse_save_path(self):
        """浏览模型保存路径"""
        dir_path = filedialog.askdirectory(title="选择模型保存目录")
        if dir_path:
            self.train_save_path_var.set(dir_path)
            self.log_train(f"已选择保存路径: {dir_path}")

    def browse_train_config(self):
        """浏览训练配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择训练配置文件",
            filetypes=[("YAML文件", "*.yaml;*.yml"), ("所有文件", "*.*")],
        )
        if file_path:
            self.train_config_path_var.set(file_path)
            self.log_train(f"已选择配置文件: {file_path}")

    def save_train_config(self):
        """保存训练配置到文件"""
        try:
            import yaml

            config = {
                "base_model": self.train_base_model_var.get(),
                "data_path": self.train_data_path_var.get(),
                "lookback_window": int(self.train_lookback_var.get()),
                "predict_window": int(self.train_predict_var.get()),
                "batch_size": int(self.train_batch_var.get()),
                "learning_rate": float(self.train_lr_var.get()),
                "tokenizer_epochs": int(self.train_tokenizer_epochs_var.get()),
                "basemodel_epochs": int(self.train_basemodel_epochs_var.get()),
                "tokenizer_learning_rate": float(self.train_tokenizer_lr_var.get()),
                "log_interval": int(self.train_log_interval_var.get()),
                "train_ratio": float(self.train_ratio_var.get()),
                "val_ratio": float(self.val_ratio_var.get()),
                "num_workers": int(self.train_num_workers_var.get()),
                "exp_name": self.train_exp_name_var.get(),
                "save_path": self.train_save_path_var.get(),
                "tokenizer_folder": self.train_tokenizer_folder_var.get(),
                "basemodel_folder": self.train_basemodel_folder_var.get(),
                "train_tokenizer": self.train_tokenizer_var.get(),
                "train_basemodel": self.train_basemodel_check_var.get(),
                "skip_existing": self.train_skip_existing_var.get(),
                "device": self.train_device_var.get(),
            }

            file_path = filedialog.asksaveasfilename(
                title="保存训练配置",
                defaultextension=".yaml",
                filetypes=[("YAML文件", "*.yaml"), ("所有文件", "*.*")],
                initialdir=".",
                initialfile="train_config.yaml",
            )

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                self.log_train(f"配置已保存: {file_path}")
                messagebox.showinfo("成功", f"配置已保存到: {file_path}")
        except Exception as e:
            self.log_train(f"保存配置失败: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def load_train_config(self):
        """从文件加载训练配置"""
        try:
            import yaml

            file_path = filedialog.askopenfilename(
                title="加载训练配置",
                filetypes=[("YAML文件", "*.yaml;*.yml"), ("所有文件", "*.*")],
            )

            if not file_path:
                return

            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if "base_model" in config:
                self.train_base_model_var.set(config["base_model"])
            if "data_path" in config:
                self.train_data_path_var.set(config["data_path"])
            if "lookback_window" in config:
                self.train_lookback_var.set(str(config["lookback_window"]))
            if "predict_window" in config:
                self.train_predict_var.set(str(config["predict_window"]))
            if "batch_size" in config:
                self.train_batch_var.set(str(config["batch_size"]))
            if "learning_rate" in config:
                self.train_lr_var.set(str(config["learning_rate"]))
            if "tokenizer_epochs" in config:
                self.train_tokenizer_epochs_var.set(str(config["tokenizer_epochs"]))
            if "basemodel_epochs" in config:
                self.train_basemodel_epochs_var.set(str(config["basemodel_epochs"]))
            if "tokenizer_learning_rate" in config:
                self.train_tokenizer_lr_var.set(str(config["tokenizer_learning_rate"]))
            if "log_interval" in config:
                self.train_log_interval_var.set(str(config["log_interval"]))
            if "train_ratio" in config:
                self.train_ratio_var.set(str(config["train_ratio"]))
            if "val_ratio" in config:
                self.val_ratio_var.set(str(config["val_ratio"]))
            if "num_workers" in config:
                self.train_num_workers_var.set(str(config["num_workers"]))
            if "exp_name" in config:
                self.train_exp_name_var.set(config["exp_name"])
            if "save_path" in config:
                self.train_save_path_var.set(config["save_path"])
            if "tokenizer_folder" in config:
                self.train_tokenizer_folder_var.set(config["tokenizer_folder"])
            if "basemodel_folder" in config:
                self.train_basemodel_folder_var.set(config["basemodel_folder"])
            if "train_tokenizer" in config:
                self.train_tokenizer_var.set(config["train_tokenizer"])
            if "train_basemodel" in config:
                self.train_basemodel_check_var.set(config["train_basemodel"])
            if "skip_existing" in config:
                self.train_skip_existing_var.set(config["skip_existing"])
            if "device" in config:
                device_val = config["device"]
                if "GPU" in str(device_val) or "CUDA" in str(device_val):
                    self.train_device_var.set("GPU (CUDA)")
                else:
                    self.train_device_var.set("CPU")

            self.log_train(f"配置已加载: {file_path}")
            messagebox.showinfo("成功", "配置已加载")
        except Exception as e:
            self.log_train(f"加载配置失败: {e}")
            messagebox.showerror("错误", f"加载配置失败: {e}")

    def show_train_help(self):
        """显示训练配置帮助文档"""
        help_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Kronos 模型训练帮助文档                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

【一、基模型选择】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Kronos-mini   │ 最小模型，速度最快，精度较低，适合快速测试
  Kronos-small  │ 小型模型，速度和精度平衡，推荐日常使用
  Kronos-base   │ 中型模型，精度最高，速度较慢，需要更多显存

【二、数据获取】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  交易对   │ 要下载数据的交易对，如 BTCUSDT、ETHUSDT 等
  周期     │ K线时间周期：
           │   1m, 3m, 5m, 15m, 30m (分钟)
           │   1h, 2h, 4h, 6h, 8h, 12h (小时)
           │   1d (天)
  数据量   │ 下载的K线条数，建议 500-2000 条

【三、模型训练参数】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  回看窗口  │ 用多少根历史K线来预测未来
           │ 建议值：256-1024，值越大考虑的历史越长
           │ 默认值：512
  
  预测窗口  │ 预测未来多少根K线
           │ 建议值：24-96，值越大预测越远但越不准
           │ 默认值：48
  
  批次大小  │ 每次训练同时处理的数据量
           │ 值越大训练越快，但需要更多显存
           │ 建议值：16-64，默认值：32
  
  学习率   │ 模型参数更新速度，越小越慢但越精细
           │ 建议值：0.00001-0.001
           │ 默认值：0.0001

【四、高级参数】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tokenizer轮数 │ 分词器训练轮数
               │ 建议值：10-50，默认值：30
  
  模型轮数    │ 基模型微调训练轮数
               │ 建议值：10-50，默认值：20
               │ 轮数越多越精细，但训练时间越长
  
  Tokenizer学习率 │ 分词器专用学习率
                 │ 建议值：0.0001-0.0005，默认值：0.0002
  
  日志间隔    │ 多少批次输出一次日志
               │ 建议值：10-100，默认值：50
  
  训练集比例  │ 数据集中用于训练的比例
               │ 建议值：0.7-0.95，默认值：0.9
  
  验证集比例  │ 数据集中用于验证的比例
               │ 建议值：0.05-0.3，默认值：0.1
  
  数据加载线程 │ 并行加载数据的线程数
               │ 建议值：4-8，默认值：6

【五、实验配置】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  实验名称 │ 训练模型的名称，用于保存模型文件夹
           │ 建议使用英文和下划线，如 btc_5m_v1
  
  保存路径  │ 模型保存的根目录
           │ 默认值：Kronos/finetune_csv/finetuned
  
  Tokenizer文件夹 │ 分词器保存的子文件夹名称
                 │ 默认值：tokenizer
  
  模型文件夹   │ 预测模型保存的子文件夹名称
              │ 默认值：basemodel
  
  最终模型路径 = {保存路径}/{实验名称}/{模型文件夹}/best_model
  
  训练Tokenizer │ 是否训练新的分词器
               │ 首次训练建议勾选，后续微调可取消
  
  训练基模型  │ 是否训练预测模型
              │ 建议始终勾选
  
  跳过已存在训练 │ 如果模型已存在是否跳过
                │ 勾选后不会覆盖已有模型

【六、训练数据格式】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CSV文件必须包含以下列：
  timestamps, open, high, low, close, volume
  
  示例：
  timestamps,open,high,low,close,volume
  2024-01-01 00:00:00,42000,42100,41900,42050,1234.56

【七、训练流程】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 选择或下载训练数据
  2. 选择基模型（推荐 Kronos-small）
  3. 设置训练参数（可使用默认值）
  4. 点击"开始训练"
  5. 等待训练完成

【八、模型保存位置】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  最终保存结构：
  {保存路径}/{实验名称}/
  ├── {Tokenizer文件夹}/best_model    (分词器)
  └── {模型文件夹}/best_model         (预测模型，交易用)
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  分词器模型:                                                            │
  │    {保存路径}/{实验名称}/{Tokenizer文件夹}/best_model                │
  │                                                                         │
  │  预测模型（交易使用）:                                                   │
  │    {保存路径}/{实验名称}/{模型文件夹}/best_model                      │
  └─────────────────────────────────────────────────────────────────────────┘
  
  例如：
    保存路径 = Kronos/finetune_csv/finetuned
    实验名称 = btc_5m_v1
    Tokenizer文件夹 = tokenizer
    模型文件夹 = basemodel
    
    → Kronos/finetune_csv/finetuned/btc_5m_v1/tokenizer/best_model
    → Kronos/finetune_csv/finetuned/btc_5m_v1/basemodel/best_model

【九、使用训练好的模型】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  训练完成后会生成两个模型目录：
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. Tokenizer目录 (分词器)                                          │
  │     路径: {保存路径}/{实验名称}/tokenizer/best_model               │
  │     作用: 将K线数据编码为模型可处理的格式                            │
  │                                                                     │
  │  2. Basemodel目录 (预测模型)                                        │
  │     路径: {保存路径}/{实验名称}/basemodel/best_model               │
  │     作用: 进行价格预测和交易信号生成                                  │
  │     (交易时只需使用这个目录)                                        │
  └─────────────────────────────────────────────────────────────────────┘
  
  使用方法：
  训练完成后，点击"刷新模型列表"按钮，训练好的模型会显示在模型选择下拉框中，直接选择即可

【十、常见问题】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q: 训练很慢怎么办？
  A: 减小回看窗口、预测窗口、批次大小，使用更小的基模型
  
  Q: 显存不足怎么办？
  A: 减小批次大小，或使用 Kronos-mini 模型
  
  Q: 批次大小和显存对应关系？
  A: 
     ┌────────────┬───────────┬─────────────────────┐
     │ 批次大小   │ 显存需求  │ 适合显卡            │
     ├────────────┼───────────┼─────────────────────┤
     │   16       │  ~2GB    │ 所有显卡            │
     │   32       │  ~4GB    │ 入门级显卡         │
     │   64       │  ~8GB    │ 中端显卡 (GTX 1660) │
     │  128       │  ~16GB   │ 高端显卡 (RTX 3080) │
     │  256       │  ~22GB+  │ 旗舰显卡 (RTX 4090) │
     └────────────┴───────────┴─────────────────────┘
     注：显存占用还与回看窗口、预测窗口大小相关
  
  Q: 训练效果不好怎么办？
  A: 增加训练数据量，增加训练轮数，调整学习率
  
  Q: 如何使用训练好的模型？
  A: 模型保存后，在交易面板的模型选择中选择"自定义模型"，输入 basemodel/best_model 路径
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("训练帮助文档")
        help_window.geometry("750x900")

        text_widget = tk.Text(
            help_window, wrap=tk.WORD, bg="#f5f5f5", font=("Consolas", 11)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(help_window, text="关闭", command=help_window.destroy).pack(pady=10)

    def log_train(self, message):
        """训练日志输出"""
        self.train_terminal.insert(tk.END, f"[{self.get_time()}] {message}\n")
        self.train_terminal.see(tk.END)
        # 同时写入训练日志文件
        if hasattr(self, "train_logger"):
            self.train_logger.info(message)
    
    def _calculate_all_technical_indicators(self, df):
        """计算所有27个技术指标"""
        import numpy as np
        import pandas as pd
        
        df = df.copy()
        
        # 基础价格数据
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # 安全获取amount列（训练时用的是 amt 和 vol）
        if "amt" in df.columns:
            amount = df["amt"].values
        elif "amount" in df.columns:
            amount = df["amount"].values
            df["amt"] = amount
        elif "volume" in df.columns:
            amount = df["volume"].values
            df["amt"] = amount
        else:
            amount = np.zeros(len(df))
            df["amt"] = amount
        
        # 添加训练时用的 vol 列
        if "vol" not in df.columns:
            if "volume" in df.columns:
                df["vol"] = df["volume"].values
            else:
                df["vol"] = amount
        
        # 1. MA5, MA10, MA20
        df["MA5"] = pd.Series(close).rolling(window=5).mean().values
        df["MA10"] = pd.Series(close).rolling(window=10).mean().values
        df["MA20"] = pd.Series(close).rolling(window=20).mean().values
        
        # 2. 乖离率
        df["BIAS20"] = (close / df["MA20"] - 1) * 100
        
        # 3. ATR(14)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values
        
        # 4. 振幅
        df["AMPLITUDE"] = (high - low) / close * 100
        
        # 5. 成交额MA5, MA10
        df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
        df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values
        df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]
        
        # 6. RSI(14) 和 RSI(7)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain14 = gain.rolling(window=14).mean()
        avg_loss14 = loss.rolling(window=14).mean()
        rs14 = avg_gain14 / avg_loss14
        df["RSI14"] = (100 - (100 / (1 + rs14))).values
        
        avg_gain7 = gain.rolling(window=7).mean()
        avg_loss7 = loss.rolling(window=7).mean()
        rs7 = avg_gain7 / avg_loss7
        df["RSI7"] = (100 - (100 / (1 + rs7))).values
        
        # 7. MACD线 和 MACD柱
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
        
        # 8. 价格斜率(5期) 和 (10期)
        df["PRICE_SLOPE5"] = (
            pd.Series(close)
            .rolling(window=5)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True,
            )
            .values
        )
        df["PRICE_SLOPE10"] = (
            pd.Series(close)
            .rolling(window=10)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True,
            )
            .values
        )
        
        # 9. 近5日/10日最高/最低
        df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
        df["LOW5"] = pd.Series(low).rolling(window=5).min().values
        df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
        df["LOW10"] = pd.Series(low).rolling(window=10).min().values
        
        # 10. 成交量突破
        df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int).values
        df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int).values
        
        # 选择需要的27个特征
        feature_list = [
            "open", "high", "low", "close", "vol", "amt", 
            "MA5", "MA10", "MA20",
            "BIAS20",
            "ATR14", "AMPLITUDE",
            "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",
            "RSI14", "RSI7",
            "MACD", "MACD_HIST",
            "PRICE_SLOPE5", "PRICE_SLOPE10",
            "HIGH5", "LOW5", "HIGH10", "LOW10",
            "VOL_BREAKOUT", "VOL_SHRINK"
        ]
        
        # 确保所有列都存在
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0.0
        
        # 保留时间列 + 特征列，并去除含 NaN 的行
        time_col = None
        if "timestamps" in df.columns:
            time_col = "timestamps"
        elif "datetime" in df.columns:
            time_col = "datetime"
        
        if time_col:
            df = df[[time_col] + feature_list]
        else:
            df = df[feature_list]
        
        df = df.dropna()
        
        return df

    def download_binance_data(self):
        """下载币安历史数据"""
        try:
            symbol = self.train_symbol_var.get()
            timeframe = self.train_timeframe_var.get()
            days = int(self.train_days_var.get() or 30)

            self.log_train(f"开始下载 {symbol} {timeframe} 最近 {days} 天数据...")
            self.train_status_label.config(text="正在下载数据...")
            self.download_data_button.config(state=tk.DISABLED)

            # 使用线程下载
            import threading

            download_thread = threading.Thread(
                target=self._download_binance_data_thread,
                args=(symbol, timeframe, days),
            )
            download_thread.daemon = True
            download_thread.start()

        except Exception as e:
            self.log_train(f"下载失败: {e}")
            self.train_status_label.config(text="下载失败")
            self.download_data_button.config(state=tk.NORMAL)

    def _download_binance_data_thread(self, symbol, timeframe, days):
        """下载数据的线程，按天数自动分批下载"""
        try:
            from binance_api import BinanceAPI
            import pandas as pd
            import os

            binance = BinanceAPI()

            # 根据时间周期计算每天多少根K线
            timeframe_minutes = {
                "1m": 1,
                "3m": 3,
                "5m": 5,
                "15m": 15,
                "30m": 30,
                "1h": 60,
                "2h": 120,
                "4h": 240,
                "6h": 360,
                "12h": 720,
                "1d": 1440,
            }
            minutes_per_candle = timeframe_minutes.get(timeframe, 5)
            candles_per_day = 1440 // minutes_per_candle
            total_candles = days * candles_per_day

            self.log_train(
                f"周期 {timeframe} 每天约 {candles_per_day} 根K线，共需 {total_candles} 条"
            )

            # 检查是否已有该币种周期的数据文件
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "training_data"
            )
            os.makedirs(save_dir, exist_ok=True)
            existing_file = os.path.join(save_dir, f"{symbol}_{timeframe}.csv")

            batch_size = 1500  # 币安每次最多1500条
            total_batches = (total_candles + batch_size - 1) // batch_size

            self.log_train(
                f"需要下载 {total_candles} 条数据，将分 {total_batches} 批次下载..."
            )

            all_dfs = []
            end_time = None

            for batch in range(total_batches):
                self.log_train(f"下载第 {batch + 1}/{total_batches} 批次...")

                # 获取历史数据
                df_batch = binance.get_historical_klines(
                    symbol, timeframe, end_str=end_time, limit=batch_size
                )

                if df_batch is None or len(df_batch) == 0:
                    self.log_train(f"第 {batch + 1} 批次获取失败或无数据")
                    break

                all_dfs.append(df_batch)

                # 更新结束时间用于下一批（取最早的时间戳）
                oldest_timestamp = df_batch["timestamps"].min()
                end_time = int(oldest_timestamp.timestamp() * 1000) - 1

                self.log_train(
                    f"第 {batch + 1} 批次: 获取 {len(df_batch)} 条，最早时间: {oldest_timestamp}"
                )

                # 如果已经获取足够数据就停止
                if sum(len(d) for d in all_dfs) >= total_candles:
                    break

            if len(all_dfs) == 0:
                self.log_train("未能获取任何数据")
                return

            # 合并所有数据
            df = pd.concat(all_dfs, ignore_index=True)

            # 去重并按时间排序
            df = (
                df.drop_duplicates(subset=["timestamps"])
                .sort_values("timestamps")
                .reset_index(drop=True)
            )

            # 如果数据超过需求，取最新的
            if len(df) > total_candles:
                df = df.tail(total_candles).reset_index(drop=True)

            if len(df) < 100:
                self.log_train("获取数据失败或数据不足")
                return

            # 保存为CSV（一个币种周期只有一个文件，增量累加）
            save_path = existing_file

            # 如果已有文件，合并增量数据
            if os.path.exists(save_path):
                self.log_train("检测到已有数据文件，进行增量更新...")
                df_existing = pd.read_csv(save_path)
                df_existing["timestamps"] = pd.to_datetime(df_existing["timestamps"])

                # 合并新旧数据
                df = pd.concat([df_existing, df], ignore_index=True)

                # 去重并按时间排序
                df = (
                    df.drop_duplicates(subset=["timestamps"])
                    .sort_values("timestamps")
                    .reset_index(drop=True)
                )

                old_count = len(df_existing)
                new_count = len(df)
                self.log_train(f"原有 {old_count} 条，合并后 {new_count} 条")

            df.to_csv(save_path, index=False)

            self.log_train(f"原始数据已保存: {save_path}")
            self.log_train(f"原始数据量: {len(df)} 条")
            self.log_train(
                f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}"
            )
            
            # 自动计算技术指标，生成带27个特征的训练文件
            self.log_train("\n正在计算技术指标...")
            try:
                # 调用我们的技术指标计算函数
                df_with_indicators = self._calculate_all_technical_indicators(df)
                
                # 生成带指标的文件名
                save_path_with_indicators = save_path.replace(".csv", "_with_indicators.csv")
                df_with_indicators.to_csv(save_path_with_indicators, index=False)
                
                self.log_train(f"✓ 技术指标计算完成！")
                self.log_train(f"✓ 带指标的数据已保存: {save_path_with_indicators}")
                self.log_train(f"✓ 特征数: {len(df_with_indicators.columns)}")
                self.log_train(f"✓ 有效数据量: {len(df_with_indicators)} 条")
                
                # 自动设置训练数据路径为带指标的文件
                self.train_data_path_var.set(save_path_with_indicators)
                self.train_status_label.config(text=f"数据准备完成: {len(df_with_indicators)}条(含指标)")
            except Exception as e:
                self.log_train(f"技术指标计算失败: {e}")
                import traceback
                self.log_train(traceback.format_exc())
                # 如果技术指标计算失败，就用原始数据
                self.train_data_path_var.set(save_path)
                self.train_status_label.config(text=f"数据下载完成: {len(df)}条")

        except Exception as e:
            self.log_train(f"下载异常: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_status_label.config(text="下载异常")
        finally:
            self.download_data_button.config(state=tk.NORMAL)

    def start_training(self):
        """开始训练"""
        try:
            data_path = self.train_data_path_var.get().strip()
            config_path = self.train_config_path_var.get().strip()

            if not data_path:
                messagebox.showwarning("警告", "请先选择数据文件！")
                return

            # 转换为绝对路径
            if not os.path.isabs(data_path):
                data_path = os.path.abspath(data_path)

            if not os.path.exists(data_path):
                messagebox.showerror("错误", f"数据文件不存在: {data_path}")
                return

            if not os.path.exists(config_path):
                messagebox.showerror("错误", f"配置文件不存在: {config_path}")
                return

            # 检查数据量是否足够
            import pandas as pd

            try:
                df_temp = pd.read_csv(data_path)
                total_rows = len(df_temp)
                lookback = int(self.train_lookback_var.get())
                predict = int(self.train_predict_var.get())
                window = lookback + predict + 1  # 与数据集计算方式一致

                train_ratio = float(self.train_ratio_var.get())
                val_ratio = float(self.val_ratio_var.get())

                train_rows = int(total_rows * train_ratio)
                val_rows = int(total_rows * val_ratio)

                train_samples = train_rows - window + 1
                val_samples = val_rows - window + 1

                min_required = int(window * 1.5)

                if total_rows < min_required:
                    messagebox.showerror(
                        "错误",
                        f"数据量不足！当前: {total_rows} 行，需要至少 {min_required} 行\n建议：减小回看窗口和预测窗口，或下载更多数据",
                    )
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_train_button.config(state=tk.DISABLED)
                    return

                if val_samples <= 0:
                    messagebox.showerror(
                        "错误",
                        f"验证集数据不足！\n当前验证集: {val_rows} 行，窗口: {window}\n需要验证集至少有 {window} 行\n\n建议：减小验证集比例或下载更多数据",
                    )
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_train_button.config(state=tk.DISABLED)
                    return

                self.log_train(
                    f"数据量检查: 总计{total_rows}行, 训练{train_rows}行({train_samples}样本), 验证{val_rows}行({val_samples}样本), 窗口{window}"
                )
            except Exception as e:
                self.log_train(f"数据检查警告: {e}")

            # 创建停止事件
            self.train_stop_event = threading.Event()

            # 禁用按钮
            self.train_button.config(state=tk.DISABLED)
            self.stop_train_button.config(state=tk.NORMAL)
            self.train_status_label.config(text="正在准备训练...")
            self.train_progress_var.set(0)

            # 清空输出
            self.train_terminal.delete(1.0, tk.END)

            self.log_train("=" * 60)
            self.log_train("开始训练")
            self.log_train("=" * 60)
            self.log_train(f"基模型: {self.train_base_model_var.get()}")
            self.log_train(f"数据文件: {data_path}")
            self.log_train(
                f"回看窗口: {self.train_lookback_var.get()}, 预测窗口: {self.train_predict_var.get()}"
            )
            self.log_train(
                f"批次大小: {self.train_batch_var.get()}, 学习率: {self.train_lr_var.get()}"
            )
            self.log_train(
                f"Tokenizer轮数: {self.train_tokenizer_epochs_var.get()}, 模型轮数: {self.train_basemodel_epochs_var.get()}"
            )
            self.log_train(
                f"训练集: {self.train_ratio_var.get()}, 验证集: {self.val_ratio_var.get()}"
            )
            self.log_train(f"数据路径: {data_path}")
            self.log_train(f"实验名称: {self.train_exp_name_var.get()}")

            # 启动训练线程
            self.train_thread = threading.Thread(
                target=self._training_thread,
                args=(data_path, config_path, self.train_base_model_var.get()),
            )
            self.train_thread.daemon = True
            self.train_thread.start()

        except Exception as e:
            self.log_train(f"启动训练失败: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

    def _training_thread(self, data_path, config_path, base_model):
        """训练线程"""
        try:
            import os
            import sys
            import yaml
            import tempfile

            # 读取配置文件
            self.log_train("正在读取配置文件...")
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 初始化配置结构（如果不存在）
            if "data" not in config:
                config["data"] = {}
            if "training" not in config:
                config["training"] = {}
            if "device" not in config:
                config["device"] = {}
            if "experiment" not in config:
                config["experiment"] = {}
            if "model_paths" not in config:
                config["model_paths"] = {}

            # 修改配置
            config["data"]["data_path"] = data_path
            config["data"]["lookback_window"] = int(self.train_lookback_var.get())
            config["data"]["predict_window"] = int(self.train_predict_var.get())
            config["data"]["train_ratio"] = float(self.train_ratio_var.get())
            config["data"]["val_ratio"] = float(self.val_ratio_var.get())
            config["data"]["test_ratio"] = 0.0
            config["data"]["clip"] = 5.0
            config["data"]["seed"] = 42
            config["training"]["batch_size"] = int(self.train_batch_var.get())
            config["training"]["tokenizer_epochs"] = int(
                self.train_tokenizer_epochs_var.get()
            )
            config["training"]["basemodel_epochs"] = int(
                self.train_basemodel_epochs_var.get()
            )
            config["training"]["tokenizer_learning_rate"] = float(
                self.train_tokenizer_lr_var.get()
            )
            config["training"]["predictor_learning_rate"] = float(
                self.train_lr_var.get()
            )
            config["training"]["log_interval"] = int(self.train_log_interval_var.get())
            config["training"]["num_workers"] = int(self.train_num_workers_var.get())
            config["training"]["seed"] = 42
            config["training"]["adam_beta1"] = 0.9
            config["training"]["adam_beta2"] = 0.95
            config["training"]["adam_weight_decay"] = 0.1
            config["training"]["accumulation_steps"] = 1  # 梯度累积步数

            # 实验配置
            if "experiment" not in config:
                config["experiment"] = {}
            config["experiment"]["train_tokenizer"] = self.train_tokenizer_var.get()
            config["experiment"][
                "train_basemodel"
            ] = self.train_basemodel_check_var.get()
            config["experiment"]["skip_existing"] = self.train_skip_existing_var.get()

            # 设置模型路径和实验名称
            if "model_paths" not in config:
                config["model_paths"] = {}
            config["model_paths"]["exp_name"] = self.train_exp_name_var.get()

            # 设置模型保存路径
            save_path = self.train_save_path_var.get()
            exp_name = self.train_exp_name_var.get()
            tokenizer_folder = self.train_tokenizer_folder_var.get()
            basemodel_folder = self.train_basemodel_folder_var.get()
            config["model_paths"]["base_save_path"] = f"{save_path}/{exp_name}"
            config["model_paths"][
                "finetuned_tokenizer"
            ] = f"{save_path}/{exp_name}/{tokenizer_folder}/best_model"
            config["model_paths"]["basemodel_save_name"] = basemodel_folder

            # 设置基模型
            model_map = {
                "Kronos-mini": "NeoQuasar/Kronos-mini",
                "Kronos-small": "NeoQuasar/Kronos-small",
                "Kronos-base": "NeoQuasar/Kronos-base",
            }
            model_name = model_map.get(base_model, "NeoQuasar/Kronos-small")
            config["model_paths"]["pretrained_predictor"] = model_name
            config["model_paths"][
                "pretrained_tokenizer"
            ] = "NeoQuasar/Kronos-Tokenizer-base"
            self.log_train(f"使用基模型: {model_name}")

            # 添加GPU配置
            use_gpu = "GPU" in self.train_device_var.get()
            if "device" not in config:
                config["device"] = {}
            config["device"]["use_cuda"] = use_gpu
            config["device"]["device_id"] = 0
            self.log_train(f"设备: {'GPU (CUDA)' if use_gpu else 'CPU'}")

            # 保存临时配置文件
            temp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            )
            yaml.dump(config, temp_config, allow_unicode=True)
            temp_config.close()

            self.log_train(f"临时配置文件: {temp_config.name}")

            # 清理GPU内存
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.log_train("已清理GPU缓存")
            except:
                pass

            # 清理已加载的Kronos模型（释放GPU内存）
            try:
                if hasattr(self, "strategy") and self.strategy:
                    # 尝试清理策略中的模型
                    if hasattr(self.strategy, "analyzer"):
                        self.strategy.analyzer = None
                    self.strategy = None
                    self.log_train("已清理交易模型缓存")
            except:
                pass

            # 切换到训练目录
            train_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Kronos", "finetune_csv"
            )
            os.chdir(train_dir)

            self.log_train(f"切换到训练目录: {train_dir}")

            # 导入训练脚本
            sys.path.insert(0, train_dir)

            # 获取训练参数
            tokenizer_train = self.train_tokenizer_var.get()
            basemodel_train = self.train_basemodel_check_var.get()

            self.log_train("=" * 60)
            self.log_train("开始真正的模型训练...")
            self.log_train(f"训练Tokenizer: {tokenizer_train}")
            self.log_train(f"训练基模型: {basemodel_train}")
            self.log_train("=" * 60)

            # 调用真正的训练函数（在子线程中运行，日志会显示在训练面板）
            sys.path.insert(0, train_dir)
            from train_sequential import SequentialTrainer

            trainer = SequentialTrainer(temp_config.name)

            if tokenizer_train:
                self.train_status_label.config(text="正在训练Tokenizer...")
                self.train_progress_var.set(10)
                self.log_train("\n开始训练Tokenizer...")

                self.log_train(f"配置检查:")
                self.log_train(f"  - data_path: {config['data']['data_path']}")
                self.log_train(
                    f"  - lookback: {config['data']['lookback_window']}, predict: {config['data']['predict_window']}"
                )
                self.log_train(
                    f"  - train_ratio: {config['data']['train_ratio']}, val_ratio: {config['data']['val_ratio']}"
                )
                self.log_train("训练中...")

                trainer.train_tokenizer_phase()
                self.train_progress_var.set(50)

            if basemodel_train:
                self.train_status_label.config(text="正在训练预测模型...")
                self.log_train("\n开始训练Predictor...")
                self.log_train("训练中...")
                trainer.train_basemodel_phase()

            self.train_progress_var.set(100)
            self.train_status_label.config(text="训练完成！")
            self.log_train("\n" + "=" * 60)
            self.log_train("训练完成！")
            self.log_train(f"模型保存路径: {save_path}/{exp_name}")
            self.log_train("=" * 60)

            # 恢复按钮状态（不阻塞GUI）
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

        except Exception as e:
            self.log_train(f"启动训练失败: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

    def stop_training(self):
        """停止训练"""
        if self.train_stop_event:
            self.train_stop_event.set()
            self.log_train("正在停止训练...")
            self.train_status_label.config(text="正在停止...")

    def on_closing(self):
        # 取消新闻自动刷新任务
        if self.news_auto_refresh_job:
            self.root.after_cancel(self.news_auto_refresh_job)
        self._stop_balance_recording()
        
        # 清理信号文件
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            signal_file = os.path.join(base_dir, "_splash_close.txt")
            if os.path.exists(signal_file):
                os.remove(signal_file)
        except Exception as e:
            print(f"清理信号文件失败: {e}")
        
        if self.is_running:
            if messagebox.askokcancel("退出", "交易正在运行，确定要退出吗？"):
                self.stop_event.set()
                self.is_running = False
                self.root.destroy()
        else:
            self.root.destroy()

    def _create_ai_trading_tab(self, parent):
        main_frame = ttk.Frame(parent, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="🤖 AI实盘策略", font=("微软雅黑", 18, "bold"), foreground="#2c3e50")
        title_label.pack(pady=(0, 8))

        desc_label = ttk.Label(main_frame, text="实盘策略配置与管理", font=("微软雅黑", 10), foreground="#7f8c8d")
        desc_label.pack(pady=(0, 15))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(button_frame, text="📋 交易风格预设:").pack(side=tk.LEFT, padx=(0, 5))
        self.ai_strategy_preset_var = tk.StringVar(value="平衡型")
        preset_combo = ttk.Combobox(
            button_frame, 
            textvariable=self.ai_strategy_preset_var,
            values=["激进超短线", "趋势追踪", "平衡型", "震荡套利", "稳健长线", "消息驱动"],
            state="readonly",
            width=15
        )
        preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._on_ai_strategy_preset_changed)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="📁 载入配置", command=self._load_ai_strategy_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="💾 保存配置", command=self._save_ai_strategy_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="↩️ 重置默认", command=self._reset_ai_strategy_default).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="✅ 应用配置", command=self._apply_ai_strategy_config, style="Accent.TButton").pack(side=tk.RIGHT, padx=(5, 0))

        self.ai_strategy_notebook = ttk.Notebook(main_frame)
        self.ai_strategy_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.ai_strategy_config_vars = {}
        self._create_ai_strategy_coordinator_tab()
        self._create_ai_strategy_basic_tab()
        self._create_ai_strategy_entry_tab()
        self._create_ai_strategy_stop_loss_tab()
        self._create_ai_strategy_take_profit_tab()
        self._create_ai_strategy_risk_tab()
        self._create_ai_strategy_frequency_tab()
        self._create_ai_strategy_position_tab()
        self._create_ai_strategy_strategy_tab()
        
        self._load_ai_strategy_config_to_ui()
    
    def _get_default_ai_strategy_config(self):
        return {
            "coordinator": {
                "min_signal_strength": 0.15,
                "max_position_size": 0.1,
                "sentiment_weight": 0.3,
                "technical_weight": 0.7,
                "black_swan_threshold": "HIGH",
                "enable_adaptive_filtering": True
            },
            "basic": {
                "POSITION_MULTIPLIER": 1.2,
                "TREND_STRENGTH_THRESHOLD": 0.0047,
                "LOOKBACK_PERIOD": 91,
                "PREDICTION_LENGTH": 90,
                "CHECK_INTERVAL": 180
            },
            "entry": {
                "max_kline_change": 0.015,
                "max_funding_rate_long": 0.03,
                "min_funding_rate_short": -0.03,
                "support_buffer": 1.001,
                "resistance_buffer": 0.999
            },
            "stop_loss": {
                "long_buffer": 0.996,
                "short_buffer": 1.004
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
                "tp3_position_ratio": 0.30
            },
            "risk": {
                "single_trade_risk": 0.029,
                "daily_loss_limit": 0.12,
                "max_consecutive_losses": 6,
                "max_single_position": 0.29,
                "max_daily_position": 0.85
            },
            "frequency": {
                "max_daily_trades": 55,
                "min_trade_interval_minutes": 3,
                "active_hours_start": 0,
                "active_hours_end": 24
            },
            "position": {
                "initial_entry_ratio": 0.35,
                "confirm_interval_kline": 2,
                "add_on_profit": True,
                "add_ratio": 0.25,
                "max_add_times": 3
            },
            "strategy": {
                "entry_confirm_count": 3,
                "reverse_confirm_count": 2,
                "require_consecutive_prediction": 2,
                "post_entry_hours": 6,
                "take_profit_min_pct": 0.5
            }
        }
    
    def _create_ai_strategy_param_row(self, parent, category, param_name, display_name, param_type, min_val=None, max_val=None, description=""):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=2)
        
        name_label = ttk.Label(row_frame, text=display_name, width=25, anchor=tk.W)
        name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        var_key = f"{category}.{param_name}"
        
        if param_type == "float":
            var = tk.DoubleVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        elif param_type == "int":
            var = tk.IntVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        elif param_type == "bool":
            var = tk.BooleanVar()
            entry = ttk.Checkbutton(row_frame, variable=var)
        elif param_type == "select":
            var = tk.StringVar()
            entry = ttk.Combobox(row_frame, textvariable=var, values=["LOW", "MEDIUM", "HIGH"], state="readonly", width=10)
        else:
            var = tk.StringVar()
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
        
        entry.pack(side=tk.LEFT, padx=(0, 5))
        self.ai_strategy_config_vars[var_key] = var
        
        # 为strategy分类的参数添加trace回调，同步更新旧变量
        if category == "strategy":
            def on_var_change(*args, key=var_key, var_obj=var):
                try:
                    value = var_obj.get()
                    if key == "strategy.entry_confirm_count" and hasattr(self, "entry_confirm_count_var"):
                        self.entry_confirm_count_var.set(value)
                    elif key == "strategy.reverse_confirm_count" and hasattr(self, "reverse_confirm_count_var"):
                        self.reverse_confirm_count_var.set(value)
                    elif key == "strategy.require_consecutive_prediction" and hasattr(self, "require_consecutive_prediction_var"):
                        self.require_consecutive_prediction_var.set(value)
                    elif key == "strategy.post_entry_hours" and hasattr(self, "post_entry_hours_var"):
                        self.post_entry_hours_var.set(value)
                    elif key == "strategy.take_profit_min_pct" and hasattr(self, "take_profit_min_pct_var"):
                        self.take_profit_min_pct_var.set(value)
                except Exception:
                    pass
            var.trace_add("write", on_var_change)
        
        if min_val is not None and max_val is not None:
            range_label = ttk.Label(row_frame, text=f"[{min_val}-{max_val}]", width=15, foreground="#7f8c8d")
            range_label.pack(side=tk.LEFT, padx=(0, 5))
        
        desc_label = ttk.Label(row_frame, text=description, foreground="#7f8c8d", wraplength=500)
        desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _create_ai_strategy_coordinator_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="1. 协调器参数")
        
        ttk.Label(frame, text="【协调器参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "coordinator", "min_signal_strength", "最小信号强度", "float", 0.0, 1.0, "只有信号强度超过此值才触发交易")
        self._create_ai_strategy_param_row(frame, "coordinator", "max_position_size", "最大仓位比例", "float", 0.0, 1.0, "单次交易的最大仓位比例")
        self._create_ai_strategy_param_row(frame, "coordinator", "sentiment_weight", "舆情信号权重", "float", 0.0, 1.0, "FinGPT舆情分析的权重")
        self._create_ai_strategy_param_row(frame, "coordinator", "technical_weight", "技术信号权重", "float", 0.0, 1.0, "Kronos技术分析的权重")
        self._create_ai_strategy_param_row(frame, "coordinator", "black_swan_threshold", "黑天鹅阈值", "select", None, None, "极端行情敏感度阈值")
        self._create_ai_strategy_param_row(frame, "coordinator", "enable_adaptive_filtering", "自适应过滤", "bool", None, None, "启用动态参数调整")
    
    def _create_ai_strategy_basic_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="2. 基础参数")
        
        ttk.Label(frame, text="【基础参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "basic", "POSITION_MULTIPLIER", "仓位倍数", "float", 0.1, 10, "交易仓位大小倍数，最小仓位100起")
        self._create_ai_strategy_param_row(frame, "basic", "TREND_STRENGTH_THRESHOLD", "趋势强度阈值", "float", 0.001, 0.05, "判断趋势有效的阈值")
        self._create_ai_strategy_param_row(frame, "basic", "LOOKBACK_PERIOD", "回看K线数量", "int", 64, 2048, "技术分析使用的历史数据长度")
        self._create_ai_strategy_param_row(frame, "basic", "PREDICTION_LENGTH", "预测K线数量", "int", 4, 200, "Kronos预测的未来K线数量")
        self._create_ai_strategy_param_row(frame, "basic", "CHECK_INTERVAL", "检查间隔(秒)", "int", 30, 3600, "系统检查交易信号的间隔")
    
    def _create_ai_strategy_entry_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="3. 入场过滤")
        
        ttk.Label(frame, text="【入场过滤参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "entry", "max_kline_change", "最大单K变化", "float", 0.001, 0.05, "限制单根K线的最大涨跌幅")
        self._create_ai_strategy_param_row(frame, "entry", "max_funding_rate_long", "多头最大资金费率", "float", -0.05, 0.05, "开多仓时资金费率上限")
        self._create_ai_strategy_param_row(frame, "entry", "min_funding_rate_short", "空头最小资金费率", "float", -0.05, 0.05, "开空仓时资金费率下限")
        self._create_ai_strategy_param_row(frame, "entry", "support_buffer", "支撑位缓冲", "float", 1.000, 1.010, "在支撑位上方多少比例入场")
        self._create_ai_strategy_param_row(frame, "entry", "resistance_buffer", "阻力位缓冲", "float", 0.990, 1.000, "在阻力位下方多少比例入场")
    
    def _create_ai_strategy_stop_loss_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="4. 止损参数")
        
        ttk.Label(frame, text="【止损参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "stop_loss", "long_buffer", "多头止损缓冲", "float", 0.900, 0.999, "多头止损相对于入场价的比例")
        self._create_ai_strategy_param_row(frame, "stop_loss", "short_buffer", "空头止损缓冲", "float", 1.001, 1.100, "空头止损相对于入场价的比例")
    
    def _create_ai_strategy_take_profit_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="5. 止盈参数")
        
        ttk.Label(frame, text="【止盈参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_multiplier_long", "多头第一止盈", "float", 1.001, 1.100, "多头第一止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_multiplier_long", "多头第二止盈", "float", 1.001, 1.200, "多头第二止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_multiplier_long", "多头第三止盈", "float", 1.001, 1.300, "多头第三止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_multiplier_short", "空头第一止盈", "float", 0.900, 0.999, "空头第一止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_multiplier_short", "空头第二止盈", "float", 0.800, 0.999, "空头第二止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_multiplier_short", "空头第三止盈", "float", 0.700, 0.999, "空头第三止盈目标")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_position_ratio", "第一止盈仓位", "float", 0.1, 1.0, "达到第一止盈时平掉的仓位比例")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_position_ratio", "第二止盈仓位", "float", 0.1, 1.0, "达到第二止盈时平掉的仓位比例")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_position_ratio", "第三止盈仓位", "float", 0.1, 1.0, "达到第三止盈时平掉的仓位比例")
    
    def _create_ai_strategy_risk_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="6. 风险管理")
        
        ttk.Label(frame, text="【风险管理参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "risk", "single_trade_risk", "单笔风险比例", "float", 0.001, 0.10, "单笔交易最大亏损比例")
        self._create_ai_strategy_param_row(frame, "risk", "daily_loss_limit", "每日亏损限制", "float", 0.01, 0.20, "单日累计亏损达到此值停止交易")
        self._create_ai_strategy_param_row(frame, "risk", "max_consecutive_losses", "最大连续亏损", "int", 1, 10, "连续亏损次数达到此值暂停交易")
        self._create_ai_strategy_param_row(frame, "risk", "max_single_position", "最大单笔仓位", "float", 0.01, 1.0, "单笔交易的最大仓位比例")
        self._create_ai_strategy_param_row(frame, "risk", "max_daily_position", "最大日仓位", "float", 0.01, 1.0, "单日累计开仓的最大仓位比例")
    
    def _create_ai_strategy_frequency_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="7. 交易频率")
        
        ttk.Label(frame, text="【交易频率参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "frequency", "max_daily_trades", "每日最大交易", "int", 1, 100, "限制单日最多交易多少次")
        self._create_ai_strategy_param_row(frame, "frequency", "min_trade_interval_minutes", "最小间隔(分)", "int", 1, 60, "两次交易之间的最小间隔")
        self._create_ai_strategy_param_row(frame, "frequency", "active_hours_start", "活跃开始时间", "int", 0, 23, "只在此时间之后进行交易")
        self._create_ai_strategy_param_row(frame, "frequency", "active_hours_end", "活跃结束时间", "int", 1, 24, "只在此时间之前进行交易")
    
    def _create_ai_strategy_position_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="8. 仓位管理")
        
        ttk.Label(frame, text="【仓位管理参数】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "position", "initial_entry_ratio", "初始入场比例", "float", 0.1, 1.0, "首次开仓时使用目标仓位的比例")
        self._create_ai_strategy_param_row(frame, "position", "confirm_interval_kline", "确认K线数量", "int", 1, 10, "初始入场后等待多少根K线确认趋势")
        self._create_ai_strategy_param_row(frame, "position", "add_on_profit", "盈利加仓", "bool", None, None, "是否在盈利时加仓")
        self._create_ai_strategy_param_row(frame, "position", "add_ratio", "加仓比例", "float", 0.1, 0.5, "每次加仓占目标仓位的比例")
        self._create_ai_strategy_param_row(frame, "position", "max_add_times", "最大加仓次数", "int", 1, 10, "最多可以加仓多少次")
    
    def _create_ai_strategy_strategy_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="9. 策略参数配置")
        
        ttk.Label(frame, text="【策略参数配置】", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_ai_strategy_param_row(frame, "strategy", "entry_confirm_count", "开仓确认次数", "int", 1, 10, "开仓信号需要确认的次数")
        self._create_ai_strategy_param_row(frame, "strategy", "reverse_confirm_count", "平仓确认次数", "int", 1, 10, "平仓信号需要确认的次数")
        self._create_ai_strategy_param_row(frame, "strategy", "require_consecutive_prediction", "连续预测确认", "int", 1, 10, "需要连续多少次预测一致才执行")
        self._create_ai_strategy_param_row(frame, "strategy", "post_entry_hours", "开仓后计时(小时)平仓", "float", 0.5, 24, "开仓后多长时间自动平仓")
        self._create_ai_strategy_param_row(frame, "strategy", "take_profit_min_pct", "最小止盈(%)", "float", 0.1, 10, "最小的止盈比例")
    
    def _load_ai_strategy_config_to_ui(self):
        current_config = self._get_current_ai_strategy_config()
        for category, params in current_config.items():
            for param_name, value in params.items():
                var_key = f"{category}.{param_name}"
                if var_key in self.ai_strategy_config_vars:
                    self.ai_strategy_config_vars[var_key].set(value)
    
    def _get_current_ai_strategy_config(self):
        default_config = self._get_default_ai_strategy_config()
        
        try:
            from strategy_config import StrategyConfig
            default_config["basic"] = {
                "POSITION_MULTIPLIER": 1,
                "TREND_STRENGTH_THRESHOLD": StrategyConfig.TREND_STRENGTH_THRESHOLD,
                "LOOKBACK_PERIOD": StrategyConfig.LOOKBACK_PERIOD,
                "PREDICTION_LENGTH": StrategyConfig.PREDICTION_LENGTH,
                "CHECK_INTERVAL": StrategyConfig.CHECK_INTERVAL
            }
            default_config["entry"] = StrategyConfig.ENTRY_FILTER.copy()
            default_config["stop_loss"] = StrategyConfig.STOP_LOSS.copy()
            default_config["take_profit"] = StrategyConfig.TAKE_PROFIT.copy()
            default_config["risk"] = StrategyConfig.RISK_MANAGEMENT.copy()
            default_config["frequency"] = StrategyConfig.TRADE_FREQUENCY.copy()
            default_config["position"] = StrategyConfig.POSITION_MANAGEMENT.copy()
            if hasattr(StrategyConfig, "STRATEGY_CONFIG"):
                default_config["strategy"] = StrategyConfig.STRATEGY_CONFIG.copy()
        except Exception:
            pass
        
        return default_config
    
    def _get_ai_strategy_config_from_ui(self):
        config = {}
        for category in ["coordinator", "basic", "entry", "stop_loss", "take_profit", "risk", "frequency", "position", "strategy"]:
            config[category] = {}
        
        for var_key, var in self.ai_strategy_config_vars.items():
            category, param_name = var_key.split(".", 1)
            config[category][param_name] = var.get()
        
        return config
    
    def _apply_ai_strategy_config(self):
        config = self._get_ai_strategy_config_from_ui()
        self._apply_strategy_config(config)
        messagebox.showinfo("成功", "策略配置已成功应用！")
    
    def _load_ai_strategy_config(self):
        file_path = filedialog.askopenfilename(
            title="载入配置",
            filetypes=[("YAML文件", "*.yaml"), ("YAML文件", "*.yml"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    self._load_ai_strategy_config_dict(config)
                messagebox.showinfo("成功", "配置已载入！")
            except Exception as e:
                messagebox.showerror("错误", f"载入配置失败: {e}")
    
    def _save_ai_strategy_config(self):
        config = self._get_ai_strategy_config_from_ui()
        file_path = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".yaml",
            filetypes=[("YAML文件", "*.yaml"), ("YAML文件", "*.yml"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                import yaml
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                messagebox.showinfo("成功", "配置已保存！")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def _reset_ai_strategy_default(self):
        if messagebox.askyesno("确认", "确定要重置为默认配置吗？"):
            default_config = self._get_default_ai_strategy_config()
            self._load_ai_strategy_config_dict(default_config)
    
    def _load_ai_strategy_config_dict(self, config):
        for category, params in config.items():
            for param_name, value in params.items():
                var_key = f"{category}.{param_name}"
                if var_key in self.ai_strategy_config_vars:
                    self.ai_strategy_config_vars[var_key].set(value)
                    
                    # 同步到旧的独立变量，保持一致性
                    if category == "strategy":
                        if param_name == "entry_confirm_count" and hasattr(self, "entry_confirm_count_var"):
                            self.entry_confirm_count_var.set(value)
                        elif param_name == "reverse_confirm_count" and hasattr(self, "reverse_confirm_count_var"):
                            self.reverse_confirm_count_var.set(value)
                        elif param_name == "require_consecutive_prediction" and hasattr(self, "require_consecutive_prediction_var"):
                            self.require_consecutive_prediction_var.set(value)
                        elif param_name == "post_entry_hours" and hasattr(self, "post_entry_hours_var"):
                            self.post_entry_hours_var.set(value)
                        elif param_name == "take_profit_min_pct" and hasattr(self, "take_profit_min_pct_var"):
                            self.take_profit_min_pct_var.set(value)
    
    def _on_ai_strategy_preset_changed(self, event):
        preset_name = self.ai_strategy_preset_var.get()
        self._apply_ai_strategy_preset(preset_name)
    
    def _get_ai_strategy_preset_config(self, preset_name):
        presets = {
            "激进超短线": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.8,
                    "TREND_STRENGTH_THRESHOLD": 0.003,
                    "LOOKBACK_PERIOD": 48,
                    "PREDICTION_LENGTH": 8,
                    "CHECK_INTERVAL": 60
                },
                "entry": {
                    "max_kline_change": 0.025,
                    "max_funding_rate_long": 0.05,
                    "min_funding_rate_short": -0.05,
                    "support_buffer": 1.0005,
                    "resistance_buffer": 0.9995
                },
                "stop_loss": {
                    "long_buffer": 0.985,
                    "short_buffer": 1.015
                },
                "take_profit": {
                    "tp1_multiplier_long": 0.965,
                    "tp2_multiplier_long": 1.03,
                    "tp3_multiplier_long": 1.06,
                    "tp1_multiplier_short": 1.035,
                    "tp2_multiplier_short": 0.97,
                    "tp3_multiplier_short": 0.94,
                    "tp1_position_ratio": 0.5,
                    "tp2_position_ratio": 0.3,
                    "tp3_position_ratio": 0.2
                },
                "risk": {
                    "single_trade_risk": 0.05,
                    "daily_loss_limit": 0.2,
                    "max_consecutive_losses": 4,
                    "max_single_position": 0.4,
                    "max_daily_position": 1.0
                },
                "frequency": {
                    "max_daily_trades": 50,
                    "min_trade_interval_minutes": 3,
                    "active_hours_start": 0,
                    "active_hours_end": 24
                },
                "position": {
                    "initial_entry_ratio": 0.5,
                    "confirm_interval_kline": 1,
                    "add_on_profit": True,
                    "add_ratio": 0.5,
                    "max_add_times": 2
                },
                "strategy": {
                    "entry_confirm_count": 1,
                    "reverse_confirm_count": 1,
                    "require_consecutive_prediction": 1,
                    "post_entry_hours": 2.0,
                    "take_profit_min_pct": 0.3
                }
            },
            "趋势追踪": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.5,
                    "TREND_STRENGTH_THRESHOLD": 0.006,
                    "LOOKBACK_PERIOD": 128,
                    "PREDICTION_LENGTH": 36,
                    "CHECK_INTERVAL": 300
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
                    "tp1_multiplier_long": 0.99,
                    "tp2_multiplier_long": 1.08,
                    "tp3_multiplier_long": 1.2,
                    "tp1_multiplier_short": 1.01,
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
                    "max_daily_trades": 10,
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
            },
            "平衡型": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.3,
                    "TREND_STRENGTH_THRESHOLD": 0.0047,
                    "LOOKBACK_PERIOD": 96,
                    "PREDICTION_LENGTH": 24,
                    "CHECK_INTERVAL": 180
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
                    "tp1_multiplier_long": 0.975,
                    "tp2_multiplier_long": 1.05,
                    "tp3_multiplier_long": 1.14,
                    "tp1_multiplier_short": 1.025,
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
            },
            "震荡套利": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.1,
                    "TREND_STRENGTH_THRESHOLD": 0.0025,
                    "LOOKBACK_PERIOD": 64,
                    "PREDICTION_LENGTH": 12,
                    "CHECK_INTERVAL": 120
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
                    "tp1_multiplier_long": 0.962,
                    "tp2_multiplier_long": 1.022,
                    "tp3_multiplier_long": 1.035,
                    "tp1_multiplier_short": 1.038,
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
            },
            "稳健长线": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.0,
                    "TREND_STRENGTH_THRESHOLD": 0.008,
                    "LOOKBACK_PERIOD": 192,
                    "PREDICTION_LENGTH": 48,
                    "CHECK_INTERVAL": 600
                },
                "entry": {
                    "max_kline_change": 0.006,
                    "max_funding_rate_long": 0.01,
                    "min_funding_rate_short": -0.01,
                    "support_buffer": 1.003,
                    "resistance_buffer": 0.997
                },
                "stop_loss": {
                    "long_buffer": 0.95,
                    "short_buffer": 1.05
                },
                "take_profit": {
                    "tp1_multiplier_long": 1.01,
                    "tp2_multiplier_long": 1.12,
                    "tp3_multiplier_long": 1.3,
                    "tp1_multiplier_short": 0.99,
                    "tp2_multiplier_short": 0.88,
                    "tp3_multiplier_short": 0.7,
                    "tp1_position_ratio": 0.2,
                    "tp2_position_ratio": 0.3,
                    "tp3_position_ratio": 0.5
                },
                "risk": {
                    "single_trade_risk": 0.015,
                    "daily_loss_limit": 0.05,
                    "max_consecutive_losses": 10,
                    "max_single_position": 0.2,
                    "max_daily_position": 0.5
                },
                "frequency": {
                    "max_daily_trades": 5,
                    "min_trade_interval_minutes": 60,
                    "active_hours_start": 0,
                    "active_hours_end": 24
                },
                "position": {
                    "initial_entry_ratio": 0.25,
                    "confirm_interval_kline": 6,
                    "add_on_profit": True,
                    "add_ratio": 0.25,
                    "max_add_times": 4
                },
                "strategy": {
                    "entry_confirm_count": 3,
                    "reverse_confirm_count": 4,
                    "require_consecutive_prediction": 4,
                    "post_entry_hours": 24.0,
                    "take_profit_min_pct": 1.2
                }
            },
            "消息驱动": {
                "basic": {
                    "POSITION_MULTIPLIER": 1.3,
                    "TREND_STRENGTH_THRESHOLD": 0.005,
                    "LOOKBACK_PERIOD": 80,
                    "PREDICTION_LENGTH": 18,
                    "CHECK_INTERVAL": 90
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
                    "tp1_multiplier_long": 0.97,
                    "tp2_multiplier_long": 1.045,
                    "tp3_multiplier_long": 1.09,
                    "tp1_multiplier_short": 1.03,
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
                    "max_daily_trades": 25,
                    "min_trade_interval_minutes": 5,
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
                    "entry_confirm_count": 1,
                    "reverse_confirm_count": 2,
                    "require_consecutive_prediction": 2,
                    "post_entry_hours": 5.0,
                    "take_profit_min_pct": 0.4
                }
            }
        }
        return presets.get(preset_name, presets["平衡型"])
    
    def _apply_ai_strategy_preset(self, preset_name):
        preset_config = self._get_ai_strategy_preset_config(preset_name)
        current_config = self._get_ai_strategy_config_from_ui()
        
        for category, params in preset_config.items():
            if category in current_config:
                current_config[category].update(params)
        
        self._load_ai_strategy_config_dict(current_config)

    def _create_ai_strategy_tab(self, parent):
        """创建AI策略中心标签页 - 简化版，仅保留自动优化系统"""
        # 主框架
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="🤖 AI策略中心",
            font=("微软雅黑", 20, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 5))

        # 说明文字
        desc_label = ttk.Label(
            main_frame,
            text="AI定期分析市场，自动优化所有交易参数（开仓、平仓、止盈止损等）",
            font=("微软雅黑", 11),
            foreground="#7f8c8d"
        )
        desc_label.pack(pady=(0, 20))

        # ==================== 系统状态面板 ====================
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # FinGPT状态
        fingpt_status_frame = ttk.Frame(status_frame)
        fingpt_status_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(fingpt_status_frame, text="FinGPT:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.fingpt_status_label = ttk.Label(
            fingpt_status_frame,
            textvariable=self.fingpt_status_var,
            font=("微软雅黑", 10, "bold"),
            foreground="#27ae60"
        )
        self.fingpt_status_label.pack(side=tk.LEFT)
        
        # 策略协调器状态
        coordinator_status_frame = ttk.Frame(status_frame)
        coordinator_status_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(coordinator_status_frame, text="协调器:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.coordinator_status_label = ttk.Label(
            coordinator_status_frame,
            textvariable=self.coordinator_status_var,
            font=("微软雅黑", 10, "bold"),
            foreground="#27ae60"
        )
        self.coordinator_status_label.pack(side=tk.LEFT)

        # ==================== AI策略中心控制面板 ====================
        scheduler_frame = ttk.LabelFrame(main_frame, text="🤖 AI策略中心 - 自动优化系统", padding="20")
        scheduler_frame.pack(fill=tk.BOTH, expand=True)
        
        # 调度器状态和控制
        scheduler_control_row = ttk.Frame(scheduler_frame)
        scheduler_control_row.pack(fill=tk.X, pady=(0, 15))
        
        # 左侧：状态显示
        status_left = ttk.Frame(scheduler_control_row)
        status_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(status_left, text="调度器状态:", font=("微软雅黑", 10)).pack(anchor=tk.W)
        self.ai_scheduler_status_var = tk.StringVar(value="未启动")
        scheduler_status_label = ttk.Label(
            status_left,
            textvariable=self.ai_scheduler_status_var,
            font=("微软雅黑", 11, "bold"),
            foreground="#e74c3c"
        )
        scheduler_status_label.pack(anchor=tk.W)
        
        ttk.Label(status_left, text="上次优化:", font=("微软雅黑", 10)).pack(anchor=tk.W, pady=(8, 0))
        self.last_optimization_var = tk.StringVar(value="--:--:--")
        ttk.Label(
            status_left,
            textvariable=self.last_optimization_var,
            font=("微软雅黑", 10)
        ).pack(anchor=tk.W)
        
        # 中间：控制按钮
        control_middle = ttk.Frame(scheduler_control_row)
        control_middle.pack(side=tk.LEFT, padx=20)
        
        self.start_scheduler_button = ttk.Button(
            control_middle,
            text="▶ 启动自动优化",
            command=self._start_ai_scheduler,
            style="Accent.TButton"
        )
        self.start_scheduler_button.pack(pady=(0, 8))
        
        self.stop_scheduler_button = ttk.Button(
            control_middle,
            text="⏹ 停止自动优化",
            command=self._stop_ai_scheduler,
            state=tk.DISABLED
        )
        self.stop_scheduler_button.pack()
        
        # 右侧：立即优化和设置
        control_right = ttk.Frame(scheduler_control_row)
        control_right.pack(side=tk.RIGHT)
        
        ttk.Button(
            control_right,
            text="⚙️ 参数配置",
            command=self._open_strategy_config
        ).pack(pady=(0, 8))
        
        ttk.Button(
            control_right,
            text="⚡ 立即优化",
            command=self._trigger_optimization_now
        ).pack(pady=(0, 8))
        
        # 检查间隔设置
        interval_frame = ttk.Frame(control_right)
        interval_frame.pack()
        ttk.Label(interval_frame, text="检查间隔(分钟):", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.optimization_interval_var = tk.IntVar(value=60)
        interval_spinbox = ttk.Spinbox(
            interval_frame,
            from_=15,
            to=1440,
            textvariable=self.optimization_interval_var,
            width=8,
            font=("微软雅黑", 9)
        )
        interval_spinbox.pack(side=tk.LEFT)
        
        # 优化参数说明
        params_info_frame = ttk.LabelFrame(scheduler_frame, text="📊 AI优化参数说明", padding="10")
        params_info_frame.pack(fill=tk.X, pady=(15, 0))
        
        params_info_text = """AI策略中心会定期优化以下8大类共50+交易参数：

【1. 协调器参数】
   • min_signal_strength: 最小信号强度 (0.0-1.0)
   • max_position_size: 最大仓位比例 (0.0-1.0)
   • sentiment_weight: 舆情信号权重 (0.0-1.0)
   • technical_weight: 技术信号权重 (0.0-1.0)
   • black_swan_threshold: 黑天鹅风险阈值 (LOW/MEDIUM/HIGH)
   • enable_adaptive_filtering: 自适应过滤开关

【2. 基础参数】
   • POSITION_MULTIPLIER: 仓位倍数 (0.1-10)
   • TREND_STRENGTH_THRESHOLD: 趋势强度阈值
   • LOOKBACK_PERIOD: 回看K线数量
   • CHECK_INTERVAL: 策略检查间隔(秒)

【3. 入场过滤参数】
   • max_kline_change: 最大单根K线变化
   • max_funding_rate_long: 多头最大资金费率
   • min_funding_rate_short: 空头最小资金费率
   • support_buffer: 支撑位缓冲
   • resistance_buffer: 阻力位缓冲

【4. 止损参数】
   • long_buffer: 多头止损缓冲
   • short_buffer: 空头止损缓冲

【5. 止盈参数】
   • tp1_multiplier_long: 多头第一止盈倍数
   • tp2_multiplier_long: 多头第二止盈倍数
   • tp1_multiplier_short: 空头第一止盈倍数
   • tp2_multiplier_short: 空头第二止盈倍数
   • tp1_position_ratio: 第一止盈仓位比例

【6. 风险管理参数】
   • single_trade_risk: 单笔交易风险比例
   • daily_loss_limit: 每日亏损限制
   • max_consecutive_losses: 最大连续亏损次数
   • pause_after_losses_minutes: 亏损后暂停时间
   • max_single_position: 最大单笔仓位
   • max_daily_position: 最大日仓位
   • extreme_move_threshold: 极端波动阈值

【7. 交易频率参数】
   • max_daily_trades: 每日最大交易次数
   • min_trade_interval_minutes: 最小交易间隔
   • active_hours_start: 活跃开始时间
   • active_hours_end: 活跃结束时间

【8. 仓位管理参数】
   • initial_entry_ratio: 初始入场比例
   • confirm_interval_kline: 确认K线数量
"""
        
        params_info_label = ttk.Label(
            params_info_frame,
            text=params_info_text,
            font=("Consolas", 8),
            foreground="#7f8c8d",
            justify=tk.LEFT
        )
        params_info_label.pack(anchor=tk.W)
        
        # 优化历史
        history_frame = ttk.LabelFrame(scheduler_frame, text="📋 优化历史", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.optimization_history_text = tk.Text(history_frame, height=10, font=("Consolas", 9), bg="#f8f9fa", relief=tk.FLAT, wrap=tk.WORD)
        self.optimization_history_text.pack(fill=tk.BOTH, expand=True)
        
        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.optimization_history_text.yview)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.optimization_history_text.configure(yscrollcommand=history_scrollbar.set)

    def _create_live_monitor_tab(self, parent):
        """创建实盘监控标签页（简化版）"""
        # 主框架
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="实盘交易监控面板",
            font=("微软雅黑", 18, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        # 交易所配置框架（固定）
        exchange_frame = ttk.LabelFrame(main_frame, text="交易所配置", padding="15")
        exchange_frame.pack(fill=tk.X, pady=(0, 15))

        # 固定显示交易所和交易对
        ttk.Label(exchange_frame, text="交易所:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(exchange_frame, text="币安 (Binance)", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(exchange_frame, text="交易对:", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(exchange_frame, text="BTCUSDT", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        # 账户信息框架
        account_frame = ttk.LabelFrame(main_frame, text="账户信息", padding="15")
        account_frame.pack(fill=tk.X, pady=(0, 15))

        # 账户余额显示
        ttk.Label(
            account_frame,
            text="账户总余额:",
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            account_frame,
            textvariable=self.account_balance_var,
            font=("微软雅黑", 12, "bold"),
            foreground="#27ae60"
        ).pack(side=tk.LEFT)

        # 更新按钮
        ttk.Button(
            account_frame,
            text="🔄 更新账户信息",
            command=self._update_account_info
        ).pack(side=tk.LEFT, padx=(20, 0))

        # 当前持仓框架
        position_frame = ttk.LabelFrame(main_frame, text="当前持仓", padding="15")
        position_frame.pack(fill=tk.X, pady=(0, 15))

        # 持仓信息显示
        ttk.Label(
            position_frame,
            text="持仓状态:",
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            position_frame,
            textvariable=self.position_info_var,
            font=("微软雅黑", 10),
            foreground="#34495e"
        ).pack(side=tk.LEFT)

        # 实时行情框架
        market_frame = ttk.LabelFrame(main_frame, text="实时行情", padding="15")
        market_frame.pack(fill=tk.X, pady=(0, 15))

        # 价格显示
        ttk.Label(
            market_frame,
            text="当前价格:",
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            market_frame,
            textvariable=self.current_price_var,
            font=("微软雅黑", 12, "bold"),
            foreground="#2980b9"
        ).pack(side=tk.LEFT, padx=(10, 0))

        # 24h涨跌幅
        ttk.Label(
            market_frame,
            text="24h涨跌幅:",
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(20, 10))
        ttk.Label(
            market_frame,
            textvariable=self.price_change_var,
            font=("微软雅黑", 12, "bold"),
            foreground="#e74c3c"
        ).pack(side=tk.LEFT)

        # 监控状态框架
        status_frame = ttk.LabelFrame(main_frame, text="监控状态", padding="15")
        status_frame.pack(fill=tk.X, pady=(0, 15))

        # 监控状态显示
        ttk.Label(
            status_frame,
            textvariable=self.monitor_status_var,
            font=("微软雅黑", 10),
            foreground="#e74c3c"
        ).pack(side=tk.LEFT, padx=(0, 20))

        # AI系统状态简要显示
        ai_brief_frame = ttk.Frame(status_frame)
        ai_brief_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(ai_brief_frame, text="AI系统:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.ai_brief_status_var = tk.StringVar(value="未初始化")
        self.ai_brief_status_label = ttk.Label(
            ai_brief_frame,
            textvariable=self.ai_brief_status_var,
            font=("微软雅黑", 9),
            foreground="#7f8c8d"
        )
        self.ai_brief_status_label.pack(side=tk.LEFT)

        # 交易日志框架
        log_frame = ttk.LabelFrame(main_frame, text="交易日志", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # 交易日志文本框
        self.live_log_text = tk.Text(
            log_frame,
            height=15,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        
        # 传统滚动条（更宽更明显）
        scrollbar = tk.Scrollbar(
            log_frame, 
            orient=tk.VERTICAL, 
            command=self.live_log_text.yview,
            width=25,
            relief=tk.SUNKEN,
            bg="#d3d3d3",
            activebackground="#a9a9a9"
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.live_log_text.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.live_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 底部按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # 启动监控按钮
        self.start_monitor_button = ttk.Button(
            button_frame,
            text="▶ 启动监控",
            command=self._start_live_monitoring,
            width=20,
            style="Accent.TButton"
        )
        self.start_monitor_button.pack(side=tk.LEFT, padx=5)

        # 停止监控按钮
        self.stop_monitor_button = ttk.Button(
            button_frame,
            text="⏹ 停止监控",
            command=self._stop_live_monitoring,
            width=20,
            style="TButton"
        )
        self.stop_monitor_button.pack(side=tk.LEFT, padx=5)
        self.stop_monitor_button.config(state=tk.DISABLED)

        # 清空日志按钮
        ttk.Button(
            button_frame,
            text="🗑 清空日志",
            command=self._clear_live_log
        ).pack(side=tk.LEFT, padx=5)

    def _update_account_info(self):
        """更新账户信息"""
        try:
            from binance_api import BinanceAPI
            binance = BinanceAPI()
            total_balance = binance.get_total_balance()
            
            if total_balance is not None:
                self.account_balance_var.set(f"${total_balance:.2f}")
                self._log_live_message(f"账户余额更新: ${total_balance:.2f} USDT (可用+持仓保证金)", "SUCCESS")
            else:
                self._log_live_message(f"获取账户总余额失败，尝试获取可用余额...", "WARNING")
                balance = binance.get_account_balance()
                if balance:
                    for asset in balance:
                        if asset['asset'] == 'USDT':
                            available = float(asset['availableBalance'])
                            self.account_balance_var.set(f"${available:.2f}")
                            self._log_live_message(f"账户余额更新: ${available:.2f} USDT", "SUCCESS")
                            break
        except Exception as e:
            self._log_live_message(f"更新账户信息失败: {e}", "ERROR")

    def _start_live_monitoring(self):
        """启动实盘监控"""
        try:
            self.is_live_monitoring = True
            self.monitor_status_var.set("运行中")
            self.monitor_status_label.config(foreground="#27ae60")
            
            self.start_monitor_button.config(state=tk.DISABLED)
            self.stop_monitor_button.config(state=tk.NORMAL)
            
            self._log_live_message("实盘监控已启动", "SUCCESS")
            self._log_live_message(f"交易所: {self.live_exchange_var.get()}", "INFO")
            self._log_live_message(f"交易对: {self.live_symbol_var.get()}", "INFO")
            
            # 启动监控线程
            import threading
            self.live_monitor_thread = threading.Thread(
                target=self._live_monitoring_thread,
                daemon=True
            )
            self.live_monitor_thread.start()
            
        except Exception as e:
            self._log_live_message(f"启动监控失败: {e}", "ERROR")

    def _stop_live_monitoring(self):
        """停止实盘监控"""
        try:
            self.is_live_monitoring = False
            self.monitor_status_var.set("已停止")
            self.monitor_status_label.config(foreground="#e74c3c")
            
            self.start_monitor_button.config(state=tk.NORMAL)
            self.stop_monitor_button.config(state=tk.DISABLED)
            
            self._log_live_message("实盘监控已停止", "WARNING")
            
        except Exception as e:
            self._log_live_message(f"停止监控失败: {e}", "ERROR")

    def _create_balance_chart_tab(self, parent):
        """创建资金曲线标签页"""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # 主框架
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 刷新按钮
        ttk.Button(
            control_frame,
            text="刷新图表",
            command=self._update_balance_chart
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 数据统计标签
        self.balance_stats_var = tk.StringVar(value="暂无数据")
        ttk.Label(
            control_frame,
            textvariable=self.balance_stats_var,
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(20, 0))
        
        # 创建3个子图
        self.balance_figure = Figure(figsize=(12, 12), dpi=100)
        
        # 24小时图表（最后1440个点）
        self.balance_ax_24h = self.balance_figure.add_subplot(3, 1, 1)
        
        # 7天图表（每天0:00以后的第一个数据点）
        self.balance_ax_7d = self.balance_figure.add_subplot(3, 1, 2)
        
        # 30天图表（每天0:00以后的第一个数据点）
        self.balance_ax_30d = self.balance_figure.add_subplot(3, 1, 3)
        
        self.balance_chart_canvas = FigureCanvasTkAgg(self.balance_figure, master=main_frame)
        self.balance_chart_canvas.draw()
        self.balance_chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.balance_chart_canvas, main_frame)
        toolbar.update()
        
        # 初始化图表
        self._update_balance_chart()
    
    def _create_news_tab(self, parent):
        """创建BTC新闻标签页"""
        from datetime import datetime
        import webbrowser
        
        # 主框架
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部控制框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 刷新按钮
        ttk.Button(
            control_frame,
            text="刷新新闻",
            command=self._refresh_news
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 新闻状态标签
        self.news_status_var = tk.StringVar(value="点击刷新获取新闻")
        ttk.Label(
            control_frame,
            textvariable=self.news_status_var,
            font=("微软雅黑", 10)
        ).pack(side=tk.LEFT, padx=(20, 0))
        
        # 创建新闻列表框架（带滚动条）
        news_container = ttk.Frame(main_frame)
        news_container.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(news_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 新闻列表文本框
        self.news_text = tk.Text(
            news_container,
            wrap=tk.WORD,
            font=("微软雅黑", 10),
            yscrollcommand=scrollbar.set,
            padx=10,
            pady=10
        )
        self.news_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.news_text.yview)
        
        # 配置文本标签样式
        self.news_text.tag_config("title", font=("微软雅黑", 12, "bold"), foreground="#2c3e50")
        self.news_text.tag_config("source", font=("微软雅黑", 9), foreground="#7f8c8d")
        self.news_text.tag_config("time", font=("微软雅黑", 9), foreground="#95a5a6")
        self.news_text.tag_config("content", font=("微软雅黑", 10), foreground="#34495e")
        self.news_text.tag_config("link", font=("微软雅黑", 9, "underline"), foreground="#3498db")
        self.news_text.tag_config("separator", foreground="#bdc3c7")
        
        # 使新闻文本框只读
        self.news_text.config(state=tk.DISABLED)
    
    def _refresh_news(self):
        """刷新新闻列表"""
        if self.news_crawler is None:
            self.news_status_var.set("新闻爬虫未初始化")
            return
        
        self.news_status_var.set("正在获取新闻...")
        self.news_text.config(state=tk.NORMAL)
        self.news_text.delete(1.0, tk.END)
        
        try:
            # 获取新闻（启用情绪分析和翻译）
            news_list = self.news_crawler.fetch_all_news(force_refresh=True, analyze_sentiment=True, translate=True)
            self.news_list = news_list
            
            if not news_list:
                self.news_text.insert(tk.END, "暂无新闻数据\n", "content")
                self.news_status_var.set("暂无新闻数据")
                self.news_text.config(state=tk.DISABLED)
                return
            
            # 显示新闻
            for i, news in enumerate(news_list):
                # 标题（优先显示中文翻译）
                title_cn = news.get('title_cn')
                title_en = news['title']
                if title_cn and title_cn != title_en:
                    self.news_text.insert(tk.END, f"{title_cn}\n", "title")
                    self.news_text.insert(tk.END, f"{title_en}\n", "source")
                else:
                    self.news_text.insert(tk.END, f"{title_en}\n", "title")
                
                # 情绪标签
                sentiment = news.get('sentiment', 'neutral')
                sentiment_score = news.get('sentiment_score', 0.0)
                if sentiment == 'positive':
                    sentiment_text = "🟢 看多"
                elif sentiment == 'negative':
                    sentiment_text = "🔴 看空"
                else:
                    sentiment_text = "⚪ 中性"
                self.news_text.insert(tk.END, f"情绪: {sentiment_text} (得分: {sentiment_score:.2f})\n", "source")
                
                # 来源和时间
                source_info = f"来源: {news['source']}"
                try:
                    pub_time = datetime.fromisoformat(news['published_at'])
                    time_str = pub_time.strftime("%Y-%m-%d %H:%M")
                    source_info += f" | 时间: {time_str}"
                except:
                    pass
                self.news_text.insert(tk.END, f"{source_info}\n", "source")
                
                # 新闻内容
                content = news['content']
                self.news_text.insert(tk.END, f"【新闻内容】\n{content}\n", "content")
                
                # 链接
                if news.get('url'):
                    self.news_text.insert(tk.END, f"链接: {news['url']}\n", "link")
                
                # 分隔线
                if i < len(news_list) - 1:
                    self.news_text.insert(tk.END, "\n" + "="*80 + "\n\n", "separator")
            
            self.news_status_var.set(f"共 {len(news_list)} 条新闻")
            
        except Exception as e:
            self.news_text.insert(tk.END, f"获取新闻失败: {e}\n", "content")
            self.news_status_var.set(f"获取新闻失败: {e}")
        
        self.news_text.config(state=tk.DISABLED)
    
    def _start_news_auto_refresh(self):
        """启动新闻自动刷新（每5分钟）"""
        if self.news_crawler:
            self._auto_refresh_news()
    
    def _auto_refresh_news(self):
        """自动刷新新闻（后台静默刷新，只做情绪分析缓存）"""
        if self.news_crawler:
            try:
                print("[新闻爬虫] 自动刷新新闻（后台）...")
                # 后台刷新：只做情绪分析，不做新闻总结
                self.news_crawler.fetch_all_news(
                    force_refresh=False,
                    analyze_sentiment=True,
                    translate=False
                )
            except Exception as e:
                print(f"[新闻爬虫] 自动刷新失败: {e}")
        
        # 安排下一次刷新（5分钟 = 300000毫秒）
        self.news_auto_refresh_job = self.root.after(300000, self._auto_refresh_news)
    
    def _update_balance_chart(self):
        """更新资金曲线图 - 显示3个不同时间范围的图表"""
        try:
            if not self.balance_data:
                # 清空所有图表并显示"暂无数据"
                for ax in [self.balance_ax_24h, self.balance_ax_7d, self.balance_ax_30d]:
                    ax.clear()
                    ax.text(0.5, 0.5, "暂无数据", ha='center', va='center',
                           transform=ax.transAxes, fontsize=14, color='gray')
                self.balance_chart_canvas.draw()
                self.balance_stats_var.set("暂无数据")
                return
            
            import pandas as pd
            from datetime import datetime, timedelta
            
            df = pd.DataFrame(self.balance_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 1. 24小时图表（最后1440个点）
            self.balance_ax_24h.clear()
            df_24h = df.tail(1440) if len(df) > 1440 else df
            
            if len(df_24h) > 0:
                # 获取24小时图表的初始资金（从该时间段内的第一个数据点）
                initial_balance_24h = self._get_initial_balance_for_chart(df_24h)
                self._plot_balance_chart(self.balance_ax_24h, df_24h, initial_balance_24h, "24小时资金曲线")
            else:
                self.balance_ax_24h.text(0.5, 0.5, "暂无24小时数据", ha='center', va='center',
                                       transform=self.balance_ax_24h.transAxes, fontsize=12, color='gray')
            
            # 2. 7天图表（每天0:00以后的第一个数据点）
            self.balance_ax_7d.clear()
            df_7d = self._get_daily_data(df, days=7)
            
            if len(df_7d) > 0:
                # 获取7天图表的初始资金（从该时间段内的第一个数据点）
                initial_balance_7d = self._get_initial_balance_for_chart(df_7d)
                self._plot_balance_chart(self.balance_ax_7d, df_7d, initial_balance_7d, "7天资金曲线")
            else:
                self.balance_ax_7d.text(0.5, 0.5, "暂无7天数据", ha='center', va='center',
                                      transform=self.balance_ax_7d.transAxes, fontsize=12, color='gray')
            
            # 3. 30天图表（每天0:00以后的第一个数据点）
            self.balance_ax_30d.clear()
            df_30d = self._get_daily_data(df, days=30)
            
            if len(df_30d) > 0:
                # 获取30天图表的初始资金（从该时间段内的第一个数据点）
                initial_balance_30d = self._get_initial_balance_for_chart(df_30d)
                self._plot_balance_chart(self.balance_ax_30d, df_30d, initial_balance_30d, "30天资金曲线")
            else:
                self.balance_ax_30d.text(0.5, 0.5, "暂无30天数据", ha='center', va='center',
                                       transform=self.balance_ax_30d.transAxes, fontsize=12, color='gray')
            
            self.balance_figure.tight_layout()
            self.balance_chart_canvas.draw()
            
            # 更新统计数据
            if len(df) >= 2:
                first_balance = df['current_balance'].iloc[0]
                last_balance = df['current_balance'].iloc[-1]
                change = last_balance - first_balance
                change_pct = (change / first_balance * 100) if first_balance > 0 else 0
                
                self.balance_stats_var.set(
                    f"总数据点: {len(df)} | 变化: ${change:+.2f} ({change_pct:+.2f}%)"
                )
            else:
                self.balance_stats_var.set(f"总数据点: {len(df)}")
            
        except Exception as e:
            print(f"更新资金曲线图失败: {e}")
    
    def _get_daily_data(self, df, days=7):
        """获取指定天数内每天0:00以后的第一个数据点"""
        try:
            from datetime import datetime, timedelta
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # 获取最近days天的数据
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            if len(recent_df) == 0:
                return pd.DataFrame()
            
            # 按日期分组，取每天的第一个数据点
            recent_df['date'] = recent_df['timestamp'].dt.date
            daily_data = recent_df.groupby('date').first().reset_index()
            
            return daily_data
        except Exception as e:
            print(f"获取每日数据失败: {e}")
            return pd.DataFrame()
    
    def _get_initial_balance_for_chart(self, df):
        """获取图表对应的初始资金"""
        try:
            if len(df) == 0:
                return 0
            
            # 优先使用记录的初始资金
            initial_data = df[df['initial_balance'] > 0]
            if len(initial_data) > 0:
                return initial_data['initial_balance'].iloc[0]
            
            # 如果没有记录初始资金，使用该时间段内的第一个资金值
            return df['current_balance'].iloc[0]
            
        except Exception as e:
            print(f"获取初始资金失败: {e}")
            return 0
    
    def _plot_balance_chart(self, ax, df, initial_balance, title):
        """绘制单个资金曲线图"""
        try:
            import matplotlib.dates as mdates
            
            # 只绘制当前资金线（去掉初始资金线）
            line, = ax.plot(
                df['timestamp'],
                df['current_balance'],
                color='green',
                linewidth=2,
                label='当前资金'
            )
            
            # 填充区域
            ax.fill_between(
                df['timestamp'],
                df['current_balance'],
                alpha=0.3,
                color='lightgreen'
            )
            
            # 设置图表属性
            ax.set_xlabel('时间', fontsize=10)
            ax.set_ylabel('资金 (USDT)', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置Y轴范围（只基于当前资金）
            min_balance = df['current_balance'].min()
            max_balance = df['current_balance'].max()
            
            margin = (max_balance - min_balance) * 0.05
            if margin == 0:
                margin = max_balance * 0.02 if max_balance > 0 else 1
            
            ax.set_ylim(min_balance - margin, max_balance + margin)
            
            # 设置X轴刻度：直接使用数据点的时间，而非均匀分布
            # 根据数据量选择合适的刻度数量
            if len(df) <= 10:
                # 数据点少，显示所有刻度
                tick_indices = list(range(len(df)))
            else:
                # 数据点多，选择约8-10个刻度
                num_ticks = min(10, len(df))
                step = max(1, len(df) // num_ticks)
                tick_indices = list(range(0, len(df), step))
                # 确保最后一个点也显示
                if tick_indices[-1] != len(df) - 1:
                    tick_indices.append(len(df) - 1)
            
            # 设置刻度位置和标签
            tick_dates = df['timestamp'].iloc[tick_indices]
            ax.set_xticks(tick_dates)
            
            # 格式化日期标签
            tick_labels = []
            for date in tick_dates:
                if '24小时' in title:
                    label = date.strftime('%H:%M')
                else:
                    label = date.strftime('%m-%d')
                tick_labels.append(label)
            
            ax.set_xticklabels(tick_labels)
            
            # 自动调整日期显示
            ax.tick_params(axis='x', rotation=45)
            
        except Exception as e:
            print(f"绘制资金曲线图失败: {e}")

    def _live_monitoring_thread(self):
        """实盘监控线程"""
        try:
            from binance_api import BinanceAPI
            import time
            
            binance = BinanceAPI()
            last_balance_update = 0
            last_stats_update = 0
            
            while self.is_live_monitoring:
                try:
                    symbol = self.live_symbol_var.get()
                    now = time.time()
                    
                    # 获取当前价格（每5秒更新）
                    ticker = binance.client.futures_symbol_ticker(symbol=symbol)
                    
                    if ticker:
                        price = float(ticker['price'])
                        self.current_price_var.set(f"${price:.2f}")
                        
                        # 获取24h涨跌幅
                        try:
                            ticker_24h = binance.client.futures_ticker(symbol=symbol)
                            if ticker_24h:
                                change = float(ticker_24h.get('priceChangePercent', 0))
                                if change > 0:
                                    self.price_change_var.set(f"+{change:.2f}%")
                                else:
                                    self.price_change_var.set(f"{change:.2f}%")
                        except:
                            self.price_change_var.set("0.00%")
                    
                    # 获取持仓信息（每5秒更新）
                    try:
                        positions = binance.client.futures_position_information(symbol=symbol)
                        has_position = False
                        for pos in positions:
                            pos_amt = float(pos['positionAmt'])
                            if pos_amt != 0:
                                side = 'LONG' if pos_amt > 0 else 'SHORT'
                                entry_price = float(pos['entryPrice'])
                                unrealized_pnl = float(pos['unRealizedProfit'])
                                self.position_info_var.set(
                                    f"{side} {abs(pos_amt):.4f} | 入场: ${entry_price:.2f} | 盈亏: ${unrealized_pnl:.2f}"
                                )
                                has_position = True
                                break
                        if not has_position:
                            self.position_info_var.set("暂无持仓")
                    except Exception as e:
                        self._log_live_message(f"获取持仓信息失败: {e}", "WARNING")
                    
                    # 更新账户余额（每30秒更新）
                    if now - last_balance_update > 30:
                        try:
                            total_balance = binance.get_total_balance()
                            if total_balance is not None:
                                self.account_balance_var.set(f"${total_balance:.2f}")
                            else:
                                balance = binance.get_account_balance()
                                if balance:
                                    for asset in balance:
                                        if asset['asset'] == 'USDT':
                                            available = float(asset['availableBalance'])
                                            self.account_balance_var.set(f"${available:.2f}")
                                            break
                            last_balance_update = now
                        except Exception as e:
                            self._log_live_message(f"更新账户余额失败: {e}", "WARNING")
                    
                    # 更新交易汇总（每10秒更新）
                    if now - last_stats_update > 10:
                        self._update_trading_stats()
                        last_stats_update = now
                    
                    time.sleep(5)  # 每5秒更新一次
                    
                except Exception as e:
                    self._log_live_message(f"获取行情数据失败: {e}", "WARNING")
                    time.sleep(5)
                    
        except Exception as e:
            self._log_live_message(f"监控线程错误: {e}", "ERROR")
    
    def _update_trading_stats(self):
        """更新交易汇总统计"""
        try:
            from datetime import datetime, timedelta
            
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)
            
            today_trades = 0
            today_profit = 0.0
            week_trades = 0
            week_profit = 0.0
            month_trades = 0
            month_profit = 0.0
            
            # 从strategy中读取交易历史
            if hasattr(self, "strategy") and self.strategy and hasattr(self.strategy, "trade_history"):
                for trade in self.strategy.trade_history:
                    trade_time = trade.get("time")
                    if trade_time:
                        trade_date = trade_time.date()
                        pnl_pct = trade.get("pnl_pct", 0.0)
                        
                        # 今日统计
                        if trade_date == today:
                            today_trades += 1
                            today_profit += pnl_pct
                        
                        # 当周统计
                        if trade_date >= week_start:
                            week_trades += 1
                            week_profit += pnl_pct
                        
                        # 当月统计
                        if trade_date >= month_start:
                            month_trades += 1
                            month_profit += pnl_pct
            
            # 更新UI显示
            self.today_trades_var.set(str(today_trades))
            if today_profit >= 0:
                self.today_profit_var.set(f"+{today_profit:.2f}%")
                if hasattr(self, "today_profit_label"):
                    self.today_profit_label.configure(foreground="#27ae60")
            else:
                self.today_profit_var.set(f"{today_profit:.2f}%")
                if hasattr(self, "today_profit_label"):
                    self.today_profit_label.configure(foreground="#e74c3c")
            
            self.week_trades_var.set(str(week_trades))
            if week_profit >= 0:
                self.week_profit_var.set(f"+{week_profit:.2f}%")
                if hasattr(self, "week_profit_label"):
                    self.week_profit_label.configure(foreground="#27ae60")
            else:
                self.week_profit_var.set(f"{week_profit:.2f}%")
                if hasattr(self, "week_profit_label"):
                    self.week_profit_label.configure(foreground="#e74c3c")
            
            self.month_trades_var.set(str(month_trades))
            if month_profit >= 0:
                self.month_profit_var.set(f"+{month_profit:.2f}%")
                if hasattr(self, "month_profit_label"):
                    self.month_profit_label.configure(foreground="#27ae60")
            else:
                self.month_profit_var.set(f"{month_profit:.2f}%")
                if hasattr(self, "month_profit_label"):
                    self.month_profit_label.configure(foreground="#e74c3c")
            
        except Exception as e:
            self._log_live_message(f"更新交易汇总失败: {e}", "WARNING")
    
    def _run_fingpt_analysis(self, symbol: str):
        """运行FinGPT舆情分析并在终端显示结果"""
        try:
            if self.fingpt_analyzer is None:
                self._log_live_message("[FinGPT] 分析器未初始化，跳过舆情分析", "WARNING")
                return
            
            self._log_live_message(f"[FinGPT] 正在分析 {symbol} 市场情绪...", "INFO")
            
            # 运行舆情分析
            coin_symbol = symbol.replace("USDT", "")
            sentiment_result = self.fingpt_analyzer.analyze_market_sentiment(coin_symbol)
            
            if sentiment_result:
                # 显示情绪结果
                sentiment = sentiment_result.get('overall_sentiment', 'UNKNOWN')
                confidence = sentiment_result.get('confidence', 0)
                risk_level = sentiment_result.get('risk_level', 'LOW')
                
                # 根据情绪设置颜色级别
                level = "INFO"
                if sentiment == "BEARISH":
                    level = "WARNING"
                elif sentiment == "BULLISH":
                    level = "SUCCESS"
                
                self._log_live_message(
                    f"[FinGPT] 市场情绪: {sentiment} | 置信度: {confidence:.1%} | 风险: {risk_level}", 
                    level
                )
                
                # 显示关键指标
                metrics = sentiment_result.get('metrics', {})
                if metrics:
                    fear_greed = metrics.get('fear_greed_index', 'N/A')
                    news_sentiment = metrics.get('news_sentiment', 'N/A')
                    self._log_live_message(
                        f"[FinGPT] 恐惧贪婪指数: {fear_greed} | 新闻情绪: {news_sentiment}", 
                        "INFO"
                    )
                
                # 显示交易建议
                recommendation = sentiment_result.get('recommendation', '')
                if recommendation:
                    self._log_live_message(f"[FinGPT] 建议: {recommendation}", "INFO")
                
                # 显示风险警告
                risk_factors = sentiment_result.get('risk_factors', [])
                if risk_factors and risk_level == "HIGH":
                    self._log_live_message(f"[FinGPT] ⚠️ 检测到高风险因素:", "WARNING")
                    for factor in risk_factors[:2]:
                        self._log_live_message(f"  • {factor}", "WARNING")
                
                # 如果有策略协调器，显示信号过滤情况
                if self.strategy_coordinator is not None:
                    self._log_live_message("[FinGPT] 信号过滤状态: 活跃", "SUCCESS")
            else:
                self._log_live_message("[FinGPT] 舆情分析返回空结果", "WARNING")
                
        except Exception as e:
            self._log_live_message(f"[FinGPT] 舆情分析失败: {e}", "ERROR")

    def _log_live_message(self, message, level="INFO"):
        """记录实盘监控日志（同时记录到两个日志文本框）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "INFO": "#34495e",
            "SUCCESS": "#27ae60",
            "WARNING": "#f39c12",
            "ERROR": "#e74c3c"
        }
        
        color = color_map.get(level, "#34495e")
        
        # 为实盘监控日志文本框配置颜色标签
        if hasattr(self, 'live_log_text'):
            self.live_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.live_log_text.tag_config("INFO", foreground=color_map["INFO"])
            self.live_log_text.tag_config("SUCCESS", foreground=color_map["SUCCESS"])
            self.live_log_text.tag_config("WARNING", foreground=color_map["WARNING"])
            self.live_log_text.tag_config("ERROR", foreground=color_map["ERROR"])
            
            # 插入日志到实盘监控文本框
            self.live_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.live_log_text.insert(tk.END, f"{message}\n", level)
            
            # 只有当滚动条在最底部时才自动跟随
            scroll_position = self.live_log_text.yview()
            if scroll_position[1] >= 0.99:
                self.live_log_text.see(tk.END)
        
        # 为AI策略中心日志文本框配置颜色标签
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.ai_strategy_log_text.tag_config("INFO", foreground=color_map["INFO"])
            self.ai_strategy_log_text.tag_config("SUCCESS", foreground=color_map["SUCCESS"])
            self.ai_strategy_log_text.tag_config("WARNING", foreground=color_map["WARNING"])
            self.ai_strategy_log_text.tag_config("ERROR", foreground=color_map["ERROR"])
            
            # 插入日志到AI策略中心文本框
            self.ai_strategy_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.ai_strategy_log_text.insert(tk.END, f"{message}\n", level)
            
            # 只有当滚动条在最底部时才自动跟随
            scroll_position = self.ai_strategy_log_text.yview()
            if scroll_position[1] >= 0.99:
                self.ai_strategy_log_text.see(tk.END)

    def _clear_live_log(self):
        """清空实盘监控日志（同时清空两个日志文本框）"""
        # 清空实盘监控日志文本框
        if hasattr(self, 'live_log_text'):
            self.live_log_text.delete(1.0, tk.END)
        
        # 清空AI策略中心日志文本框
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.delete(1.0, tk.END)
        
        # 记录清空日志消息
        self._log_live_message("日志已清空", "INFO")

    def _clear_ai_strategy_log(self):
        """清空AI策略中心日志"""
        # 只清空AI策略中心日志文本框
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.delete(1.0, tk.END)
            # 记录清空日志消息到AI策略中心日志
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.ai_strategy_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.ai_strategy_log_text.tag_config("INFO", foreground="#34495e")
            self.ai_strategy_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.ai_strategy_log_text.insert(tk.END, "AI策略中心日志已清空\n", "INFO")
            self.ai_strategy_log_text.see(tk.END)

    def _load_live_config(self):
        """加载实盘监控配置"""
        try:
            # 加载配置文件
            config_path = "live_config.json"
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                if "symbol" in config:
                    self.live_symbol_var.set(config["symbol"])
                if "exchange" in config:
                    self.live_exchange_var.set(config["exchange"])
                    
        except Exception as e:
            pass


    def update_multi_agent_status(self):
        """更新多智能体系统状态显示"""
        # 更新FinGPT状态（AI策略中心标签）
        try:
            if self.fingpt_analyzer is not None:
                self.fingpt_status_var.set("运行中")
                if hasattr(self, 'fingpt_status_label'):
                    self.fingpt_status_label.config(foreground="#27ae60")
            else:
                self.fingpt_status_var.set("未初始化")
                if hasattr(self, 'fingpt_status_label'):
                    self.fingpt_status_label.config(foreground="#e74c3c")
        except:
            pass

        # 更新策略协调器状态（AI策略中心标签）
        try:
            if self.strategy_coordinator is not None:
                self.coordinator_status_var.set("运行中")
                if hasattr(self, 'coordinator_status_label'):
                    self.coordinator_status_label.config(foreground="#27ae60")
            else:
                self.coordinator_status_var.set("未初始化")
                if hasattr(self, 'coordinator_status_label'):
                    self.coordinator_status_label.config(foreground="#e74c3c")
        except:
            pass
        
        # 更新实盘监控标签的AI状态简要显示
        try:
            if hasattr(self, 'ai_brief_status_var') and hasattr(self, 'ai_brief_status_label'):
                if self.fingpt_analyzer is not None and self.strategy_coordinator is not None:
                    self.ai_brief_status_var.set("运行中")
                    self.ai_brief_status_label.config(foreground="#27ae60")
                elif self.fingpt_analyzer is not None or self.strategy_coordinator is not None:
                    self.ai_brief_status_var.set("部分运行")
                    self.ai_brief_status_label.config(foreground="#f39c12")
                else:
                    self.ai_brief_status_var.set("未初始化")
                    self.ai_brief_status_label.config(foreground="#7f8c8d")
        except:
            pass

    def _init_ai_scheduler(self):
        """初始化AI策略中心调度器"""
        try:
            # 尝试导入调度器
            from ai_strategy_scheduler import AIStrategyScheduler
            from parameter_integrator import ParameterIntegrator
            
            # 初始化参数集成器
            self.parameter_integrator = ParameterIntegrator()
            
            # 初始化调度器
            self.ai_scheduler = AIStrategyScheduler(
                parameter_integrator=self.parameter_integrator,
                strategy_coordinator=self.strategy_coordinator,
                check_interval_minutes=60
            )
            
            # 设置回调
            self.ai_scheduler.on_optimization_complete = self._on_optimization_complete
            self.ai_scheduler.on_error = self._on_optimization_error
            
            print("✓ AI策略中心调度器初始化完成")
            return True
            
        except Exception as e:
            print(f"✗ AI策略中心调度器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_ai_scheduler(self):
        """启动AI策略中心调度器"""
        try:
            # 确保调度器已初始化
            if not hasattr(self, "ai_scheduler") or self.ai_scheduler is None:
                if not self._init_ai_scheduler():
                    self._log_ai_strategy_message("❌ AI策略中心调度器初始化失败", "ERROR")
                    return
            
            # 设置检查间隔
            interval = self.optimization_interval_var.get()
            self.ai_scheduler.check_interval_minutes = interval
            
            # 启动调度器
            if self.ai_scheduler.start():
                self.ai_scheduler_status_var.set("运行中")
                self.start_scheduler_button.config(state=tk.DISABLED)
                self.stop_scheduler_button.config(state=tk.NORMAL)
                
                # 更新状态标签颜色
                for widget in self.ai_scheduler_status_var._root.winfo_children():
                    if hasattr(widget, 'configure') and 'foreground' in widget.keys():
                        widget.configure(foreground="#27ae60")
                
                self._log_ai_strategy_message(f"✅ AI策略中心调度器已启动 (检查间隔: {interval}分钟)", "SUCCESS")
                self._log_ai_strategy_message("   📊 调度器将定期分析市场并自动优化交易参数", "INFO")
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 启动AI策略中心调度器失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def _stop_ai_scheduler(self):
        """停止AI策略中心调度器"""
        try:
            if hasattr(self, "ai_scheduler") and self.ai_scheduler:
                if self.ai_scheduler.stop():
                    self.ai_scheduler_status_var.set("已停止")
                    self.start_scheduler_button.config(state=tk.NORMAL)
                    self.stop_scheduler_button.config(state=tk.DISABLED)
                    
                    self._log_ai_strategy_message("⏹ AI策略中心调度器已停止", "WARNING")
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 停止AI策略中心调度器失败: {e}", "ERROR")
    
    def _open_strategy_config(self):
        """打开策略参数配置对话框"""
        try:
            # 获取当前配置
            current_config = self._get_current_strategy_config()
            
            # 打开对话框
            dialog = StrategyConfigDialog(self.root, current_config)
            result = dialog.show()
            
            if result is not None:
                # 应用配置
                self._apply_strategy_config(result)
                self._log_ai_strategy_message("✅ 策略配置已应用", "SUCCESS")
                messagebox.showinfo("成功", "策略配置已成功应用！")
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 打开配置对话框失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def _get_current_strategy_config(self):
        """获取当前策略配置"""
        config = {
            "coordinator": {},
            "basic": {},
            "entry": {},
            "stop_loss": {},
            "take_profit": {},
            "risk": {},
            "frequency": {},
            "position": {}
        }
        
        try:
            # 从策略协调器获取配置
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                coord_config = getattr(self.strategy_coordinator, "config", {})
                config["coordinator"] = coord_config.copy()
        except Exception:
            pass
        
        try:
            # 从strategy_config.py读取默认配置
            from strategy_config import StrategyConfig
            config["basic"] = {
                "POSITION_MULTIPLIER": 1,
                "TREND_STRENGTH_THRESHOLD": StrategyConfig.TREND_STRENGTH_THRESHOLD,
                "LOOKBACK_PERIOD": StrategyConfig.LOOKBACK_PERIOD,
                "PREDICTION_LENGTH": StrategyConfig.PREDICTION_LENGTH,
                "CHECK_INTERVAL": StrategyConfig.CHECK_INTERVAL
            }
            config["entry"] = StrategyConfig.ENTRY_FILTER.copy()
            config["stop_loss"] = StrategyConfig.STOP_LOSS.copy()
            config["take_profit"] = StrategyConfig.TAKE_PROFIT.copy()
            config["risk"] = StrategyConfig.RISK_MANAGEMENT.copy()
            config["frequency"] = StrategyConfig.TRADE_FREQUENCY.copy()
            config["position"] = StrategyConfig.POSITION_MANAGEMENT.copy()
        except Exception:
            pass
        
        return config
    
    def _apply_strategy_config(self, config):
        """应用策略配置"""
        try:
            # 更新策略协调器配置
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                if "coordinator" in config:
                    for key, value in config["coordinator"].items():
                        if key in self.strategy_coordinator.config:
                            self.strategy_coordinator.config[key] = value
            
            # 更新正在运行的策略实例
            if hasattr(self, "strategy") and self.strategy:
                if hasattr(self.strategy, "update_config"):
                    success = self.strategy.update_config(config)
                    if success:
                        self._log_ai_strategy_message("✅ 策略实例配置已动态更新", "SUCCESS")
            
            # 更新strategy_config.py文件
            self._update_strategy_config_file(config)
            
            self._log_ai_strategy_message("✅ 策略配置已更新", "SUCCESS")
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 应用配置失败: {e}", "ERROR")
            raise
    
    def _update_strategy_config_file(self, config):
        """更新strategy_config.py文件"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "strategy_config.py")
            
            config_content = '''"""策略配置文件 - 由参数集成器自动生成"""

class StrategyConfig:
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '5m'
    LEVERAGE = {LEVERAGE}
    TREND_STRENGTH_THRESHOLD = {TREND_STRENGTH_THRESHOLD}
    LOOKBACK_PERIOD = {LOOKBACK_PERIOD}
    PREDICTION_LENGTH = {PREDICTION_LENGTH}
    CHECK_INTERVAL = {CHECK_INTERVAL}
    ENTRY_FILTER = {{
        "max_kline_change": {max_kline_change},
        "max_funding_rate_long": {max_funding_rate_long},
        "min_funding_rate_short": {min_funding_rate_short},
        "support_buffer": {support_buffer},
        "resistance_buffer": {resistance_buffer},
    }}

    STOP_LOSS = {{
        "long_buffer": {long_buffer},
        "short_buffer": {short_buffer},
    }}

    TAKE_PROFIT = {{
        "tp1_multiplier_long": {tp1_multiplier_long},
        "tp2_multiplier_long": {tp2_multiplier_long},
        "tp3_multiplier_long": {tp3_multiplier_long},
        "tp1_multiplier_short": {tp1_multiplier_short},
        "tp2_multiplier_short": {tp2_multiplier_short},
        "tp3_multiplier_short": {tp3_multiplier_short},
        "tp1_position_ratio": {tp1_position_ratio},
        "tp2_position_ratio": {tp2_position_ratio},
        "tp3_position_ratio": {tp3_position_ratio},
    }}

    RISK_MANAGEMENT = {{
        "single_trade_risk": {single_trade_risk},
        "daily_loss_limit": {daily_loss_limit},
        "max_consecutive_losses": {max_consecutive_losses},
        "max_single_position": {max_single_position},
        "max_daily_position": {max_daily_position},
    }}

    TRADE_FREQUENCY = {{
        "max_daily_trades": {max_daily_trades},
        "min_trade_interval_minutes": {min_trade_interval_minutes},
        "active_hours_start": {active_hours_start},
        "active_hours_end": {active_hours_end},
    }}

    POSITION_MANAGEMENT = {{
        "initial_entry_ratio": {initial_entry_ratio},
        "confirm_interval_kline": {confirm_interval_kline},
        "add_on_profit": {add_on_profit},
        "add_ratio": {add_ratio},
        "max_add_times": {max_add_times},
    }}

    STRATEGY_CONFIG = {{
        "entry_confirm_count": {entry_confirm_count},
        "reverse_confirm_count": {reverse_confirm_count},
        "require_consecutive_prediction": {require_consecutive_prediction},
        "post_entry_hours": {post_entry_hours},
        "take_profit_min_pct": {take_profit_min_pct},
    }}
'''
            
            # 填充配置值
            basic = config.get("basic", {})
            entry = config.get("entry", {})
            stop_loss = config.get("stop_loss", {})
            take_profit = config.get("take_profit", {})
            risk = config.get("risk", {})
            frequency = config.get("frequency", {})
            position = config.get("position", {})
            strategy = config.get("strategy", {})
            
            config_content = config_content.format(
                LEVERAGE=basic.get("LEVERAGE", 10),
                TREND_STRENGTH_THRESHOLD=basic.get("TREND_STRENGTH_THRESHOLD", 0.0047),
                LOOKBACK_PERIOD=basic.get("LOOKBACK_PERIOD", 91),
                PREDICTION_LENGTH=basic.get("PREDICTION_LENGTH", 90),
                CHECK_INTERVAL=basic.get("CHECK_INTERVAL", 180),
                max_kline_change=entry.get("max_kline_change", 0.015),
                max_funding_rate_long=entry.get("max_funding_rate_long", 0.03),
                min_funding_rate_short=entry.get("min_funding_rate_short", -0.03),
                support_buffer=entry.get("support_buffer", 1.001),
                resistance_buffer=entry.get("resistance_buffer", 0.999),
                long_buffer=stop_loss.get("long_buffer", 0.996),
                short_buffer=stop_loss.get("short_buffer", 1.004),
                tp1_multiplier_long=take_profit.get("tp1_multiplier_long", 1.025),
                tp2_multiplier_long=take_profit.get("tp2_multiplier_long", 1.05),
                tp3_multiplier_long=take_profit.get("tp3_multiplier_long", 1.14),
                tp1_multiplier_short=take_profit.get("tp1_multiplier_short", 0.975),
                tp2_multiplier_short=take_profit.get("tp2_multiplier_short", 0.95),
                tp3_multiplier_short=take_profit.get("tp3_multiplier_short", 0.86),
                tp1_position_ratio=take_profit.get("tp1_position_ratio", 0.35),
                tp2_position_ratio=take_profit.get("tp2_position_ratio", 0.35),
                tp3_position_ratio=take_profit.get("tp3_position_ratio", 0.30),
                single_trade_risk=risk.get("single_trade_risk", 0.029),
                daily_loss_limit=risk.get("daily_loss_limit", 0.12),
                max_consecutive_losses=risk.get("max_consecutive_losses", 6),
                max_single_position=risk.get("max_single_position", 0.29),
                max_daily_position=risk.get("max_daily_position", 0.85),
                max_daily_trades=frequency.get("max_daily_trades", 55),
                min_trade_interval_minutes=frequency.get("min_trade_interval_minutes", 3),
                active_hours_start=frequency.get("active_hours_start", 0),
                active_hours_end=frequency.get("active_hours_end", 24),
                initial_entry_ratio=position.get("initial_entry_ratio", 0.35),
                confirm_interval_kline=position.get("confirm_interval_kline", 2),
                add_on_profit=position.get("add_on_profit", True),
                add_ratio=position.get("add_ratio", 0.25),
                max_add_times=position.get("max_add_times", 3),
                entry_confirm_count=strategy.get("entry_confirm_count", 3),
                reverse_confirm_count=strategy.get("reverse_confirm_count", 2),
                require_consecutive_prediction=strategy.get("require_consecutive_prediction", 2),
                post_entry_hours=strategy.get("post_entry_hours", 6),
                take_profit_min_pct=strategy.get("take_profit_min_pct", 0.5)
            )
            
            # 写入文件
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
        except Exception as e:
            print(f"更新配置文件失败: {e}")
            raise
    
    def _trigger_optimization_now(self):
        """立即触发一次优化"""
        try:
            # 确保调度器已初始化
            if not hasattr(self, "ai_scheduler") or self.ai_scheduler is None:
                if not self._init_ai_scheduler():
                    self._log_ai_strategy_message("❌ AI策略中心调度器初始化失败", "ERROR")
                    return
            
            self._log_ai_strategy_message("⚡ 触发立即优化...", "INFO")
            
            if self.ai_scheduler.trigger_optimization_now():
                self._log_ai_strategy_message("✓ 优化任务已提交", "SUCCESS")
            else:
                self._log_ai_strategy_message("⚠️ 请先启动调度器", "WARNING")
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 触发优化失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def _on_optimization_complete(self, record: Dict):
        """优化完成回调"""
        try:
            timestamp = record.get("timestamp", "")
            dt = datetime.fromisoformat(timestamp)
            
            # 更新上次优化时间
            self.last_optimization_var.set(dt.strftime("%H:%M:%S"))
            
            # 添加到历史记录
            self._add_optimization_history(record)
            
            # 记录日志
            self._log_ai_strategy_message(f"✅ 优化完成: {dt.strftime('%Y-%m-%d %H:%M:%S')}", "SUCCESS")
            
        except Exception as e:
            print(f"优化完成回调错误: {e}")
    
    def _on_optimization_error(self, error: Exception):
        """优化错误回调"""
        try:
            self._log_ai_strategy_message(f"❌ 优化错误: {error}", "ERROR")
        except Exception as e:
            print(f"优化错误回调错误: {e}")
    
    def _add_optimization_history(self, record: Dict):
        """添加优化历史记录"""
        try:
            timestamp = record.get("timestamp", "")
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")
            
            params = record.get("optimized_parameters", {})
            
            history_entry = f"[{time_str}] 优化完成\n"
            
            if "market_analysis" in params:
                history_entry += f"  市场分析: {params['market_analysis']}\n"
            
            if "optimization_reasoning" in params:
                history_entry += f"  优化理由: {params['optimization_reasoning']}\n"
            
            history_entry += "-" * 50 + "\n"
            
            # 添加到文本框
            self.optimization_history_text.insert(tk.END, history_entry)
            self.optimization_history_text.see(tk.END)
            
        except Exception as e:
            print(f"添加优化历史错误: {e}")

    def _initialize_auto_optimization_system(self):
        """初始化自动化优化系统"""
        try:
            if PerformanceMonitor is None or AutoOptimizationPipeline is None:
                self._log_ai_strategy_message("⚠️ 自动化优化模块不可用", "WARNING")
                return False
            
            # 初始化性能监控器
            if self.performance_monitor is None:
                # 从GUI选项获取判断模式
                mode_map = {
                    "固定阈值": "threshold",
                    "AI判断": "ai",
                    "混合模式": "hybrid"
                }
                judgment_mode = mode_map.get(self.optimization_judgment_mode_var.get(), "threshold")
                
                self.performance_monitor = PerformanceMonitor(
                    strategy_instance=None,  # 稍后设置
                    check_interval_seconds=900,  # 默认15分钟
                    min_trades_for_analysis=5,
                    performance_history_file="performance_history.json",
                    judgment_mode=judgment_mode
                )
                
                # 设置阈值突破回调
                self.performance_monitor.on_threshold_breach = self._on_threshold_breach_callback
                self.performance_monitor.on_performance_update = self._on_performance_update_callback
                
                self._log_ai_strategy_message("✓ 性能监控器初始化完成", "SUCCESS")
                
                # 如果已有策略实例，立即连接
                if hasattr(self, 'strategy') and self.strategy:
                    try:
                        self.performance_monitor.set_strategy_instance(self.strategy)
                        self._log_ai_strategy_message("  已连接到当前策略实例", "INFO")
                    except Exception as e:
                        self._log_ai_strategy_message(f"  连接策略实例失败: {e}", "WARNING")
            
            # 初始化自动化优化管道
            if self.auto_optimization_pipeline is None:
                self.auto_optimization_pipeline = AutoOptimizationPipeline(
                    symbol="BTCUSDT",
                    config_path="strategy_config.py",
                    optimization_history_file="auto_optimization_history.json"
                )
                
                # 设置优化回调
                self.auto_optimization_pipeline.set_callback("optimization_complete", self._on_optimization_complete_callback)
                self.auto_optimization_pipeline.set_callback("optimization_error", self._on_optimization_error_callback)
                
                self._log_ai_strategy_message("✓ 自动化优化管道初始化完成", "SUCCESS")
            
            # 初始化参数集成器
            if self.parameter_integrator is None:
                self.parameter_integrator = ParameterIntegrator(
                    config_file="strategy_config.py",
                    backup_dir="config_backups"
                )
                self._log_ai_strategy_message("✓ 参数集成器初始化完成", "SUCCESS")
            
            # 确保性能监控器连接到策略实例（如果存在）
            if self.performance_monitor and hasattr(self, 'strategy') and self.strategy:
                try:
                    self.performance_monitor.set_strategy_instance(self.strategy)
                    self._log_ai_strategy_message("性能监控器已连接到策略实例", "INFO")
                except Exception as e:
                    self._log_ai_strategy_message(f"连接性能监控器失败: {e}", "WARNING")
            
            return True
            
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 自动化优化系统初始化失败: {e}", "ERROR")
            return False

    def _toggle_auto_optimization(self):
        """切换自动化优化状态"""
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("❌ 自动化优化系统初始化失败，无法启用", "ERROR")
                return
            
            if not self.is_auto_optimization_enabled:
                # 启用自动化优化
                self.is_auto_optimization_enabled = True
                self.auto_optimization_status_var.set("运行中")
                
                # 启动性能监控
                if self.performance_monitor:
                    self.performance_monitor.start_monitoring()
                
                self._log_ai_strategy_message("✅ 自动化优化已启用", "SUCCESS")
                self._log_ai_strategy_message("  系统将持续监控交易表现并在阈值突破时自动优化", "INFO")
                
                # 更新按钮文本
                # 按钮文本会在下次界面更新时反映
            else:
                # 禁用自动化优化
                self.is_auto_optimization_enabled = False
                self.auto_optimization_status_var.set("禁用")
                
                # 停止性能监控
                if self.performance_monitor:
                    self.performance_monitor.stop_monitoring()
                
                self._log_ai_strategy_message("⏸️ 自动化优化已禁用", "INFO")
                
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 切换自动化优化状态失败: {e}", "ERROR")

    def _apply_judgment_mode(self):
        """应用判断模式切换"""
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("❌ 自动化优化系统未初始化，无法切换模式", "ERROR")
                return
            
            mode_display = self.optimization_judgment_mode_var.get()
            mode_map = {
                "固定阈值": "threshold",
                "AI判断": "ai",
                "混合模式": "hybrid"
            }
            judgment_mode = mode_map.get(mode_display, "threshold")
            
            # 切换模式
            if self.performance_monitor:
                self.performance_monitor.set_judgment_mode(
                    mode=judgment_mode
                )
                self._log_ai_strategy_message(f"✅ 判断模式已切换为: {mode_display}", "SUCCESS")
            else:
                self._log_ai_strategy_message("❌ 性能监控器未初始化", "ERROR")
                
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 切换判断模式失败: {e}", "ERROR")

    def _trigger_manual_optimization(self):
        """触发手动优化"""
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("❌ 自动化优化系统初始化失败，无法优化", "ERROR")
                return
            
            self._log_ai_strategy_message("🚀 触发手动优化...", "INFO")
            
            # 触发优化管道
            if self.auto_optimization_pipeline:
                success = self.auto_optimization_pipeline.trigger_optimization(
                    threshold_breaches=["手动触发"],
                    performance_data={},
                    force_optimization=True
                )
                
                if success:
                    self._log_ai_strategy_message("✅ 手动优化已启动，请等待完成", "SUCCESS")
                else:
                    self._log_ai_strategy_message("⚠️ 手动优化启动失败（可能正在运行）", "WARNING")
            else:
                self._log_ai_strategy_message("❌ 自动化优化管道未初始化", "ERROR")
                
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 触发手动优化失败: {e}", "ERROR")

    def _on_threshold_breach_callback(self, breaches, performance_data):
        """阈值突破回调函数"""
        try:
            self._log_ai_strategy_message("⚠️ 检测到性能阈值突破！", "WARNING")
            self._log_ai_strategy_message(f"  突破项: {', '.join(breaches)}", "INFO")
            
            # 显示性能数据摘要
            if performance_data:
                summary = []
                for key, value in performance_data.items():
                    if isinstance(value, (int, float)):
                        summary.append(f"{key}: {value:.3f}")
                
                if summary:
                    self._log_ai_strategy_message(f"  当前性能: {', '.join(summary[:3])}", "INFO")
            
            # 触发自动化优化
            if self.auto_optimization_pipeline and self.is_auto_optimization_enabled:
                self._log_ai_strategy_message("🔄 自动触发优化流程...", "INFO")
                
                success = self.auto_optimization_pipeline.trigger_optimization(
                    threshold_breaches=breaches,
                    performance_data=performance_data,
                    force_optimization=True
                )
                
                if success:
                    self._log_ai_strategy_message("✅ 自动化优化已启动", "SUCCESS")
                else:
                    self._log_ai_strategy_message("⚠️ 自动化优化启动失败（可能正在运行）", "WARNING")
                    
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 阈值突破处理失败: {e}", "ERROR")

    def _on_performance_update_callback(self, performance_data):
        """性能更新回调函数"""
        # 可以在这里更新GUI显示性能指标
        pass

    def _on_optimization_complete_callback(self, optimization_record):
        """优化完成回调函数"""
        try:
            if optimization_record.get("success"):
                self._log_ai_strategy_message("✅ 自动化优化完成！", "SUCCESS")
                
                # 显示优化摘要
                report = optimization_record.get("final_report", {})
                if report:
                    performance = report.get("backtest_performance", {})
                    
                    summary = []
                    if "total_return_pct" in performance:
                        summary.append(f"收益率: {performance['total_return_pct']:.1f}%")
                    if "sharpe_ratio" in performance:
                        summary.append(f"夏普比率: {performance['sharpe_ratio']:.2f}")
                    if "win_rate_pct" in performance:
                        summary.append(f"胜率: {performance['win_rate_pct']:.1f}%")
                    
                    if summary:
                        self._log_ai_strategy_message(f"  优化结果: {', '.join(summary)}", "INFO")
                
                # 尝试自动集成参数
                self._integrate_optimized_parameters(optimization_record)
                
            else:
                error = optimization_record.get("error", "未知错误")
                self._log_ai_strategy_message(f"❌ 自动化优化失败: {error}", "ERROR")
                
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 优化完成回调处理失败: {e}", "ERROR")

    def _on_optimization_error_callback(self, error, optimization_record):
        """优化错误回调函数"""
        self._log_ai_strategy_message(f"❌ 自动化优化发生错误: {error}", "ERROR")

    def _integrate_optimized_parameters(self, optimization_record):
        """集成优化后的参数"""
        try:
            if not self.parameter_integrator:
                self._log_ai_strategy_message("⚠️ 参数集成器未初始化，跳过参数集成", "WARNING")
                return
            
            # 从优化记录中提取参数
            steps = optimization_record.get("steps", [])
            integrated_params = {}
            
            for step in steps:
                if step.get("description", "").startswith("整合完成"):
                    integrated_params = step.get("integrated_parameters", {})
                    break
            
            if not integrated_params:
                self._log_ai_strategy_message("⚠️ 未找到优化后的参数，跳过集成", "WARNING")
                return
            
            self._log_ai_strategy_message("🔧 正在集成优化后的参数...", "INFO")
            
            # 集成参数
            result = self.parameter_integrator.integrate_parameters(integrated_params)
            
            if result.get("success"):
                self._log_ai_strategy_message(f"✅ 参数集成成功: {result.get('parameters_updated', 0)}个参数已更新", "SUCCESS")
                
                # 生成报告
                report = self.parameter_integrator.generate_integration_report(result)
                self._log_ai_strategy_message("=== 参数集成报告 ===", "INFO")
                for line in report.split('\n'):
                    if line.strip():
                        self._log_ai_strategy_message(f"  {line}", "INFO")
            else:
                error = result.get("error", "未知错误")
                self._log_ai_strategy_message(f"❌ 参数集成失败: {error}", "ERROR")
                
        except Exception as e:
            self._log_ai_strategy_message(f"❌ 参数集成失败: {e}", "ERROR")

    def _log_ai_strategy_message(self, message, level="INFO"):
        """记录AI策略中心消息"""
        try:
            # 记录到AI策略中心日志
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            if hasattr(self, 'ai_strategy_log_text'):
                self.ai_strategy_log_text.insert(tk.END, log_entry)
                self.ai_strategy_log_text.see(tk.END)
                
                # 根据级别应用颜色
                if level == "ERROR":
                    self.ai_strategy_log_text.tag_add("ERROR", f"end-2l linestart", f"end-2l lineend")
                elif level == "SUCCESS":
                    self.ai_strategy_log_text.tag_add("SUCCESS", f"end-2l linestart", f"end-2l lineend")
                elif level == "WARNING":
                    self.ai_strategy_log_text.tag_add("WARNING", f"end-2l linestart", f"end-2l lineend")
            
            # 同时记录到控制台（用于调试）
            print(f"[AI策略中心] {message}")
            
        except Exception as e:
            print(f"记录AI策略中心消息失败: {e}")

    def _update_confirm_counts_display(self):
        """更新确认次数显示（Spinbox值改变时调用）"""
        try:
            entry_count = self.entry_confirm_count_var.get()
            reverse_count = self.reverse_confirm_count_var.get()
            consecutive_pred = self.require_consecutive_prediction_var.get()
            post_entry_hours = self.post_entry_hours_var.get()
            take_profit_min_pct = self.take_profit_min_pct_var.get()
            
            # 这里可以添加实时显示更新逻辑
            # 暂时只记录日志
            self._log_ai_strategy_message(
                f"参数设置: 开仓{entry_count}次, 平仓{reverse_count}次, 连续预测{consecutive_pred}次, 开仓后计时{post_entry_hours}小时, 最小止盈{take_profit_min_pct}%", 
                "INFO"
            )
        except Exception as e:
            self._log_ai_strategy_message(f"更新参数显示失败: {e}", "ERROR")

    def _apply_confirm_counts(self):
        """应用确认次数设置到策略实例"""
        try:
            # 获取Spinbox的值
            entry_count = int(self.entry_confirm_count_var.get())
            reverse_count = int(self.reverse_confirm_count_var.get())
            consecutive_pred = int(self.require_consecutive_prediction_var.get())
            post_entry_hours = float(self.post_entry_hours_var.get())
            take_profit_min_pct = float(self.take_profit_min_pct_var.get())
            
            # 验证范围
            if not (1 <= entry_count <= 10):
                self._log_ai_strategy_message(f"开仓确认次数超出范围: {entry_count} (1-10)", "ERROR")
                return
            if not (1 <= reverse_count <= 10):
                self._log_ai_strategy_message(f"平仓确认次数超出范围: {reverse_count} (1-10)", "ERROR")
                return
            if not (1 <= consecutive_pred <= 10):
                self._log_ai_strategy_message(f"连续预测确认次数超出范围: {consecutive_pred} (1-10)", "ERROR")
                return
            if not (0.5 <= post_entry_hours <= 24):
                self._log_ai_strategy_message(f"开仓后计时超出范围: {post_entry_hours} (0.5-24)", "ERROR")
                return
            if not (0.1 <= take_profit_min_pct <= 10):
                self._log_ai_strategy_message(f"最小止盈比例超出范围: {take_profit_min_pct} (0.1-10)", "ERROR")
                return
            
            # 更新策略实例（如果存在）
            if hasattr(self, 'strategy') and self.strategy is not None:
                self.strategy.entry_confirm_count = entry_count
                self.strategy.reverse_confirm_count = reverse_count
                self.strategy.require_consecutive_prediction = consecutive_pred
                self.strategy.post_entry_hours = post_entry_hours
                self.strategy.take_profit_min_pct = take_profit_min_pct
                self.strategy.consecutive_entry_count = 0  # 重置计数器
                self.strategy.consecutive_reverse_count = 0  # 重置计数器
                self.strategy.last_entry_signal = None  # 重置信号
                self.strategy.last_reverse_signal = None  # 重置信号
                
                self._log_ai_strategy_message(
                    f"✅ 参数已应用到当前策略: 开仓{entry_count}次, 平仓{reverse_count}次, 连续预测{consecutive_pred}次, 开仓后计时{post_entry_hours}小时, 最小止盈{take_profit_min_pct}%", 
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  注意: 已重置连续计数器和最后信号记录", 
                    "INFO"
                )
            else:
                self._log_ai_strategy_message(
                    f"✅ 参数已保存: 开仓{entry_count}次, 平仓{reverse_count}次, 连续预测{consecutive_pred}次, 开仓后计时{post_entry_hours}小时, 最小止盈{take_profit_min_pct}%", 
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  将在下次启动交易时生效", 
                    "INFO"
                )
            
            # 保存设置到配置文件
            self._save_confirm_counts_settings(entry_count, reverse_count, consecutive_pred, post_entry_hours, take_profit_min_pct)
            
        except ValueError as e:
            self._log_ai_strategy_message(f"数值转换失败: {e} (请输入1-10的整数)", "ERROR")
        except Exception as e:
            self._log_ai_strategy_message(f"应用确认次数失败: {e}", "ERROR")

    def _save_confirm_counts_settings(self, entry_count, reverse_count, consecutive_pred, post_entry_hours, take_profit_min_pct):
        """保存确认次数设置到配置文件"""
        try:
            settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")
            
            # 加载现有设置
            settings = {}
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            
            # 更新确认次数设置
            if 'confirm_counts' not in settings:
                settings['confirm_counts'] = {}
            
            settings['confirm_counts']['entry_confirm_count'] = entry_count
            settings['confirm_counts']['reverse_confirm_count'] = reverse_count
            settings['confirm_counts']['require_consecutive_prediction'] = consecutive_pred
            settings['confirm_counts']['post_entry_hours'] = post_entry_hours
            settings['confirm_counts']['take_profit_min_pct'] = take_profit_min_pct
            settings['confirm_counts']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存设置
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
                
            self._log_ai_strategy_message(f"设置已保存到配置文件: {settings_file}", "INFO")
            
        except Exception as e:
            self._log_ai_strategy_message(f"保存设置失败: {e}", "WARNING")

    def _load_confirm_counts_settings(self):
        """从配置文件加载确认次数设置"""
        try:
            settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")
            
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                if 'confirm_counts' in settings:
                    confirm_counts = settings['confirm_counts']
                    entry_count = confirm_counts.get('entry_confirm_count', 2)
                    reverse_count = confirm_counts.get('reverse_confirm_count', 2)
                    consecutive_pred = confirm_counts.get('require_consecutive_prediction', 3)
                    post_entry_hours = confirm_counts.get('post_entry_hours', 2.0)
                    take_profit_min_pct = confirm_counts.get('take_profit_min_pct', 0.6)
                    
                    self.entry_confirm_count_var.set(str(entry_count))
                    self.reverse_confirm_count_var.set(str(reverse_count))
                    self.require_consecutive_prediction_var.set(str(consecutive_pred))
                    self.post_entry_hours_var.set(str(post_entry_hours))
                    self.take_profit_min_pct_var.set(str(take_profit_min_pct))
                    
                    self._log_ai_strategy_message(
                        f"已加载参数设置: 开仓{entry_count}次, 平仓{reverse_count}次, 连续预测{consecutive_pred}次, 开仓后计时{post_entry_hours}小时, 最小止盈{take_profit_min_pct}%", 
                        "INFO"
                    )
                    return True
            
            return False
            
        except Exception as e:
            self._log_ai_strategy_message(f"加载设置失败: {e}", "WARNING")
            return False

    def _create_main_backtest_tab(self, parent):
        """创建主回测标签页"""
        frame = ttk.Frame(parent, padding="15")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(
            frame,
            text="📊 策略回测系统",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(pady=(0, 15))
        
        # 回测参数配置区域
        config_frame = ttk.LabelFrame(frame, text="⚙️ 回测参数配置", padding="12")
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 数据文件选择
        load_row = ttk.Frame(config_frame)
        load_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_row, text="历史数据文件:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_data_file_var = tk.StringVar(value="")
        data_file_entry = ttk.Entry(load_row, textvariable=self.backtest_data_file_var, width=50)
        data_file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_row, text="📂 选择文件", command=self._select_backtest_data_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_row, text="📋 刷新列表", command=self._refresh_backtest_data_list).pack(side=tk.LEFT, padx=5)
        
        # 时间范围选择
        time_range_frame = ttk.LabelFrame(config_frame, text="📅 数据抓取时间范围", padding="10")
        time_range_frame.pack(fill=tk.X, pady=10)
        
        # 开始时间
        start_time_row = ttk.Frame(time_range_frame)
        start_time_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(start_time_row, text="开始时间:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_start_year_var = tk.StringVar(value="")
        self.backtest_start_month_var = tk.StringVar(value="")
        self.backtest_start_day_var = tk.StringVar(value="")
        
        ttk.Label(start_time_row, text="年:").pack(side=tk.LEFT, padx=(10, 0))
        year_values = [str(y) for y in range(2020, 2027)]
        start_year_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_year_var, values=year_values, width=6, state="readonly")
        start_year_combobox.pack(side=tk.LEFT)
        
        ttk.Label(start_time_row, text="月:").pack(side=tk.LEFT, padx=(10, 0))
        month_values = [f"{m:02d}" for m in range(1, 13)]
        start_month_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_month_var, values=month_values, width=4, state="readonly")
        start_month_combobox.pack(side=tk.LEFT)
        
        ttk.Label(start_time_row, text="日:").pack(side=tk.LEFT, padx=(10, 0))
        day_values = [f"{d:02d}" for d in range(1, 32)]
        start_day_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_day_var, values=day_values, width=4, state="readonly")
        start_day_combobox.pack(side=tk.LEFT)
        
        # 结束时间
        end_time_row = ttk.Frame(time_range_frame)
        end_time_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(end_time_row, text="结束时间:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_end_year_var = tk.StringVar(value="")
        self.backtest_end_month_var = tk.StringVar(value="")
        self.backtest_end_day_var = tk.StringVar(value="")
        
        ttk.Label(end_time_row, text="年:").pack(side=tk.LEFT, padx=(10, 0))
        end_year_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_year_var, values=year_values, width=6, state="readonly")
        end_year_combobox.pack(side=tk.LEFT)
        
        ttk.Label(end_time_row, text="月:").pack(side=tk.LEFT, padx=(10, 0))
        end_month_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_month_var, values=month_values, width=4, state="readonly")
        end_month_combobox.pack(side=tk.LEFT)
        
        ttk.Label(end_time_row, text="日:").pack(side=tk.LEFT, padx=(10, 0))
        end_day_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_day_var, values=day_values, width=4, state="readonly")
        end_day_combobox.pack(side=tk.LEFT)
        
        # 设置默认时间（最近30天）
        from datetime import datetime, timedelta
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        self.backtest_start_year_var.set(str(thirty_days_ago.year))
        self.backtest_start_month_var.set(f"{thirty_days_ago.month:02d}")
        self.backtest_start_day_var.set(f"{thirty_days_ago.day:02d}")
        self.backtest_end_year_var.set(str(today.year))
        self.backtest_end_month_var.set(f"{today.month:02d}")
        self.backtest_end_day_var.set(f"{today.day:02d}")
        
        # 参数行2：初始资金
        config_row2 = ttk.Frame(config_frame)
        config_row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_row2, text="初始资金(USDT):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_capital_var = tk.StringVar(value="10000")
        capital_entry = ttk.Entry(config_row2, textvariable=self.backtest_capital_var, width=15)
        capital_entry.pack(side=tk.LEFT, padx=5)
        
        # 参数行3：手续费率
        config_row3 = ttk.Frame(config_frame)
        config_row3.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_row3, text="手续费率(%):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_fee_var = tk.StringVar(value="0.1")
        fee_spinbox = ttk.Spinbox(config_row3, from_=0, to=1, increment=0.01, textvariable=self.backtest_fee_var, width=12)
        fee_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 参数行4：滑点
        config_row4 = ttk.Frame(config_frame)
        config_row4.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_row4, text="滑点(%):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_slippage_var = tk.StringVar(value="0.05")
        slippage_spinbox = ttk.Spinbox(config_row4, from_=0, to=1, increment=0.01, textvariable=self.backtest_slippage_var, width=12)
        slippage_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 操作按钮区
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.run_backtest_btn = ttk.Button(
            button_frame,
            text="▶️ 开始回测",
            command=self._run_backtest,
            style="Accent.TButton",
            width=20
        )
        self.run_backtest_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_backtest_btn = ttk.Button(
            button_frame,
            text="⏹️ 停止回测",
            command=self._stop_backtest,
            style="Accent.TButton",
            width=20,
            state="disabled"
        )
        self.stop_backtest_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📥 抓取储存数据",
            command=self._download_historical_data,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📋 导出结果",
            command=self._export_backtest_results,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="🗑️ 清空结果",
            command=self._clear_backtest_results,
            width=15
        ).pack(side=tk.LEFT)
        
        # 进度条
        progress_frame = ttk.Frame(frame)
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(progress_frame, text="回测进度:").pack(side=tk.LEFT, padx=(0, 10))
        self.backtest_progress = ttk.Progressbar(progress_frame, mode="determinate", length=400, style="green.Horizontal.TProgressbar")
        self.backtest_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.backtest_progress_label = ttk.Label(progress_frame, text="0%", width=6)
        self.backtest_progress_label.pack(side=tk.LEFT, padx=10)
        
        # 回测结果统计区
        stats_frame = ttk.LabelFrame(frame, text="📈 回测结果统计", padding="12")
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 统计行1
        stats_row1 = ttk.Frame(stats_frame)
        stats_row1.pack(fill=tk.X, pady=5)
        
        self.backtest_total_return_var = tk.StringVar(value="-")
        self.backtest_win_rate_var = tk.StringVar(value="-")
        self.backtest_profit_factor_var = tk.StringVar(value="-")
        self.backtest_max_drawdown_var = tk.StringVar(value="-")
        
        self._create_stat_item(stats_row1, "总收益率", self.backtest_total_return_var, "#27ae60")
        self._create_stat_item(stats_row1, "胜率", self.backtest_win_rate_var, "#3498db")
        self._create_stat_item(stats_row1, "盈亏比", self.backtest_profit_factor_var, "#9b59b6")
        self._create_stat_item(stats_row1, "最大回撤", self.backtest_max_drawdown_var, "#e74c3c")
        
        # 统计行2
        stats_row2 = ttk.Frame(stats_frame)
        stats_row2.pack(fill=tk.X, pady=5)
        
        self.backtest_total_trades_var = tk.StringVar(value="-")
        self.backtest_win_trades_var = tk.StringVar(value="-")
        self.backtest_loss_trades_var = tk.StringVar(value="-")
        self.backtest_avg_profit_var = tk.StringVar(value="-")
        
        self._create_stat_item(stats_row2, "总交易次数", self.backtest_total_trades_var, "#2c3e50")
        self._create_stat_item(stats_row2, "盈利次数", self.backtest_win_trades_var, "#27ae60")
        self._create_stat_item(stats_row2, "亏损次数", self.backtest_loss_trades_var, "#e74c3c")
        self._create_stat_item(stats_row2, "平均盈利", self.backtest_avg_profit_var, "#f39c12")
        
        # 回测交易日志区
        log_frame = ttk.LabelFrame(frame, text="📝 交易记录", padding="12")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.backtest_log_text = tk.Text(
            log_frame,
            height=12,
            font=("Consolas", 11),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        
        # 传统滚动条（更宽更明显）
        log_scrollbar = tk.Scrollbar(
            log_frame, 
            orient=tk.VERTICAL, 
            command=self.backtest_log_text.yview,
            width=25,
            relief=tk.SUNKEN,
            bg="#d3d3d3",
            activebackground="#a9a9a9"
        )
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.backtest_log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # 布局
        self.backtest_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _create_stat_item(self, parent, label_text, var, color):
        """创建统计项组件"""
        item_frame = ttk.Frame(parent)
        item_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(item_frame, text=label_text, font=("微软雅黑", 9), foreground="#7f8c8d").pack(anchor=tk.W)
        ttk.Label(item_frame, textvariable=var, font=("微软雅黑", 14, "bold"), foreground=color).pack(anchor=tk.W)
    
    def _run_backtest(self):
        """运行回测"""
        import threading
        self.run_backtest_btn.config(state="disabled")
        self.stop_backtest_btn.config(state="normal")
        self.backtest_progress["value"] = 0
        self.backtest_progress_label.config(text="0%")
        
        try:
            data_file_path = self.backtest_data_file_var.get().strip()
            if not data_file_path:
                messagebox.showwarning("警告", "请先选择历史数据文件！")
                self.run_backtest_btn.config(state="normal")
                self.stop_backtest_btn.config(state="disabled")
                return
            
            capital = float(self.backtest_capital_var.get())
            fee_rate = float(self.backtest_fee_var.get()) / 100
            slippage = float(self.backtest_slippage_var.get()) / 100
            
            self.backtest_log_text.insert(tk.END, f"[开始回测] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.backtest_log_text.insert(tk.END, f"数据获取方式: 载入历史数据\n")
            self.backtest_log_text.insert(tk.END, f"数据文件: {data_file_path}\n")
            self.backtest_log_text.insert(tk.END, f"参数: 资金=${capital:.2f}, 手续费={fee_rate*100:.2f}%, 滑点={slippage*100:.2f}%\n")
            
            self.backtest_log_text.insert(tk.END, "="*80 + "\n")
            
            import threading
            self.backtest_stop_event = threading.Event()
            self.backtest_stop_event.clear()
            
            backtest_thread = threading.Thread(
                target=self._simulate_backtest,
                args=(capital, fee_rate, slippage, data_file_path)
            )
            backtest_thread.daemon = True
            backtest_thread.start()
            
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[错误] {e}\n")
            import traceback
            traceback.print_exc()
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
    
    def _simulate_backtest(self, initial_capital, fee_rate, slippage, data_file_path):
        """模拟回测核心逻辑 - 使用真实策略回测"""
        from datetime import datetime, timedelta
        from binance_api import BinanceAPI
        from professional_strategy import ProfessionalTradingStrategy
        import pandas as pd
        import os
        
        self.backtest_log_text.insert(tk.END, "获取历史数据...\n")
        self._update_progress(10)
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("获取历史数据...")
        
        df = None
        
        self.backtest_log_text.insert(tk.END, "[载入数据] 正在从文件加载数据...\n")
        try:
            if not os.path.exists(data_file_path):
                self.backtest_log_text.insert(tk.END, f"[错误] 文件不存在: {data_file_path}\n")
                self._update_progress(100)
                self.run_backtest_btn.config(state="normal")
                self.stop_backtest_btn.config(state="disabled")
                return
            
            df = pd.read_csv(data_file_path, encoding='utf-8')
            self.backtest_log_text.insert(tk.END, f"[载入数据] 成功加载 {len(df)} 条数据\n")
            if len(df) > 0 and 'timestamps' in df.columns:
                self.backtest_log_text.insert(tk.END, f"[数据] 最早时间: {df['timestamps'].iloc[0]}\n")
                self.backtest_log_text.insert(tk.END, f"[数据] 最晚时间: {df['timestamps'].iloc[-1]}\n")
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[错误] 载入数据失败: {e}\n")
            import traceback
            self.backtest_log_text.insert(tk.END, f"{traceback.format_exc()}\n")
            self._update_progress(100)
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
            return
        
        if df is None or len(df) < 100:
            self.backtest_log_text.insert(tk.END, f"K线数据不足，无法回测 (df={df is not None}, len={len(df) if df is not None else 0})\n")
            self.backtest_log_text.insert(tk.END, f"[错误] K线数据不足，至少需要100条 (当前: {len(df) if df is not None else 0})\n")
            self._update_progress(100)
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
            return
        
        self.backtest_log_text.insert(tk.END, f"成功获取 {len(df)} 条K线数据\n")
        self.backtest_log_text.insert(tk.END, f"[数据] 获取到 {len(df)} 条K线数据\n")
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info(f"成功获取 {len(df)} 条K线数据")
        
        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        self.backtest_log_text.insert(tk.END, "[开始回测] 正在运行策略回测...\n")
        
        self._update_progress(20)
        self.backtest_log_text.insert(tk.END, "开始运行策略回测...\n")
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("[开始回测] 正在运行策略回测...")
        
        def progress_callback(value):
            self._update_progress(20 + int(value * 0.7))
        
        def log_callback(msg):
            self.backtest_log_text.insert(tk.END, f"{msg}\n")
            if hasattr(self, "backtest_logger"):
                self.backtest_logger.info(msg)
            scroll_position = self.backtest_log_text.yview()
            if scroll_position[1] >= 0.99:
                self.backtest_log_text.see(tk.END)
        
        if not hasattr(self, 'strategy') or not self.strategy:
            self.backtest_log_text.insert(tk.END, "[策略创建] 策略不存在，正在创建...\n")
            symbol = self.symbol_var.get()
            model_name = self.model_var.get()
            timeframe = self.timeframe_var.get()
            leverage = int(self.leverage_var.get())
            interval = int(self.interval_var.get())
            min_position = float(self.min_position_var.get())
            ai_min_trend = float(self.ai_min_trend_var.get())
            ai_min_deviation = float(self.ai_min_deviation_var.get())
            max_funding = float(self.max_funding_var.get())
            min_funding = float(self.min_funding_var.get())
            
            strategy_map = {
                "趋势爆发": "trend",
                "震荡套利": "range",
                "消息突破": "breakout",
                "自动策略": "auto",
                "时间策略": "time",
            }
            strategy_type = strategy_map.get(self.strategy_var.get(), "trend")
            
            # 5套交易策略强制使用5m时间周期（因为模型训练数据是5m）
            is_professional_strategy = self.strategy_var.get() in strategy_map
            if is_professional_strategy:
                if timeframe != "5m":
                    self.backtest_log_text.insert(tk.END, f"[策略创建] 检测到选择了{self.strategy_var.get()}策略，强制使用5m时间周期\n")
                    timeframe = "5m"
                    self.timeframe_var.set("5m")

            BinanceAPI()
            
            # 获取完整的AI策略配置
            ai_strategy_config = None
            if hasattr(self, "_get_ai_strategy_config_from_ui"):
                try:
                    ai_strategy_config = self._get_ai_strategy_config_from_ui()
                    self.backtest_log_text.insert(tk.END, "[策略创建] 已获取AI策略配置\n")
                except Exception as e:
                    self.backtest_log_text.insert(tk.END, f"[策略创建] 获取AI策略配置失败: {e}\n")
            
            def analysis_callback(history_timestamps=None, history_prices=None, 
                                  pred_timestamps=None, pred_prices=None,
                                  trend_direction='NEUTRAL', trend_strength=0, 
                                  pred_change=0, threshold=0.008):
                pass
            
            self.strategy = ProfessionalTradingStrategy(
                symbol=symbol,
                leverage=leverage,
                interval=interval,
                model_name=model_name,
                timeframe=timeframe,
                threshold=float(self.threshold_var.get()),
                strategy_type=strategy_type,
                min_position=min_position,
                ai_min_trend=ai_min_trend,
                ai_min_deviation=ai_min_deviation,
                max_funding=max_funding,
                min_funding=min_funding,
                analysis_callback=analysis_callback,
                strategy_config=ai_strategy_config,
                backtest_mode=True,
                log_callback=log_callback
            )
            
            self.backtest_log_text.insert(tk.END, "[策略创建] 策略创建成功！\n")
            self.backtest_log_text.insert(tk.END, "[策略配置] 正在加载策略预设配置...\n")
            
            # 加载策略预设的完整配置（包括止盈止损参数等）
            from professional_strategy import StrategyProfiles
            self.strategy.strategy_profile = StrategyProfiles.get_profile(strategy_type)
            self.strategy._load_strategy_config()
            
            self.backtest_log_text.insert(tk.END, "[策略配置] 策略预设配置加载完成！\n")
            if hasattr(self, "backtest_logger"):
                self.backtest_logger.info("[策略创建] 策略创建成功！")
        
        try:
            results = self.strategy.run_backtest(
                df_historical=df,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage=slippage,
                progress_callback=progress_callback,
                log_callback=log_callback,
                stop_event=self.backtest_stop_event
            )
            
            self._update_progress(100)
            self._display_real_backtest_results(results)
            self.backtest_log_text.insert(tk.END, "回测完成！\n")
            if hasattr(self, "backtest_logger"):
                self.backtest_logger.info("回测完成！")
            
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[错误] 回测执行失败: {e}\n")
            import traceback
            traceback.print_exc()
        finally:
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
    
    def _display_real_backtest_results(self, results):
        """显示真实回测结果"""
        total_return = results['total_return']
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        max_drawdown = results['max_drawdown']
        total_trades = results['total_trades']
        win_trades = results['win_trades']
        loss_trades = results['loss_trades']
        avg_profit = results['avg_profit']
        
        self.backtest_total_return_var.set(f"{total_return:+.2%}")
        self.backtest_win_rate_var.set(f"{win_rate:.2%}")
        self.backtest_profit_factor_var.set(f"{profit_factor:.2f}")
        self.backtest_max_drawdown_var.set(f"{-max_drawdown:.2%}")
        self.backtest_total_trades_var.set(f"{total_trades}")
        self.backtest_win_trades_var.set(f"{win_trades}")
        self.backtest_loss_trades_var.set(f"{loss_trades}")
        self.backtest_avg_profit_var.set(f"{avg_profit:+.2f}%")
        
        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        self.backtest_log_text.insert(tk.END, "📊 回测结果统计\n")
        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        self.backtest_log_text.insert(tk.END, f"初始资金: ${results['initial_capital']:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"最终资金: ${results['final_capital']:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"总收益率: {total_return:+.2%}\n")
        self.backtest_log_text.insert(tk.END, f"胜率: {win_rate:.2%}\n")
        self.backtest_log_text.insert(tk.END, f"盈亏比: {profit_factor:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"最大回撤: {-max_drawdown:.2%}\n")
        self.backtest_log_text.insert(tk.END, f"总交易次数: {total_trades}\n")
        self.backtest_log_text.insert(tk.END, f"盈利次数: {win_trades}\n")
        self.backtest_log_text.insert(tk.END, f"亏损次数: {loss_trades}\n")
        self.backtest_log_text.insert(tk.END, f"平均盈利: {avg_profit:+.2f}%\n")
        self.backtest_log_text.insert(tk.END, f"总手续费: ${results['total_fees']:.2f}\n")
        
        # 同时记录到回测日志文件
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("="*80)
            self.backtest_logger.info("📊 回测结果统计")
            self.backtest_logger.info("="*80)
            self.backtest_logger.info(f"初始资金: ${results['initial_capital']:.2f}")
            self.backtest_logger.info(f"最终资金: ${results['final_capital']:.2f}")
            self.backtest_logger.info(f"总收益率: {total_return:+.2%}")
            self.backtest_logger.info(f"胜率: {win_rate:.2%}")
            self.backtest_logger.info(f"盈亏比: {profit_factor:.2f}")
            self.backtest_logger.info(f"最大回撤: {-max_drawdown:.2%}")
            self.backtest_logger.info(f"总交易次数: {total_trades}")
            self.backtest_logger.info(f"盈利次数: {win_trades}")
            self.backtest_logger.info(f"亏损次数: {loss_trades}")
            self.backtest_logger.info(f"平均盈利: {avg_profit:+.2f}%")
            self.backtest_logger.info(f"总手续费: ${results['total_fees']:.2f}")
        
        self.backtest_log_text.insert(tk.END, "="*80 + "\n\n")
        
        self.backtest_log_text.insert(tk.END, "[详细交易记录]\n")
        for i, trade in enumerate(results['trades'][:50], 1):
            if trade['type'] == 'OPEN':
                self.backtest_log_text.insert(tk.END, f"{i}. [开仓] {trade['time']} - {trade['direction']} - 价格: {trade['price']:.2f} - 数量: {trade['size']:.4f}\n")
                if hasattr(self, "backtest_logger"):
                    self.backtest_logger.info(f"{i}. [开仓] {trade['time']} - {trade['direction']} - 价格: {trade['price']:.2f} - 数量: {trade['size']:.4f}")
            else:
                self.backtest_log_text.insert(tk.END, f"{i}. [平仓] {trade['time']} - {trade['direction']} - 入场: {trade['entry_price']:.2f} - 出场: {trade['exit_price']:.2f} - 盈亏: {trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)\n")
                if hasattr(self, "backtest_logger"):
                    self.backtest_logger.info(f"{i}. [平仓] {trade['time']} - {trade['direction']} - 入场: {trade['entry_price']:.2f} - 出场: {trade['exit_price']:.2f} - 盈亏: {trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)")
        
        if len(results['trades']) > 50:
            self.backtest_log_text.insert(tk.END, f"... (还有 {len(results['trades']) - 50} 笔交易)\n")
        
        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        # 只有当滚动条在最底部时才自动跟随
        scroll_position = self.backtest_log_text.yview()
        if scroll_position[1] >= 0.9:
            self.backtest_log_text.see(tk.END)
    
    def _update_progress(self, value):
        """更新进度条"""
        self.backtest_progress["value"] = value
        self.backtest_progress_label.config(text=f"{value}%")
        self.backtest_progress.update()
    
    def _export_backtest_results(self):
        """导出回测结果"""
        from tkinter import filedialog
        try:
            content = self.backtest_log_text.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("提示", "没有可导出的回测结果！")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="导出回测结果",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("成功", "回测结果已导出！")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
    
    def _stop_backtest(self):
        """停止回测"""
        if hasattr(self, 'backtest_stop_event'):
            self.backtest_stop_event.set()
            self.backtest_log_text.insert(tk.END, f"[停止回测] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 正在停止...\n")
            self.stop_backtest_btn.config(state="disabled")
    
    def _clear_backtest_results(self):
        """清空回测结果"""
        if messagebox.askyesno("确认", "确定要清空回测结果吗？"):
            self.backtest_log_text.delete("1.0", tk.END)
            self.backtest_total_return_var.set("-")
            self.backtest_win_rate_var.set("-")
            self.backtest_profit_factor_var.set("-")
            self.backtest_max_drawdown_var.set("-")
            self.backtest_total_trades_var.set("-")
            self.backtest_win_trades_var.set("-")
            self.backtest_loss_trades_var.set("-")
            self.backtest_avg_profit_var.set("-")
            self.backtest_progress["value"] = 0
            self.backtest_progress_label.config(text="0%")
            self.log("回测结果已清空")
    
    def _get_backtest_data_dir(self):
        """获取回测数据存储目录"""
        import os
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
    
    def _select_backtest_data_file(self):
        """选择历史数据文件"""
        from tkinter import filedialog
        data_dir = self._get_backtest_data_dir()
        file_path = filedialog.askopenfilename(
            title="选择历史数据文件",
            initialdir=data_dir,
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.backtest_data_file_var.set(file_path)
    
    def _refresh_backtest_data_list(self):
        """刷新历史数据文件列表"""
        import os
        data_dir = self._get_backtest_data_dir()
        try:
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                self.backtest_log_text.insert(tk.END, f"[数据文件] 找到 {len(files)} 个历史数据文件:\n")
                for f in sorted(files):
                    self.backtest_log_text.insert(tk.END, f"  - {f}\n")
            else:
                self.backtest_log_text.insert(tk.END, "[数据文件] 未找到历史数据文件\n")
            self.backtest_log_text.see(tk.END)
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[错误] 刷新文件列表失败: {e}\n")
    
    def _download_historical_data(self):
        """按时间范围下载历史数据并保存，包含27个技术指标"""
        import threading
        import os
        from datetime import datetime
        from binance_api import BinanceAPI
        
        def download_data_thread():
            try:
                self.backtest_log_text.insert(tk.END, "[抓取数据] 开始获取历史数据...\n")
                
                start_year = self.backtest_start_year_var.get().strip()
                start_month = self.backtest_start_month_var.get().strip()
                start_day = self.backtest_start_day_var.get().strip()
                end_year = self.backtest_end_year_var.get().strip()
                end_month = self.backtest_end_month_var.get().strip()
                end_day = self.backtest_end_day_var.get().strip()
                
                if not (start_year and start_month and start_day and end_year and end_month and end_day):
                    self.backtest_log_text.insert(tk.END, "[错误] 请选择完整的开始时间和结束时间！\n")
                    return
                
                start_time_str = f"{start_year}-{start_month}-{start_day} 00:00"
                end_time_str = f"{end_year}-{end_month}-{end_day} 23:59"
                
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 时间范围: {start_time_str} 至 {end_time_str}\n")
                
                if not hasattr(self, 'strategy') or not self.strategy:
                    self.backtest_log_text.insert(tk.END, "[错误] 策略未初始化，请先运行一次回测\n")
                    return
                
                self.backtest_log_text.insert(tk.END, "[抓取数据] 正在获取K线数据...\n")
                
                df = self.strategy.binance.get_historical_klines(
                    self.strategy.symbol, 
                    self.strategy.timeframe, 
                    start_str=start_time_str,
                    end_str=end_time_str
                )
                
                if df is None or len(df) == 0:
                    self.backtest_log_text.insert(tk.END, "[错误] 获取数据失败\n")
                    return
                
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 成功获取 {len(df)} 条原始K线数据\n")
                
                self.backtest_log_text.insert(tk.END, "[抓取数据] 正在计算27个技术指标...\n")
                
                df = df.copy()
                
                if 'volume' in df.columns:
                    df['vol'] = df['volume']
                if 'amount' in df.columns:
                    df['amt'] = df['amount']
                
                close = df["close"].values
                high = df["high"].values
                low = df["low"].values
                amount = df["amt"].values
                
                df["MA5"] = pd.Series(close).rolling(window=5).mean().values
                df["MA10"] = pd.Series(close).rolling(window=10).mean().values
                df["MA20"] = pd.Series(close).rolling(window=20).mean().values
                
                df["BIAS20"] = (close / df["MA20"] - 1) * 100
                
                tr1 = high - low
                tr2 = np.abs(high - np.roll(close, 1))
                tr3 = np.abs(low - np.roll(close, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = 0
                df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values
                
                df["AMPLITUDE"] = (high - low) / close * 100
                
                df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
                df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values
                df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]
                
                delta = pd.Series(close).diff()
                gain = delta.where(delta > 0, 0)
                loss = (-delta.where(delta < 0, 0))
                avg_gain14 = gain.rolling(window=14).mean()
                avg_loss14 = loss.rolling(window=14).mean()
                rs14 = avg_gain14 / avg_loss14
                df["RSI14"] = (100 - (100 / (1 + rs14))).values
                
                avg_gain7 = gain.rolling(window=7).mean()
                avg_loss7 = loss.rolling(window=7).mean()
                rs7 = avg_gain7 / avg_loss7
                df["RSI7"] = (100 - (100 / (1 + rs7))).values
                
                ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
                ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
                df["MACD"] = ema12 - ema26
                df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
                df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
                
                df["PRICE_SLOPE5"] = (
                    pd.Series(close)
                    .rolling(window=5)
                    .apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                        raw=True,
                    )
                    .values
                )
                df["PRICE_SLOPE10"] = (
                    pd.Series(close)
                    .rolling(window=10)
                    .apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                        raw=True,
                    )
                    .values
                )
                
                df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
                df["LOW5"] = pd.Series(low).rolling(window=5).min().values
                df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
                df["LOW10"] = pd.Series(low).rolling(window=10).min().values
                
                df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int)
                df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int)
                
                feature_list = [
                    "open", "high", "low", "close", "vol", "amt", 
                    "MA5", "MA10", "MA20",
                    "BIAS20",
                    "ATR14", "AMPLITUDE",
                    "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",
                    "RSI14", "RSI7",
                    "MACD", "MACD_HIST",
                    "PRICE_SLOPE5", "PRICE_SLOPE10",
                    "HIGH5", "LOW5", "HIGH10", "LOW10",
                    "VOL_BREAKOUT", "VOL_SHRINK"
                ]
                
                for col in feature_list:
                    if col not in df.columns:
                        df[col] = 0.0
                
                time_col = None
                if "timestamps" in df.columns:
                    time_col = "timestamps"
                elif "datetime" in df.columns:
                    time_col = "datetime"
                
                if time_col:
                    df = df[[time_col] + feature_list]
                else:
                    df = df[feature_list]
                
                df = df.dropna()
                
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 成功计算27个技术指标，剩余 {len(df)} 条有效数据\n")
                
                data_dir = self._get_backtest_data_dir()
                
                if len(df) > 0 and 'timestamps' in df.columns:
                    first_time = df['timestamps'].iloc[0]
                    last_time = df['timestamps'].iloc[-1]
                    
                    first_time_str = str(first_time).replace(':', '-').replace(' ', '_')
                    last_time_str = str(last_time).replace(':', '-').replace(' ', '_')
                    
                    filename = f"{self.strategy.symbol}_{self.strategy.timeframe}_{first_time_str}_to_{last_time_str}.csv"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.strategy.symbol}_{self.strategy.timeframe}_{timestamp}.csv"
                
                file_path = os.path.join(data_dir, filename)
                
                df.to_csv(file_path, index=False, encoding='utf-8')
                
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 数据已保存到: {file_path}\n")
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 文件大小: {os.path.getsize(file_path)} 字节\n")
                self.backtest_log_text.insert(tk.END, f"[抓取数据] 包含特征: {len(df.columns)} 个 (时间戳 + 27个技术指标)\n")
                
                self.backtest_data_file_var.set(file_path)
                
                messagebox.showinfo("成功", f"历史数据已抓取并保存！\n文件: {filename}\n包含27个技术指标")
                
            except Exception as e:
                self.backtest_log_text.insert(tk.END, f"[错误] 抓取数据失败: {e}\n")
                import traceback
                self.backtest_log_text.insert(tk.END, f"{traceback.format_exc()}\n")
        
        thread = threading.Thread(target=download_data_thread)
        thread.daemon = True
        thread.start()

    def _create_help_tab(self, parent):
        """创建帮助标签页"""
        try:
            # 创建滚动文本框
            help_text_frame = ttk.Frame(parent)
            help_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 创建滚动条
            scrollbar = ttk.Scrollbar(help_text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 创建文本框
            help_text = tk.Text(
                help_text_frame,
                wrap=tk.WORD,
                yscrollcommand=scrollbar.set,
                font=("微软雅黑", 14),
                bg="#f8f9fa",
                relief=tk.FLAT,
                padx=10,
                pady=10
            )
            help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=help_text.yview)
            
            # 加载帮助文档内容
            help_file_path = os.path.join(os.path.dirname(__file__), "帮助.txt")
            if os.path.exists(help_file_path):
                with open(help_file_path, 'r', encoding='utf-8') as f:
                    help_content = f.read()
                    help_text.insert(tk.END, help_content)
            else:
                help_text.insert(tk.END, "帮助文档未找到，请确保 '帮助.txt' 文件存在。")
            
            # 设置文本框为只读
            help_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"创建帮助标签页失败: {e}")


def main():
    root = tk.Tk()
    root.withdraw()  # 先隐藏窗口
    KronosTradingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
