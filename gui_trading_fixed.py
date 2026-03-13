 # -*- coding: utf-8 -*-
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

# 閰嶇疆鏃 織 - 鍏煎  PyInstaller 鎵撳寘鐜  锛堟棤缁堢 妯 紡锛?
log_file = None
try:
    # 濡傛灉鏄 墦鍖呭悗鐨勬棤缁堢 妯 紡锛屽垱寤烘棩蹇楁枃浠?
    if getattr(sys, 'frozen', False):
        import tempfile
        log_dir = os.path.join(os.path.dirname(sys.executable), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'kronos_trading.log')

    # 閰嶇疆鏃 織
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a',
            encoding='utf-8'
        )
    else:
        # 寮 鍙戞 寮忥紝杈撳嚭鍒版帶鍒跺彴
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
except Exception:
    # 澶囩敤鏂规 锛氫笉閰嶇疆鏃 織锛岄伩鍏嶅穿婧?
    pass

# 绂佺敤 transformers 鐨勮 缁嗘棩蹇?
try:
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.ERROR)
except Exception:
    pass

matplotlib.use("TkAgg")  # 璁剧疆鍚庣
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 鏂板 妯 潡瀵煎叆 - 澶氭櫤鑳戒綋閲忓寲浜 槗绯荤粺
try:
    from fingpt_analyzer import FinGPTSentimentAnalyzer
except ImportError as e:
    print(f"FinGPT妯 潡瀵煎叆澶辫触: {e}")
    FinGPTSentimentAnalyzer = None

try:
    from strategy_coordinator import StrategyCoordinator
except ImportError as e:
    print(f"绛栫暐鍗忚皟鍣  鍧楀 鍏  璐? {e}")
    StrategyCoordinator = None

# 鏂板 妯 潡瀵煎叆 - BTC鏂伴椈鐖 櫕
try:
    from btc_news_crawler import BTCNewsCrawler
except ImportError as e:
    print(f"BTC鏂伴椈鐖 櫕妯 潡瀵煎叆澶辫触: {e}")
    BTCNewsCrawler = None

# 鏂板 妯 潡瀵煎叆 - 鑷 姩鍖栦紭鍖栫郴缁?
try:
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"鎬 兘鐩戞帶妯 潡瀵煎叆澶辫触: {e}")
    PerformanceMonitor = None

try:
    from auto_optimization_pipeline import AutoOptimizationPipeline
except ImportError as e:
    print(f"鑷 姩鍖栦紭鍖栫 閬撴 鍧楀 鍏  璐? {e}")
    AutoOptimizationPipeline = None

try:
    from parameter_integrator import ParameterIntegrator
except ImportError as e:
    print(f"鍙傛暟闆嗘垚鍣  鍧楀 鍏  璐? {e}")
    ParameterIntegrator = None

# 閰嶇疆matplotlib涓 枃瀛椾綋锛堟渶鍙 潬鐨勬柟娉曪級
try:
    plt.rcParams["font.sans-serif"] = [
        "寰 蒋闆呴粦",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Arial Unicode MS",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False  # 瑙 喅璐熷彿鏄剧 闂
    # print("宸查厤缃甿atplotlib涓 枃瀛椾綋鏀 寔")
except Exception as e:
    print(f"閰嶇疆涓 枃瀛椾綋鏃跺嚭閿? {e}")

os.environ["HF_HUB_DISABLE_XET"] = "1"

# 灏濊瘯鍔犺浇.env鏂囦欢浠 幏鍙朒F_TOKEN
try:
    from dotenv import load_dotenv

    # 鍔犺浇.env鏂囦欢
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # 濡傛灉.env鏂囦欢涓  缃 簡HF_TOKEN锛屽垯璁剧疆鐜  鍙橀噺
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.strip() and not hf_token.startswith("your_"):
        os.environ["HF_TOKEN"] = hf_token
        print(f"宸茶 缃瓾F_TOKEN (闀垮害: {len(hf_token)})")
    else:
        print("鎻愮 : 濡傞渶鏇村揩鐨勬 鍨嬩笅杞介 熷害锛岃 鍦?env鏂囦欢涓  缃 湁鏁堢殑HF_TOKEN")
        print("鑾峰彇鍦板潃: https://huggingface.co/settings/tokens")
except ImportError:
    print("dotenv鏈 畨瑁咃紝璺宠繃.env鏂囦欢鍔犺浇")
except Exception as e:
    print(f"鍔犺浇HF_TOKEN鏃跺嚭閿? {e}")

LARGE_FONT = ("寰 蒋闆呴粦", 12)
BOLD_FONT = ("寰 蒋闆呴粦", 12, "bold")
TITLE_FONT = ("寰 蒋闆呴粦", 14, "bold")


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

                # 杩涘害妫 娴嬮 昏緫
                progress_detected = False

                # 1. 妫 娴?"100%" 妯 紡
                if "100%" in line:
                    if self.progress_callback:
                        self.progress_callback(100)
                        # self.queue.put(("log", f"[杩涘害] 100% 瀹屾垚"))
                    progress_detected = True

                # 2. 妫 娴?"it/s]" 妯 紡 (tqdm杩涘害鏉?
                elif "it/s]" in line:
                    try:
                        import re

                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if self.progress_callback:
                                self.progress_callback(pct)
                                # self.queue.put(("log", f"[杩涘害] {pct}%"))
                            progress_detected = True
                    except:
                        pass

                # 3. 妫 娴嬬畝鍗曠櫨鍒嗘瘮妯 紡 (渚嬪 : "杩涘害: 50%")
                elif "%" in line and not progress_detected:
                    try:
                        import re

                        # 鏌 壘浠讳綍鏁板瓧鍚庤窡%鐨勬 寮?
                        match = re.search(r"(\d+)%", line)
                        if match:
                            pct = int(match.group(1))
                            if 0 <= pct <= 100:
                                if self.progress_callback:
                                    self.progress_callback(pct)
                                    # self.queue.put(("log", f"[杩涘害] 妫 娴嬪埌鐧惧垎姣? {pct}%"))
                                progress_detected = True
                    except:
                        pass

                # 4. 妫 娴嬭繘搴 潯鍙  鍖栨 寮?(渚嬪 : "[====>    ] 50%")
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
                                    # self.queue.put(("log", f"[杩涘害] 妫 娴嬪埌鐧惧垎姣? {pct}%"))
                                progress_detected = True
                    except:
                        pass

                # 濡傛灉涓嶆槸杩涘害淇 伅锛屽垯浣滀负鏅  氭棩蹇楄緭鍑?
                if not progress_detected:
                    self.queue.put(("log", line))

    def flush(self):
        pass

    def isatty(self):
        return False  # OutputRedirector涓嶆槸缁堢 璁惧

    def fileno(self):
        # 杩斿洖涓 涓 湁鏁堢殑鏂囦欢鎻忚堪绗 紝閫氬父涓?(stdout)鎴?(stderr)
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
        # 杩欐槸鍘熸湁鐨剋rite鏂规硶锛岄渶瑕佷繚鎸?
        self.buffer += text
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]
            for line in lines[:-1]:
                if not line.strip():
                    continue

                # 杩涘害妫 娴嬮 昏緫
                progress_detected = False

                # 1. 妫 娴?"100%" 妯 紡
                if "100%" in line:
                    if self.progress_callback:
                        self.progress_callback(100)
                    progress_detected = True

                # 2. 妫 娴?"it/s]" 妯 紡 (tqdm杩涘害鏉?
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

                # 3. 妫 娴嬬畝鍗曠櫨鍒嗘瘮妯 紡
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

                # 4. 妫 娴嬭繘搴 潯鍙  鍖栨 寮?
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

                # 濡傛灉涓嶆槸杩涘害淇 伅锛屽垯浣滀负鏅  氭棩蹇楄緭鍑?
                if not progress_detected:
                    self.queue.put(("log", line))


class StrategyConfigDialog:
    """绛栫暐鍙傛暟閰嶇疆瀵硅瘽妗?"

    def __init__(self, parent, current_config=None):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("鈿欙笍 绛栫暐鍙傛暟閰嶇疆")
        self.dialog.geometry("1200x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.result = None
        self.current_config = current_config if current_config else self._get_default_config()

        # 鍒涘缓涓绘 鏋?
        self._create_main_frame()
        self._load_config_to_ui()

    def _get_default_config(self):
        """鑾峰彇榛樿 閰嶇疆"""
        try:
            from strategy_config import StrategyConfig
            trend_strength_threshold = StrategyConfig.TREND_STRENGTH_THRESHOLD
        except Exception:
            trend_strength_threshold = 0.0047

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
                "TREND_STRENGTH_THRESHOLD": trend_strength_threshold,
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
        """鍒涘缓涓绘 鏋?"
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 椤堕儴鎸夐挳鏍?
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # 棰勮 閫夋嫨
        ttk.Label(button_frame, text="馃搵 浜 槗椋庢牸棰勮 :").pack(side=tk.LEFT, padx=(0, 5))
        self.preset_var = tk.StringVar(value="骞宠 鍨?)"
        preset_combo = ttk.Combobox(
            button_frame,
            textvariable=self.preset_var,
            values=["婵 杩涜秴鐭 嚎", "瓒嬪娍杩借釜", "骞宠 鍨?, "闇囪崱濂楀埄", "绋冲仴闀跨嚎", "娑堟伅椹卞姩"],"
            state="readonly",
            width=15
        )
        preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="馃搧 杞藉叆閰嶇疆", command=self._load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="馃捑 淇濆瓨閰嶇疆", command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="鈫 笍 閲嶇疆榛樿 ", command=self._reset_default).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="鉁?搴旂敤", command=self._apply_config, style="Accent.TButton").pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="鉂?鍙栨秷", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 5))

        # 鍒涘缓Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 鍒涘缓鍚勪釜鏍囩 椤?
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
        """鍒涘缓鍙傛暟琛?"
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=2)

        # 鍙傛暟鍚嶇
        name_label = ttk.Label(row_frame, text=display_name, width=25, anchor=tk.W)
        name_label.pack(side=tk.LEFT, padx=(0, 5))

        # 杈撳叆鎺 欢
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

        # 鍙栧 艰寖鍥?
        if min_val is not None and max_val is not None:
            range_label = ttk.Label(row_frame, text=f"[{min_val}-{max_val}]", width=15, foreground="#7f8c8d")
            range_label.pack(side=tk.LEFT, padx=(0, 5))

        # 璇存槑
        desc_label = ttk.Label(row_frame, text=description, foreground="#7f8c8d", wraplength=500)
        desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_coordinator_tab(self):
        """鍒涘缓鍗忚皟鍣 弬鏁版爣绛鹃 """
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="1. 鍗忚皟鍣 弬鏁?)"

        ttk.Label(frame, text="銆愬崗璋冨櫒鍙傛暟銆?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "coordinator", "min_signal_strength", "鏈 灏忎俊鍙峰己搴?, "float", 0.0, 1.0, "鍙 湁淇 彿寮哄害瓒呰繃姝  兼墠瑙 彂浜 槗")"
        self._create_param_row(frame, "coordinator", "max_position_size", "鏈 澶 粨浣嶆瘮渚?, "float", 0.0, 1.0, "鍗曟 浜 槗鐨勬渶澶 粨浣嶆瘮渚?)
        self._create_param_row(frame, "coordinator", "sentiment_weight", "鑸嗘儏淇 彿鏉冮噸", "float", 0.0, 1.0, "FinGPT鑸嗘儏鍒嗘瀽鐨勬潈閲?)"
        self._create_param_row(frame, "coordinator", "technical_weight", "鎶 鏈 俊鍙锋潈閲?, "float", 0.0, 1.0, "Kronos鎶 鏈 垎鏋愮殑鏉冮噸")"
        self._create_param_row(frame, "coordinator", "black_swan_threshold", "榛戝 楣呴槇鍊?, "select", None, None, "鏋佺 琛屾儏鏁忔劅搴 槇鍊?)
        self._create_param_row(frame, "coordinator", "enable_adaptive_filtering", "鑷  傚簲杩囨护", "bool", None, None, "鍚 敤鍔  佸弬鏁拌皟鏁?)"

    def _create_basic_tab(self):
        """鍒涘缓鍩虹 鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="2. 鍩虹 鍙傛暟")

        ttk.Label(frame, text="銆愬熀纭 鍙傛暟銆?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "basic", "LEVERAGE", "鏉犳潌鍊嶆暟", "int", 1, 125, "浜 槗浣跨敤鐨勬潬鏉嗗 灏?)"
        self._create_param_row(frame, "basic", "LOOKBACK_PERIOD", "鍥炵湅K绾挎暟閲?, "int", 64, 2048, "鎶 鏈 垎鏋愪娇鐢 殑鍘嗗彶鏁版嵁闀垮害")"
        self._create_param_row(frame, "basic", "PREDICTION_LENGTH", "棰勬祴K绾挎暟閲?, "int", 4, 200, "Kronos棰勬祴鐨勬湭鏉 绾挎暟閲?)
        self._create_param_row(frame, "basic", "CHECK_INTERVAL", "妫 鏌 棿闅?绉?", "int", 30, 3600, "绯荤粺妫 鏌 氦鏄撲俊鍙风殑闂撮殧")

    def _create_entry_tab(self):
        """鍒涘缓鍏 満杩囨护鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="3. 鍏 満杩囨护")

        ttk.Label(frame, text="銆愬叆鍦鸿繃婊 弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "entry", "max_kline_change", "鏈 澶 崟K鍙樺寲", "float", 0.001, 0.05, "闄愬埗鍗曟牴K绾跨殑鏈 澶 定璺屽箙")
        self._create_param_row(frame, "entry", "max_funding_rate_long", "澶氬 鏈 澶 祫閲戣垂鐜?, "float", -0.05, 0.05, "寮 澶氫粨鏃惰祫閲戣垂鐜囦笂闄?)
        self._create_param_row(frame, "entry", "min_funding_rate_short", "绌哄 鏈 灏忚祫閲戣垂鐜?, "float", -0.05, 0.05, "寮 绌轰粨鏃惰祫閲戣垂鐜囦笅闄?)
        self._create_param_row(frame, "entry", "support_buffer", "鏀 拺浣嶇紦鍐?, "float", 1.000, 1.010, "鍦 敮鎾戜綅涓婃柟澶氬皯姣斾緥鍏 満")"
        self._create_param_row(frame, "entry", "resistance_buffer", "闃诲姏浣嶇紦鍐?, "float", 0.990, 1.000, "鍦 樆鍔涗綅涓嬫柟澶氬皯姣斾緥鍏 満")"

    def _create_stop_loss_tab(self):
        """鍒涘缓姝 崯鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="4. 姝 崯鍙傛暟")

        ttk.Label(frame, text="銆愭 鎹熷弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "stop_loss", "long_buffer", "澶氬 姝 崯缂撳啿", "float", 0.900, 0.999, "澶氬 姝 崯鐩稿 浜庡叆鍦轰环鐨勬瘮渚?)"
        self._create_param_row(frame, "stop_loss", "short_buffer", "绌哄 姝 崯缂撳啿", "float", 1.001, 1.100, "绌哄 姝 崯鐩稿 浜庡叆鍦轰环鐨勬瘮渚?)"

    def _create_take_profit_tab(self):
        """鍒涘缓姝 泩鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="5. 姝 泩鍙傛暟")

        ttk.Label(frame, text="銆愭 鐩堝弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "take_profit", "tp1_multiplier_long", "澶氬 绗 竴姝 泩", "float", 1.001, 1.100, "澶氬 绗 竴姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp2_multiplier_long", "澶氬 绗 簩姝 泩", "float", 1.001, 1.200, "澶氬 绗 簩姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp3_multiplier_long", "澶氬 绗 笁姝 泩", "float", 1.001, 1.300, "澶氬 绗 笁姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp1_multiplier_short", "绌哄 绗 竴姝 泩", "float", 0.900, 0.999, "绌哄 绗 竴姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp2_multiplier_short", "绌哄 绗 簩姝 泩", "float", 0.800, 0.999, "绌哄 绗 簩姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp3_multiplier_short", "绌哄 绗 笁姝 泩", "float", 0.700, 0.999, "绌哄 绗 笁姝 泩鐩 爣")
        self._create_param_row(frame, "take_profit", "tp1_position_ratio", "绗 竴姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 竴姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")
        self._create_param_row(frame, "take_profit", "tp2_position_ratio", "绗 簩姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 簩姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")
        self._create_param_row(frame, "take_profit", "tp3_position_ratio", "绗 笁姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 笁姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")

    def _create_risk_tab(self):
        """鍒涘缓椋庨櫓绠 悊鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="6. 椋庨櫓绠 悊")

        ttk.Label(frame, text="銆愰 闄  鐞嗗弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "risk", "single_trade_risk", "鍗曠瑪椋庨櫓姣斾緥", "float", 0.001, 0.10, "鍗曠瑪浜 槗鏈 澶 簭鎹熸瘮渚?)"
        self._create_param_row(frame, "risk", "daily_loss_limit", "姣忔棩浜忔崯闄愬埗", "float", 0.01, 0.20, "鍗曟棩绱  浜忔崯杈惧埌姝  煎仠姝 氦鏄?)"
        self._create_param_row(frame, "risk", "max_consecutive_losses", "鏈 澶 繛缁 簭鎹?, "int", 1, 10, "杩炵画浜忔崯娆 暟杈惧埌姝  兼殏鍋滀氦鏄?)
        self._create_param_row(frame, "risk", "max_single_position", "鏈 澶 崟绗斾粨浣?, "float", 0.01, 1.0, "鍗曠瑪浜 槗鐨勬渶澶 粨浣嶆瘮渚?)
        self._create_param_row(frame, "risk", "max_daily_position", "鏈 澶 棩浠撲綅", "float", 0.01, 1.0, "鍗曟棩绱  寮 浠撶殑鏈 澶 粨浣嶆瘮渚?)"

    def _create_frequency_tab(self):
        """鍒涘缓浜 槗棰戠巼鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="7. 浜 槗棰戠巼")

        ttk.Label(frame, text="銆愪氦鏄撻 鐜囧弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "frequency", "max_daily_trades", "姣忔棩鏈 澶 氦鏄?, "int", 1, 100, "闄愬埗鍗曟棩鏈 澶氫氦鏄撳 灏戞 ")"
        self._create_param_row(frame, "frequency", "min_trade_interval_minutes", "鏈 灏忛棿闅?鍒?", "int", 1, 60, "涓  浜 槗涔嬮棿鐨勬渶灏忛棿闅?)"
        self._create_param_row(frame, "frequency", "active_hours_start", "娲昏穬寮 濮嬫椂闂?, "int", 0, 23, "鍙 湪姝 椂闂翠箣鍚庤繘琛屼氦鏄?)
        self._create_param_row(frame, "frequency", "active_hours_end", "娲昏穬缁撴潫鏃堕棿", "int", 1, 24, "鍙 湪姝 椂闂翠箣鍓嶈繘琛屼氦鏄?)"

    def _create_position_tab(self):
        """鍒涘缓浠撲綅绠 悊鍙傛暟鏍囩 椤?"
        frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(frame, text="8. 浠撲綅绠 悊")

        ttk.Label(frame, text="銆愪粨浣嶇 鐞嗗弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_param_row(frame, "position", "initial_entry_ratio", "鍒濆 鍏 満姣斾緥", "float", 0.1, 1.0, "棣栨 寮 浠撴椂浣跨敤鐩 爣浠撲綅鐨勬瘮渚?)"
        self._create_param_row(frame, "position", "confirm_interval_kline", "纭  K绾挎暟閲?, "int", 1, 10, "鍒濆 鍏 満鍚庣瓑寰呭 灏戞牴K绾跨 璁 秼鍔?)
        self._create_param_row(frame, "position", "add_on_profit", "鐩堝埄鍔犱粨", "int", 0, 1, "鏄 惁鍦 泩鍒 椂鍔犱粨(0=鍚?1=鏄?")
        self._create_param_row(frame, "position", "add_ratio", "鍔犱粨姣斾緥", "float", 0.1, 0.5, "姣忔 鍔犱粨鍗犵洰鏍囦粨浣嶇殑姣斾緥")
        self._create_param_row(frame, "position", "max_add_times", "鏈 澶 姞浠撴 鏁?, "int", 1, 10, "鏈 澶氬彲浠 姞浠撳 灏戞 ")"

    def _load_config_to_ui(self):
        """灏嗛厤缃 姞杞藉埌UI"""
        for category, params in self.current_config.items():
            for param_name, value in params.items():
                var_key = f"{category}.{param_name}"
                if var_key in self.config_vars:
                    self.config_vars[var_key].set(value)

    def _get_config_from_ui(self):
        """浠嶶I鑾峰彇閰嶇疆"""
        config = {}
        for category in ["coordinator", "basic", "entry", "stop_loss", "take_profit", "risk", "frequency", "position"]:
            config[category] = {}

        for var_key, var in self.config_vars.items():
            category, param_name = var_key.split(".", 1)
            config[category][param_name] = var.get()

        return config

    def _load_config(self):
        """杞藉叆閰嶇疆鏂囦欢"""
        file_path = filedialog.askopenfilename(
            title="杞藉叆閰嶇疆",
            filetypes=[("YAML鏂囦欢", "*.yaml"), ("YAML鏂囦欢", "*.yml"), ("鎵 鏈夋枃浠?, "*.*")]"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.current_config = yaml.safe_load(f)
                self._load_config_to_ui()
                messagebox.showinfo("鎴愬姛", "閰嶇疆宸茶浇鍏 紒")
            except Exception as e:
                messagebox.showerror("閿欒 ", f"杞藉叆閰嶇疆澶辫触: {e}")

    def _save_config(self):
        """淇濆瓨閰嶇疆鏂囦欢"""
        config = self._get_config_from_ui()
        file_path = filedialog.asksaveasfilename(
            title="淇濆瓨閰嶇疆",
            defaultextension=".yaml",
            filetypes=[("YAML鏂囦欢", "*.yaml"), ("YAML鏂囦欢", "*.yml"), ("鎵 鏈夋枃浠?, "*.*")]"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                messagebox.showinfo("鎴愬姛", "閰嶇疆宸蹭繚瀛橈紒")
            except Exception as e:
                messagebox.showerror("閿欒 ", f"淇濆瓨閰嶇疆澶辫触: {e}")

    def _reset_default(self):
        """閲嶇疆涓洪粯璁 厤缃?"
        if messagebox.askyesno("纭  ", "纭 畾瑕侀噸缃 负榛樿 閰嶇疆鍚楋紵"):
            self.current_config = self._get_default_config()
            self._load_config_to_ui()

    def _apply_config(self):
        """搴旂敤閰嶇疆"""
        self.result = self._get_config_from_ui()
        self.dialog.destroy()

    def _get_preset_config(self, preset_name):
        """鑾峰彇棰勮 閰嶇疆"""
        presets = {
            "婵 杩涜秴鐭 嚎": {
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
            "瓒嬪娍杩借釜": {
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
            "骞宠 鍨?: {"
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
            "闇囪崱濂楀埄": {
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
            "绋冲仴闀跨嚎": {
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
            "娑堟伅椹卞姩": {
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
        return presets.get(preset_name, presets["骞宠 鍨?])"

    def _apply_preset(self, preset_name):
        """搴旂敤棰勮 閰嶇疆"""
        preset_config = self._get_preset_config(preset_name)

        # 鏇存柊閰嶇疆瀛楀吀
        for category, params in preset_config.items():
            if category in self.current_config:
                self.current_config[category].update(params)

        # 鏇存柊UI
        self._load_config_to_ui()
        print(f"宸插簲鐢  璁? {preset_name}")

    def _on_preset_changed(self, event):
        """棰勮 閫夋嫨鏀瑰彉鏃惰 鍙?"
        preset_name = self.preset_var.get()
        self._apply_preset(preset_name)

    def _cancel(self):
        """鍙栨秷"""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """鏄剧 瀵硅瘽妗嗗苟杩斿洖缁撴灉"""
        self.dialog.wait_window()
        return self.result


class KronosTradingGUI:
    def _center_window(self):
        """灏嗙獥鍙 眳涓 樉绀?"
        self.root.update_idletasks()

        # 鑾峰彇绐楀彛瀹藉害鍜岄珮搴?
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # 鑾峰彇灞忓箷瀹藉害鍜岄珮搴?
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 璁 畻绐楀彛浣嶇疆
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # 璁剧疆绐楀彛浣嶇疆
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # 瀵归綈瀹屾垚鍚庢樉绀虹獥鍙?
        self.root.deiconify()

    def __init__(self, root):
        self.root = root
        self.root.title("榛戠尗浜 槗绯荤粺v2.0")
        self.root.geometry("1650x980")
        self.root.configure(bg="#f0f0f0")

        # 绐楀彛灞呬腑鏄剧
        self._center_window()

        # 璁剧疆搴旂敤鍥炬爣
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
                # 鍚屾椂璁剧疆 iconphoto 浠  淇濅换鍔 爮鍥炬爣姝  鏄剧
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(icon_path)
                    photo = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, photo)
                except Exception:
                    pass
        except Exception as e:
            print(f"璁剧疆鍥炬爣澶辫触: {e}")

        self.trading_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.output_redirector = OutputRedirector(self.queue, self.set_progress)

        # 浜忔崯瑙 彂AI浼樺寲鐩稿叧鍙橀噺
        self.optimization_baseline_balance = 0.0  # 浼樺寲鍩哄噯璧勯噾
        self.loss_check_timer = None  # 浜忔崯妫 鏌 畾鏃跺櫒

        # 浜 槗闂撮殧璁 椂鍣 彉閲?
        self.interval_seconds = 120  # 榛樿 闂撮殧
        self.remaining_seconds = 0   # 鍓 綑绉掓暟
        self.interval_timer = None   # 璁 椂鍣 D
        self.is_interval_running = False  # 鏄 惁姝 湪鍊掕 鏃?

        # 鍒濆 鍖栨棩蹇楁枃浠?
        self._init_log_files()

        # Kronos妯 瀷缂撳瓨锛堢敤浜庡彲瑙嗗寲锛?
        self.kronos_model = None
        self.kronos_tokenizer = None
        self.kronos_predictor = None
        self.kronos_model_name = "custom:custom_model"  # 榛樿 妯 瀷

        # 澶氭櫤鑳戒綋閲忓寲浜 槗绯荤粺灞炴 ?
        self.fingpt_analyzer = None
        self.strategy_coordinator = None
        self.use_sentiment_analysis = True  # 榛樿 鍚 敤鑸嗘儏鍒嗘瀽
        self.sentiment_filter_enabled = True  # 榛樿 鍚 敤淇 彿杩囨护

        # BTC鏂伴椈鐖 櫕
        self.news_crawler = None
        self.news_list = []

        # GPU妫 娴嬪垵濮嬪寲
        self._nvml_initialized = False
        self._gpu_available = False
        self._gpu_handle = None

        # 灏濊瘯鍒濆 鍖朑PU妫 娴?
        try:
            print("鍒濆 鍖朑PU妫 娴?..")
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
                gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                print(f"GPU妫 娴嬫垚鍔? {gpu_name}")
            else:
                print("鏃燝PU璁惧 ")
            self._nvml_initialized = True
        except Exception as e:
            print(f"GPU鍒濆 鍖栧 璐? {e}")
            self._nvml_initialized = False

        self.create_styles()

        # 鍒濆 鍖栧疄鐩樼洃鎺 浉鍏冲彉閲忥紙鍦 reate_widgets涔嬪墠锛?
        self.is_live_monitoring = False
        self.live_monitor_thread = None
        self.trade_executor = None
        self.live_exchange_var = tk.StringVar(value="binance")
        self.live_symbol_var = tk.StringVar(value="BTCUSDT")
        self.account_balance_var = tk.StringVar(value="$0.00")
        self.position_info_var = tk.StringVar(value="鏆傛棤鎸佷粨")
        self.current_price_var = tk.StringVar(value="--")
        self.price_change_var = tk.StringVar(value="0.00%")
        self.monitor_status_var = tk.StringVar(value="鏈 惎鍔?)"

        # 鍒濆 鍖朅I绛栫暐涓 績鐩稿叧鍙橀噺
        self.fingpt_status_var = tk.StringVar(value="鏈 垵濮嬪寲")
        self.coordinator_status_var = tk.StringVar(value="鏈 垵濮嬪寲")

        # 浜 槗缁熻 鍙橀噺
        self.today_trades_var = tk.StringVar(value="0")
        self.today_profit_var = tk.StringVar(value="$0.00")
        self.week_trades_var = tk.StringVar(value="0")
        self.week_profit_var = tk.StringVar(value="$0.00")
        self.month_trades_var = tk.StringVar(value="0")
        self.month_profit_var = tk.StringVar(value="$0.00")

        # 绛栫暐纭  娆 暟鍙橀噺
        self.entry_confirm_count_var = tk.StringVar(value="2")
        self.reverse_confirm_count_var = tk.StringVar(value="2")
        self.require_consecutive_prediction_var = tk.StringVar(value="3")

        # 寮 浠撳悗璁 椂鍙傛暟
        self.post_entry_hours_var = tk.StringVar(value="2")
        self.take_profit_min_pct_var = tk.StringVar(value="0.6")

        # 鍒濆 鍖栬嚜鍔 寲浼樺寲绯荤粺鍙橀噺
        self.performance_monitor = None
        self.auto_optimization_pipeline = None
        self.parameter_integrator = None
        self.is_auto_optimization_enabled = False
        self.auto_optimization_status_var = tk.StringVar(value="绂佺敤")
        self.optimization_threshold_var = tk.StringVar(value="涓 瓑")
        self.optimization_frequency_var = tk.StringVar(value="姣?鍒嗛挓")
        self.optimization_judgment_mode_var = tk.StringVar(value="鍥哄畾闃堝 ?)"

        # 鏂伴椈鑷 姩鍒锋柊鐩稿叧
        self.news_auto_refresh_job = None

        # 鍒濆 鍖栧 鏅鸿兘浣撻噺鍖栦氦鏄撶郴缁?
        self.initialize_multi_agent_system()

        self.create_widgets()
        self.load_settings()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.process_queue)
        # 鍚 姩鏂伴椈鑷 姩鍒锋柊锛?0绉掑悗寮 濮嬶紝姣?鍒嗛挓鍒锋柊锛?
        self.root.after(30000, self._start_news_auto_refresh)
        # 鑷 姩鍒锋柊IP鍦板潃锛堢晫闈 姞杞藉悗500ms锛?
        self.root.after(500, self.refresh_ip_address)

        # 鍒涘缓淇 彿鏂囦欢锛岄 氱煡鍚 姩鍣 富绋嬪簭宸插噯澶囧
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            signal_file = os.path.join(base_dir, "_splash_close.txt")
            with open(signal_file, 'w') as f:
                f.write("ready")
        except Exception as e:
            print(f"鍒涘缓淇 彿鏂囦欢澶辫触: {e}")

    def _init_log_files(self):
        """鍒濆 鍖栨棩蹇楁枃浠?"
        import logging

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 浜 槗鏃 織
        trade_log = os.path.join(
            log_dir, f"trade_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.trade_logger = logging.getLogger("trade")
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.handlers = []
        self.trade_logger.addHandler(logging.FileHandler(trade_log, encoding="utf-8"))

        # 璁 粌鏃 織
        train_log = os.path.join(
            log_dir, f"train_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.train_logger = logging.getLogger("train")
        self.train_logger.setLevel(logging.INFO)
        self.train_logger.handlers = []
        self.train_logger.addHandler(logging.FileHandler(train_log, encoding="utf-8"))

        # 鍙  鍖栨棩蹇?
        visualize_log = os.path.join(
            log_dir, f"visualize_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.visualize_logger = logging.getLogger("visualize")
        self.visualize_logger.setLevel(logging.INFO)
        self.visualize_logger.handlers = []
        self.visualize_logger.addHandler(
            logging.FileHandler(visualize_log, encoding="utf-8")
        )

        # 鍥炴祴鏃 織
        backtest_log = os.path.join(
            log_dir, f"backtest_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.backtest_logger = logging.getLogger("backtest")
        self.backtest_logger.setLevel(logging.INFO)
        self.backtest_logger.handlers = []
        self.backtest_logger.addHandler(logging.FileHandler(backtest_log, encoding="utf-8"))

    def _load_balance_data(self):
        """鍔犺浇鍘嗗彶璧勯噾鏁版嵁"""
        try:
            if os.path.exists(self.balance_data_file):
                df = pd.read_csv(self.balance_data_file)
                self.balance_data = df.to_dict('records')
                print(f"宸插姞杞?{len(self.balance_data)} 鏉 巻鍙茶祫閲戞暟鎹?)"
            else:
                self.balance_data = []
        except Exception as e:
            print(f"鍔犺浇璧勯噾鏁版嵁澶辫触: {e}")
            self.balance_data = []

    def _save_balance_data(self):
        """淇濆瓨璧勯噾鏁版嵁鍒版枃浠?"
        try:
            if self.balance_data:
                df = pd.DataFrame(self.balance_data)
                df.to_csv(self.balance_data_file, index=False)
        except Exception as e:
            print(f"淇濆瓨璧勯噾鏁版嵁澶辫触: {e}")

    def _record_balance(self):
        """璁板綍褰撳墠璧勯噾鏁版嵁锛堟瘡鍒嗛挓璋冪敤锛?"
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 浣跨敤鍚堢害浠撲綅鎬昏祫閲戯紙鍖呭惈鎸佷粨鐩堜簭锛?
            initial_balance = self.initial_futures_balance

            current_balance = 0.0
            is_strategy_running = False

            try:
                if hasattr(self, 'strategy') and self.strategy:
                    # 浣跨敤鍚堢害浠撲綅鎬昏祫閲戯紙鍖呭惈鎸佷粨鐩堜簭锛?
                    balance = self.strategy.binance.get_total_balance()
                    if balance and balance > 0:
                        current_balance = balance
                        is_strategy_running = True
            except:
                pass

            # 鍙 湁鍦 瓥鐣 繍琛屾椂鎵嶈 褰曟暟鎹?
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
            print(f"璁板綍璧勯噾鏁版嵁澶辫触: {e}")

        self.balance_record_timer = self.root.after(60000, self._record_balance)

    def _start_balance_recording(self):
        """寮 濮嬭 褰曡祫閲戞暟鎹?"
        if self.balance_record_timer is None:
            self._record_balance()

    def _stop_balance_recording(self):
        """鍋滄 璁板綍璧勯噾鏁版嵁"""
        if self.balance_record_timer is not None:
            self.root.after_cancel(self.balance_record_timer)
            self.balance_record_timer = None

    def initialize_multi_agent_system(self):
        """鍒濆 鍖栧 鏅鸿兘浣撻噺鍖栦氦鏄撶郴缁?"
        print("鍒濆 鍖栧 鏅鸿兘浣撻噺鍖栦氦鏄撶郴缁?..")

        # 鍒濆 鍖朏inGPT鑸嗘儏鍒嗘瀽鍣?
        if FinGPTSentimentAnalyzer is not None and self.use_sentiment_analysis:
            try:
                print("  姝 湪鍒濆 鍖朏inGPT鑸嗘儏鍒嗘瀽鍣?..")
                self.fingpt_analyzer = FinGPTSentimentAnalyzer(
                    use_local_model=True,
                    use_qwen_preprocessing=True
                )
                print("  鉁?FinGPT鑸嗘儏鍒嗘瀽鍣 垵濮嬪寲瀹屾垚")
            except Exception as e:
                print(f"  鉁?FinGPT鑸嗘儏鍒嗘瀽鍣 垵濮嬪寲澶辫触: {e}")
                self.fingpt_analyzer = None
        else:
            print("  鈿?FinGPT妯 潡涓嶅彲鐢 垨宸茬 鐢?)"
            self.fingpt_analyzer = None

        # 鍒濆 鍖栫瓥鐣 崗璋冨櫒
        if StrategyCoordinator is not None:
            try:
                print("  姝 湪鍒濆 鍖栫瓥鐣 崗璋冨櫒...")
                self.strategy_coordinator = StrategyCoordinator(
                    kronos_model_name="custom:custom_model",
                    use_fingpt=(self.fingpt_analyzer is not None),
                    symbol="BTC"
                )
                print("  鉁?绛栫暐鍗忚皟鍣 垵濮嬪寲瀹屾垚")

                # 搴旂敤绛夊緟鐨勫弬鏁帮紙濡傛灉鏈夛級
                if hasattr(self, "_pending_coordinator_params") and self._pending_coordinator_params:
                    print("  搴旂敤涔嬪墠浼樺寲鐨勫崗璋冨櫒鍙傛暟...")
                    self.strategy_coordinator.update_config(self._pending_coordinator_params)
                    print("  鉁?鍗忚皟鍣 弬鏁板凡搴旂敤")
            except Exception as e:
                print(f"  鉁?绛栫暐鍗忚皟鍣 垵濮嬪寲澶辫触: {e}")
                self.strategy_coordinator = None
        else:
            print("  鈿?绛栫暐鍗忚皟鍣  鍧椾笉鍙 敤")
            self.strategy_coordinator = None

        print("澶氭櫤鑳戒綋閲忓寲浜 槗绯荤粺鍒濆 鍖栧畬鎴?)"

        # 鍒濆 鍖朆TC鏂伴椈鐖 櫕
        if BTCNewsCrawler is not None:
            try:
                print("  姝 湪鍒濆 鍖朆TC鏂伴椈鐖 櫕...")
                self.news_crawler = BTCNewsCrawler()
                print("  鉁?BTC鏂伴椈鐖 櫕鍒濆 鍖栧畬鎴?)"
            except Exception as e:
                print(f"  鉁?BTC鏂伴椈鐖 櫕鍒濆 鍖栧 璐? {e}")
                self.news_crawler = None

        # 鏇存柊GUI鐘舵 佹樉绀猴紙浣跨敤after纭 繚GUI宸插垱寤猴級
        self.root.after(100, self.update_multi_agent_status)

    def update_multi_agent_status(self):
        """鏇存柊澶氭櫤鑳戒綋绯荤粺鐘舵 佹樉绀?"
        # 鏇存柊FinGPT鐘舵 ?
        if self.fingpt_analyzer is not None:
            self.fingpt_status_var.set("杩愯 涓?)"
            self.fingpt_status_label.config(foreground="#27ae60")  # 缁胯壊
        else:
            self.fingpt_status_var.set("鏈 惎鐢?)"
            self.fingpt_status_label.config(foreground="#f39c12")  # 姗欒壊

        # 鏇存柊绛栫暐鍗忚皟鍣 姸鎬?
        if self.strategy_coordinator is not None:
            self.coordinator_status_var.set("杩愯 涓?)"
            self.coordinator_status_label.config(foreground="#27ae60")  # 缁胯壊
        else:
            self.coordinator_status_var.set("鏈 惎鐢?)"
            self.coordinator_status_label.config(foreground="#f39c12")  # 姗欒壊

    def refresh_model_list(self):
        """鍒锋柊妯 瀷鍒楄 """
        model_list = ["kronos-small", "kronos-mini", "kronos-base"]
        seen_models = set(model_list)

        # 鎵 弿璁 粌濂界殑妯 瀷鐩 綍
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
                        # 妫 鏌 槸鍚 湁璁 粌濂界殑妯 瀷
                        tokenizer_path = os.path.join(
                            exp_path, "tokenizer", "best_model"
                        )
                        basemodel_path = os.path.join(
                            exp_path, "basemodel", "best_model"
                        )
                        # 涔熸 鏌 祵濂楃洰褰曠粨鏋?
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
                print(f"鎵 弿璁 粌妯 瀷澶辫触: {e}")

        # 涔熷彲浠  鏌 敤鎴疯嚜瀹氫箟鐨勬 鍨嬬洰褰?
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
                print(f"鎵 弿鑷 畾涔夋 鍨嬪 璐? {e}")

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
                       font=('寰 蒋闆呴粦', 11, 'bold'),
                       padding=10,
                       foreground='white',
                       background='#3498db',
                       bordercolor='#2980b9',
                       focuscolor='none')
        style.map('Accent.TButton',
                 background=[('active', '#2980b9')],
                 relief=[('pressed', 'sunken')])

    def refresh_ip_address(self):
        """鍒锋柊鑾峰彇褰撳墠IP鍦板潃"""
        try:
            self.ip_address_var.set("鑾峰彇涓?..")
            import socket
            import requests

            # 鏂规硶1: 浣跨敤socket鑾峰彇鏈 湴IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "127.0.0.1"

            # 鏂规硶2: 浣跨敤API鑾峰彇鍏 綉IP锛堝彲閫夛級
            public_ip = "鑾峰彇涓?.."
            try:
                response = requests.get("https://api.ipify.org?format=json", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    public_ip = data.get("ip", "鏈 煡")
            except:
                public_ip = "鑾峰彇澶辫触"

            # 鏄剧 鏈 湴IP鍜屽叕缃慖P
            self.ip_address_var.set(f"鏈 湴:{local_ip} | 鍏 綉:{public_ip}")
        except Exception as e:
            self.ip_address_var.set(f"鑾峰彇澶辫触: {str(e)}")

    def create_widgets(self):
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_container, style="TFrame", width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_container, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 鍙充晶鍒嗕负涓婁笅涓 儴鍒嗭細涓婇潰鏄 郴缁熺洃鎺 紝涓嬮潰鏄 粓绔?
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
        """鍒涘缓绯荤粺鐩戞帶闈 澘"""
        # CPU
        cpu_frame = ttk.Frame(parent)
        cpu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(cpu_frame, text="CPU", font=("寰 蒋闆呴粦", 10, "bold")).pack()
        self.cpu_var = tk.StringVar(value="0%")
        self.cpu_label = ttk.Label(
            cpu_frame,
            textvariable=self.cpu_var,
            font=("寰 蒋闆呴粦", 16, "bold"),
            foreground="#007bff",
        )
        self.cpu_label.pack()
        self.cpu_progress = ttk.Progressbar(cpu_frame, mode="determinate", length=80)
        self.cpu_progress.pack(fill=tk.X, pady=(0, 5))

        self.cpu_temp_var = tk.StringVar(value="--掳C")
        self.cpu_temp_label = ttk.Label(
            cpu_frame, textvariable=self.cpu_temp_var, font=("寰 蒋闆呴粦", 9)
        )
        self.cpu_temp_label.pack()

        # 鍐呭瓨
        mem_frame = ttk.Frame(parent)
        mem_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(mem_frame, text="鍐呭瓨", font=("寰 蒋闆呴粦", 10, "bold")).pack()
        self.mem_var = tk.StringVar(value="0%")
        self.mem_label = ttk.Label(
            mem_frame,
            textvariable=self.mem_var,
            font=("寰 蒋闆呴粦", 16, "bold"),
            foreground="#ff9500",
        )
        self.mem_label.pack()
        self.mem_progress = ttk.Progressbar(mem_frame, mode="determinate", length=80)
        self.mem_progress.pack(fill=tk.X, pady=(0, 5))

        self.mem_detail_var = tk.StringVar(value="")
        self.mem_detail_label = ttk.Label(
            mem_frame, textvariable=self.mem_detail_var, font=("寰 蒋闆呴粦", 9)
        )
        self.mem_detail_label.pack()

        # GPU
        gpu_frame = ttk.Frame(parent)
        gpu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(gpu_frame, text="GPU", font=("寰 蒋闆呴粦", 10, "bold")).pack()
        self.gpu_var = tk.StringVar(value="妫 娴嬩腑...")
        self.gpu_label = ttk.Label(
            gpu_frame,
            textvariable=self.gpu_var,
            font=("寰 蒋闆呴粦", 16, "bold"),
            foreground="#28a745",
        )
        self.gpu_label.pack()
        self.gpu_progress = ttk.Progressbar(gpu_frame, mode="determinate", length=80)
        self.gpu_progress.pack(fill=tk.X, pady=(0, 5))

        self.gpu_temp_var = tk.StringVar(value="--掳C")
        self.gpu_temp_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_temp_var, font=("寰 蒋闆呴粦", 9)
        )
        self.gpu_temp_label.pack()

        self.gpu_3d_var = tk.StringVar(value="3D: --%")
        self.gpu_3d_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_3d_var, font=("寰 蒋闆呴粦", 9)
        )
        self.gpu_3d_label.pack()

        self.gpu_detail_var = tk.StringVar(value="")
        self.gpu_detail_label = ttk.Label(
            gpu_frame, textvariable=self.gpu_detail_var, font=("寰 蒋闆呴粦", 8)
        )
        self.gpu_detail_label.pack()

        # 纾佺洏
        disk_frame = ttk.Frame(parent)
        disk_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(disk_frame, text="纾佺洏", font=("寰 蒋闆呴粦", 10, "bold")).pack()
        self.disk_var = tk.StringVar(value="0%")
        self.disk_label = ttk.Label(
            disk_frame,
            textvariable=self.disk_var,
            font=("寰 蒋闆呴粦", 16, "bold"),
            foreground="#6c757d",
        )
        self.disk_label.pack()
        self.disk_progress = ttk.Progressbar(disk_frame, mode="determinate", length=80)
        self.disk_progress.pack(fill=tk.X, pady=(0, 5))

        self.disk_detail_var = tk.StringVar(value="")
        self.disk_detail_label = ttk.Label(
            disk_frame, textvariable=self.disk_detail_var, font=("寰 蒋闆呴粦", 9)
        )
        self.disk_detail_label.pack()

        # 鍚 姩瀹氭椂鏇存柊
        self._update_system_info()
        self._system_timer = self.root.after(2000, self._update_system_info)

    def _update_system_info(self):
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.3)
            self.cpu_var.set(f"{cpu_percent:.0f}%")
            self.cpu_progress["value"] = cpu_percent

            # CPU娓 害
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries and hasattr(entries[0], "current"):
                            self.cpu_temp_var.set(f"{entries[0].current:.0f}掳C")
                            break
            except:
                self.cpu_temp_var.set("--掳C")

            # 鍐呭瓨
            mem = psutil.virtual_memory()
            self.mem_var.set(f"{mem.percent:.0f}%")
            self.mem_progress["value"] = mem.percent
            self.mem_detail_var.set(f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB")

            # GPU - 浣跨敤pynvml鐩存帴妫 娴?
            if not self._nvml_initialized:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        self._gpu_available = True
                        self._nvml_initialized = True
                        print("GPU閲嶆柊鍒濆 鍖栨垚鍔?)"
                    else:
                        self._gpu_available = False
                        self._gpu_handle = None
                except Exception as e:
                    print(f"GPU鍒濆 鍖栧 璐? {e}")
                    self._nvml_initialized = False
                    self._gpu_available = False
                    self._gpu_handle = None

            if self._gpu_available and self._gpu_handle:
                try:
                    # GPU娓 害
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        self.gpu_temp_var.set(f"{temp}掳C")
                    except Exception as e:
                        print(f"鑾峰彇娓 害澶辫触: {e}")
                        self.gpu_temp_var.set("--掳C")

                    # GPU 3D浣跨敤鐜?
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                        self.gpu_3d_var.set(f"3D: {util.gpu}%")
                    except Exception as e:
                        print(f"鑾峰彇浣跨敤鐜囧 璐? {e}")
                        self.gpu_3d_var.set("3D: --%")

                    # GPU鏄惧瓨
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
                        print(f"鑾峰彇鏄惧瓨澶辫触: {e}")
                        # 濡傛灉鑾峰彇鏄惧瓨澶辫触锛屾樉绀?D浣跨敤鐜?
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(
                                self._gpu_handle
                            )
                            self.gpu_var.set(f"{util.gpu}%")
                            self.gpu_progress["value"] = util.gpu
                        except Exception as e2:
                            print(f"鑾峰彇3D浣跨敤鐜囧 璐? {e2}")
                            self.gpu_var.set("--%")
                        self.gpu_detail_var.set("")
                except Exception as e:
                    print(f"GPU妫 娴嬮敊璇? {e}")
                    self.gpu_temp_var.set("--掳C")
                    self.gpu_3d_var.set("3D: --%")
                    self.gpu_var.set("鏃燝PU")
                    self.gpu_detail_var.set("")
                    # 閲嶇疆鐘舵 侊紝涓嬫 閲嶆柊鍒濆 鍖?
                    self._nvml_initialized = False
                    self._gpu_available = False
                    self._gpu_handle = None
            else:
                self.gpu_temp_var.set("--掳C")
                self.gpu_3d_var.set("3D: --%")
                self.gpu_var.set("鏃燝PU")
                self.gpu_detail_var.set("")
                # 灏濊瘯閲嶆柊鍒濆 鍖?
                if not self._nvml_initialized:
                    try:
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        if device_count > 0:
                            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            self._gpu_available = True
                            self._nvml_initialized = True
                            print("GPU閲嶆柊鍒濆 鍖栨垚鍔?)"
                        else:
                            self._gpu_available = False
                            self._gpu_handle = None
                    except Exception as e:
                        print(f"GPU鍒濆 鍖栧 璐? {e}")
                        self._nvml_initialized = False
                        self._gpu_available = False
                        self._gpu_handle = None

            # 纾佺洏
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
            print(f"绯荤粺鐩戞帶鏇存柊閿欒 : {e}")

        self._system_timer = self.root.after(2000, self._update_system_info)

    def create_api_section(self, parent):
        api_frame = ttk.LabelFrame(parent, text="API 閰嶇疆", style="TFrame", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))

        api_grid = ttk.Frame(api_frame)
        api_grid.pack(fill=tk.X, pady=(0, 5))

        api_col1 = ttk.Frame(api_grid)
        api_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(api_col1, text="甯佸畨 API Key:").pack(anchor=tk.W)
        self.api_key_entry = ttk.Entry(api_col1, width=20)
        self.api_key_entry.pack(fill=tk.X, pady=(0, 0))

        api_col2 = ttk.Frame(api_grid)
        api_col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(api_col2, text="甯佸畨 Secret Key:").pack(anchor=tk.W)
        self.secret_key_entry = ttk.Entry(api_col2, width=20)
        self.secret_key_entry.pack(fill=tk.X, pady=(0, 0))

        self.show_api_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            api_frame,
            text="鏄剧 API瀵嗛挜",
            variable=self.show_api_var,
            command=self.toggle_api_visibility,
        ).pack(anchor=tk.W)

    def create_config_section(self, parent):
        config_frame = ttk.LabelFrame(
            parent, text="浜 槗閰嶇疆", style="TFrame", padding=10
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # 浜 槗瀵?- 绗 竴琛?
        pair_frame = ttk.Frame(config_frame)
        pair_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pair_frame, text="浜 槗瀵?", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Label(pair_frame, text="BTCUSDT", font=("寰 蒋闆呴粦", 10, "bold")).pack(side=tk.LEFT)

        # IP鍦板潃 - 绗 簩琛?
        ip_frame = ttk.Frame(config_frame)
        ip_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ip_frame, text="IP鍦板潃:", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.ip_address_var = tk.StringVar(value="鑾峰彇涓?..")
        self.ip_address_label = ttk.Label(ip_frame, textvariable=self.ip_address_var, font=("寰 蒋闆呴粦", 10))
        self.ip_address_label.pack(side=tk.LEFT)
        # 娣诲姞鑾峰彇IP鍦板潃鎸夐挳
        ttk.Button(ip_frame, text="鍒锋柊", command=self.refresh_ip_address).pack(side=tk.LEFT, padx=(10, 0))

        # 浜 槗绛栫暐鍜孠ronos妯 瀷 - 涓 琛屼袱鍒?
        strategy_model_grid = ttk.Frame(config_frame)
        strategy_model_grid.pack(fill=tk.X, pady=(0, 5))

        strategy_col = ttk.Frame(strategy_model_grid)
        strategy_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(strategy_col, text="浜 槗绛栫暐:").pack(anchor=tk.W)
        self.strategy_var = tk.StringVar(value="鑷 姩绛栫暐")
        strategy_combo = ttk.Combobox(
            strategy_col, textvariable=self.strategy_var, state="readonly", width=15
        )
        strategy_combo["values"] = (
            "瓒嬪娍鐖嗗彂",
            "闇囪崱濂楀埄",
            "娑堟伅绐佺牬",
            "鑷 姩绛栫暐",
            "鏃堕棿绛栫暐",
        )
        strategy_combo.pack(fill=tk.X, pady=(0, 0))
        strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_changed)

        model_col = ttk.Frame(strategy_model_grid)
        model_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(model_col, text="Kronos妯 瀷:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="custom:custom_model")
        self.model_combo = ttk.Combobox(
            model_col, textvariable=self.model_var, state="readonly", width=15
        )
        self.refresh_model_list()
        self.model_combo.pack(fill=tk.X, pady=(0, 0))

        # 妯 瀷鍒囨崲鏃堕噸缃甂ronos predictor
        def on_model_change(*args):
            self.kronos_predictor = None
            self.kronos_model = None
            self.kronos_tokenizer = None
            self.kronos_model_name = self.model_var.get()

        self.model_var.trace_add("write", on_model_change)

        # 鍒锋柊妯 瀷鎸夐挳
        refresh_model_frame = ttk.Frame(config_frame)
        refresh_model_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            refresh_model_frame, text="鍒锋柊妯 瀷鍒楄 ", command=self.refresh_model_list
        ).pack(side=tk.RIGHT)

        # 鍙傛暟 - 5琛屼袱鍒楋紝鍏?0涓 弬鏁?
        params_grid = ttk.Frame(config_frame)
        params_grid.pack(fill=tk.X, pady=(0, 0))

        # 琛?: 鍒嗘瀽鍛 湡 + 鏉犳潌鍊嶆暟
        row1 = ttk.Frame(params_grid)
        row1.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row1)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="鍒嗘瀽鍛 湡:").pack(anchor=tk.W)
        self.timeframe_var = tk.StringVar(value="5m")
        timeframe_combo = ttk.Combobox(
            col1, textvariable=self.timeframe_var, state="readonly", width=15
        )
        timeframe_combo["values"] = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        timeframe_combo.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row1)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="鏉犳潌鍊嶆暟:").pack(anchor=tk.W)
        self.leverage_var = tk.StringVar(value="10")
        leverage_combo = ttk.Combobox(
            col2, textvariable=self.leverage_var, state="readonly", width=15
        )
        leverage_combo["values"] = ("1", "2", "3", "5", "10", "20", "25", "50", "75", "100")
        leverage_combo.pack(fill=tk.X, pady=(0, 0))

        # 琛?: 鏈 灏忎粨浣?+ 浜 槗闂撮殧
        row2 = ttk.Frame(params_grid)
        row2.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row2)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="鏈 灏忎粨浣?(USDT):").pack(anchor=tk.W)
        self.min_position_var = tk.StringVar(value="100")
        min_pos_entry = ttk.Entry(
            col1, textvariable=self.min_position_var, width=18
        )
        min_pos_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row2)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="浜 槗闂撮殧 (绉?:").pack(anchor=tk.W)
        self.interval_var = tk.StringVar(value="120")
        interval_combo = ttk.Combobox(
            col2, textvariable=self.interval_var, state="readonly", width=15
        )
        interval_combo["values"] = ("30", "60", "120", "180", "300", "600")
        interval_combo.pack(fill=tk.X, pady=(0, 0))

        # 琛?: 瓒嬪娍闃堝 ?+ AI鏈 灏忚秼鍔垮己搴?
        row3 = ttk.Frame(params_grid)
        row3.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row3)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="瓒嬪娍闃堝 ?").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="0.008")
        threshold_entry = ttk.Entry(
            col1, textvariable=self.threshold_var, width=18
        )
        threshold_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row3)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="AI鏈 灏忚秼鍔垮己搴?").pack(anchor=tk.W)
        self.ai_min_trend_var = tk.StringVar(value="0.010")
        ai_min_trend_entry = ttk.Entry(
            col2, textvariable=self.ai_min_trend_var, width=18
        )
        ai_min_trend_entry.pack(fill=tk.X, pady=(0, 0))

        # 琛?: AI鏈 灏忛 娴嬪亸绂诲害 + 鏈 澶 祫閲戣垂鐜?
        row4 = ttk.Frame(params_grid)
        row4.pack(fill=tk.X, pady=(0, 4))
        col1 = ttk.Frame(row4)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="AI鏈 灏忛 娴嬪亸绂诲害:").pack(anchor=tk.W)
        self.ai_min_deviation_var = tk.StringVar(value="0.008")
        ai_min_deviation_entry = ttk.Entry(
            col1, textvariable=self.ai_min_deviation_var, width=18
        )
        ai_min_deviation_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row4)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="鏈 澶 祫閲戣垂鐜?(%):").pack(anchor=tk.W)
        self.max_funding_var = tk.StringVar(value="1.0")
        max_funding_entry = ttk.Entry(
            col2, textvariable=self.max_funding_var, width=18
        )
        max_funding_entry.pack(fill=tk.X, pady=(0, 0))

        # 琛?: 鏈 灏忚祫閲戣垂鐜?+ 浜忔崯瑙 彂浼樺寲
        row5 = ttk.Frame(params_grid)
        row5.pack(fill=tk.X, pady=(0, 0))
        col1 = ttk.Frame(row5)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(col1, text="鏈 灏忚祫閲戣垂鐜?(%):").pack(anchor=tk.W)
        self.min_funding_var = tk.StringVar(value="-1.0")
        min_funding_entry = ttk.Entry(
            col1, textvariable=self.min_funding_var, width=18
        )
        min_funding_entry.pack(fill=tk.X, pady=(0, 0))

        col2 = ttk.Frame(row5)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(col2, text="浜忔崯瑙 彂浼樺寲 (%):").pack(anchor=tk.W)
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
            button_frame, text="鍚 姩浜 槗", command=self.start_trading
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        self.stop_button = ttk.Button(
            button_frame, text="鍋滄 浜 槗", command=self.stop_trading, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        # 璧勯噾淇 伅鏄剧
        fund_frame = ttk.LabelFrame(control_frame, text="鍚堢害璧勯噾", padding=8)
        fund_frame.pack(fill=tk.X, pady=(10, 0))

        # 璁剧疆瀛椾綋
        label_font = ("寰 蒋闆呴粦", 10)
        value_font = ("寰 蒋闆呴粦", 11, "bold")

        # 鍒濆 璧勯噾 (绗?琛岀 1鍒?
        ttk.Label(fund_frame, text="鍒濆 璧勯噾:", font=label_font).grid(
            row=0, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.initial_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="blue"
        )
        self.initial_balance_label.grid(row=0, column=1, sticky=tk.E, pady=4, padx=5)

        # 褰撳墠璧勯噾 (绗?琛岀 2鍒?
        ttk.Label(fund_frame, text="褰撳墠璧勯噾:", font=label_font).grid(
            row=0, column=2, sticky=tk.W, pady=4, padx=5
        )
        self.current_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="green"
        )
        self.current_balance_label.grid(row=0, column=3, sticky=tk.E, pady=4, padx=5)

        # 鐩堜簭閲戦  (绗?琛岀 1鍒?
        ttk.Label(fund_frame, text="鐩堜簭閲戦 :", font=label_font).grid(
            row=1, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="red"
        )
        self.pnl_label.grid(row=1, column=1, sticky=tk.E, pady=4, padx=5)

        # 鐩堜簭姣斾緥 (绗?琛岀 2鍒?
        ttk.Label(fund_frame, text="鐩堜簭姣斾緥:", font=label_font).grid(
            row=1, column=2, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_pct_label = ttk.Label(
            fund_frame, text="0.00%", font=value_font, foreground="red"
        )
        self.pnl_pct_label.grid(row=1, column=3, sticky=tk.E, pady=4, padx=5)

        # 鍒濆 鍖栬祫閲戝彉閲?
        self.initial_futures_balance = 0.0

        # 璧勯噾鏁版嵁璁板綍
        self.balance_data_file = os.path.join(os.path.dirname(__file__), "balance_data.csv")
        self.balance_data = []
        self.balance_record_timer = None
        self._load_balance_data()

        # 绋嬪簭鍚 姩鍚庣珛鍗冲紑濮嬭 褰曡祫閲戞暟鎹?
        self.root.after(1000, self._start_balance_recording)

        # Kronos棰勬祴鍙  鍖栧浘琛?
        prediction_chart_frame = ttk.LabelFrame(control_frame, text="馃搳 Kronos棰勬祴璧板娍", padding=8)
        prediction_chart_frame.pack(fill=tk.X, pady=(10, 0))

        # 瀵煎叆matplotlib
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # 瀛樺偍Kronos鍒嗘瀽缁撴灉鐨勫彉閲?
            self.kronos_analysis_data = {
                'trend_direction': None,
                'trend_strength': 0,
                'pred_change': 0,
                'threshold': 0.008,
                'timestamp': None
            }

            # 鍒涘缓鍥捐
            self.prediction_fig = plt.Figure(figsize=(5.5, 3.0), dpi=100)
            self.prediction_ax = self.prediction_fig.add_subplot(111)

            # 璁剧疆涓 枃瀛椾綋
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 鍒濆 缁樺埗绌虹櫧鍥捐
            self._draw_prediction_chart()

            # 宓屽叆鍒癟kinter
            self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, master=prediction_chart_frame)
            self.prediction_canvas.draw()
            self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except ImportError as e:
            ttk.Label(prediction_chart_frame, text=f"鍥捐 搴撴湭瀹夎 : {e}", foreground="red").pack()
            self.prediction_fig = None
            self.prediction_ax = None
            self.prediction_canvas = None
            self.kronos_analysis_data = None

    def create_terminal_section(self, parent):
        # 鍒涘缓绗旇 鏈 帶浠讹紙鏍囩 椤碉級
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 浜 槗鏃 織鏍囩 椤?
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="浜 槗鏃 織")
        self._create_log_tab(log_frame)

        # AI瀹炵洏绛栫暐鏍囩 椤碉紙鍚堝苟AI绛栫暐涓 績 + 瀹炵洏鐩戞帶锛?
        ai_trading_frame = ttk.Frame(notebook)
        notebook.add(ai_trading_frame, text="AI瀹炵洏绛栫暐")
        self._create_ai_trading_tab(ai_trading_frame)

        # 绛栫暐鍥炴祴鏍囩 椤?
        backtest_frame = ttk.Frame(notebook)
        notebook.add(backtest_frame, text="绛栫暐鍥炴祴")
        self._create_main_backtest_tab(backtest_frame)

        # BTC鏂伴椈鏍囩 椤?
        news_frame = ttk.Frame(notebook)
        notebook.add(news_frame, text="BTC鏂伴椈")
        self._create_news_tab(news_frame)

        # 鍙  鍖栭 娴嬫爣绛鹃
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="鍙  鍖栭 娴?)"
        self._create_visualization_tab(viz_frame)

        # 璧勯噾鏇茬嚎鏍囩 椤?
        balance_chart_frame = ttk.Frame(notebook)
        notebook.add(balance_chart_frame, text="璧勯噾鏇茬嚎")
        self._create_balance_chart_tab(balance_chart_frame)

        # 璁 粌Kronos妯 瀷鏍囩 椤?
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="璁 粌Kronos妯 瀷")
        self._create_training_tab(train_frame)

        # 璁 粌鏂囦欢绠 悊鏍囩 椤?
        training_manager_frame = ttk.Frame(notebook)
        notebook.add(training_manager_frame, text="璁 粌鏂囦欢绠 悊")
        self._create_training_manager_tab(training_manager_frame)

        # 甯 姪鏍囩 椤?
        help_frame = ttk.Frame(notebook)
        notebook.add(help_frame, text="甯 姪")
        self._create_help_tab(help_frame)

    def _create_log_tab(self, parent):
        # 浜 槗闂撮殧杩涘害鏉  鏋?
        interval_frame = ttk.Frame(parent)
        interval_frame.pack(fill=tk.X, pady=(0, 10))

        # 浜 槗闂撮殧鏍囩
        self.interval_label = ttk.Label(
            interval_frame,
            text="浜 槗闂撮殧: 绌洪棽",
            font=("寰 蒋闆呴粦", 10),
            style="TLabel"
        )
        self.interval_label.pack(anchor=tk.W, pady=(0, 3))

        # 杩涘害鏉?- 鏄剧 浜 槗闂撮殧鍊掕 鏃?
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

        # 鍓 綑鏃堕棿鏍囩
        self.remaining_time_var = tk.StringVar(value="-- 绉?)"
        self.remaining_time_label = ttk.Label(
            interval_frame,
            textvariable=self.remaining_time_var,
            font=("寰 蒋闆呴粦", 9),
            foreground="#7f8c8d"
        )
        self.remaining_time_label.pack(anchor=tk.E)

        # 鐘舵 佹爣绛?
        self.status_label = ttk.Label(parent, text="灏辩华", style="TLabel")
        self.status_label.pack(anchor=tk.W, pady=(0, 5))

        # 缁堢 杈撳嚭
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
        # 鍙  鍖栨帶鍒舵寜閽  鏋?
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 鐢熸垚鍙  鍖栨寜閽?
        self.viz_button = ttk.Button(
            control_frame,
            text="鐢熸垚Kronos棰勬祴鍥捐 ",
            command=self.generate_visualization,
        )
        self.viz_button.pack(side=tk.LEFT, padx=5)

        # 妯 紡閫夋嫨
        ttk.Label(control_frame, text="妯 紡:").pack(side=tk.LEFT, padx=(20, 5))
        self.viz_mode_var = tk.StringVar(value="棰勬祴鏈 潵")
        mode_combo = ttk.Combobox(
            control_frame, textvariable=self.viz_mode_var, state="readonly", width=12
        )
        mode_combo["values"] = ("棰勬祴鏈 潵", "鍥炴祴杩囧幓")
        mode_combo.pack(side=tk.LEFT, padx=5)

        # 閰嶇疆閫夐
        ttk.Label(control_frame, text="棰勬祴闀垮害:").pack(side=tk.LEFT, padx=(20, 5))
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

        ttk.Label(control_frame, text="鍥炵湅鍛 湡:").pack(side=tk.LEFT, padx=(20, 5))
        self.lookback_var = tk.StringVar(value="512")
        lookback_combo = ttk.Combobox(
            control_frame, textvariable=self.lookback_var, state="readonly", width=10
        )
        lookback_combo["values"] = ("50", "100", "200", "300", "400", "512", "600")
        lookback_combo.pack(side=tk.LEFT, padx=5)

        # 鍥捐 鐢诲竷妗嗘灦
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 鍒涘缓matplotlib鍥惧舰
        self.viz_figure = Figure(figsize=(8, 6), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, master=canvas_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 鐘舵 佹爣绛?
        self.viz_status_label = ttk.Label(
            parent, text="灏辩华锛岀偣鍑绘寜閽 敓鎴愬彲瑙嗗寲", style="TLabel"
        )
        self.viz_status_label.pack(anchor=tk.W, pady=(5, 0))

    def generate_visualization(self):
        """鐢熸垚Kronos棰勬祴鍙  鍖栧浘琛?"
        try:
            # 绂佺敤鎸夐挳闃叉 閲嶅 鐐瑰嚮
            self.viz_button.config(state=tk.DISABLED)
            self.viz_status_label.config(text="姝 湪鐢熸垚鍙  鍖?..")
            self.viz_canvas.get_tk_widget().update()

            # 娣诲姞DatetimeIndex鍏煎 鎬 紙Kronos妯 瀷闇 瑕侊級
            pd.DatetimeIndex.dt = property(lambda self: self)

            # 鑾峰彇閰嶇疆鍙傛暟
            lookback = int(self.lookback_var.get())
            pred_len = int(self.pred_len_var.get())
            viz_mode = self.viz_mode_var.get()

            # 鏇存柊鐘舵 ?
            if viz_mode == "棰勬祴鏈 潵":
                self.viz_status_label.config(
                    text=f"[棰勬祴鏈 潵] 姝 湪鑾峰彇甯傚満鏁版嵁... (鍥炵湅{lookback}鍛 湡)"
                )
                data_len = lookback
            else:
                self.viz_status_label.config(
                    text=f"[鍥炴祴杩囧幓] 姝 湪鑾峰彇甯傚満鏁版嵁... (鍥炵湅{lookback}+{pred_len}鍛 湡)"
                )
                data_len = lookback + pred_len

            # 鑾峰彇K绾挎暟鎹?
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()

            # 浣跨敤binance_api鑾峰彇鏁版嵁
            from binance_api import BinanceAPI

            binance = BinanceAPI()
            klines_df = binance.get_recent_klines(symbol, timeframe, data_len)

            if klines_df is None or len(klines_df) < lookback:
                self.viz_status_label.config(text="鑾峰彇鏁版嵁澶辫触鎴栨暟鎹 笉瓒?)"
                self.viz_button.config(state=tk.NORMAL)
                return

            # get_recent_klines杩斿洖鐨勬槸宸茬粡澶勭悊濂界殑DataFrame
            # 妫 鏌 垪鍚?
            self.log(f"K绾挎暟鎹 垪: {list(klines_df.columns)}")
            self.log(f"K绾挎暟鎹 舰鐘? {klines_df.shape}")

            # 閲嶅懡鍚嶅垪浠 尮閰岾ronos鏈熸湜鐨勬牸寮?
            # get_klines杩斿洖鐨勫垪鏄? ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            if "timestamps" in klines_df.columns:
                df = klines_df.rename(columns={"timestamps": "timestamp"})
            else:
                df = klines_df.copy()

            # 纭 繚鏈塼imestamp鍒?
            if "timestamp" not in df.columns:
                self.viz_status_label.config(text="K绾挎暟鎹 己灏憈imestamp鍒?)"
                self.viz_button.config(state=tk.NORMAL)
                return

            # 鍙 繚鐣欓渶瑕佺殑鍒?
            needed_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in needed_cols if col not in df.columns]
            if missing_cols:
                self.log(f"缂哄皯鍒? {missing_cols}")
                self.viz_status_label.config(text=f"K绾挎暟鎹 己灏戝垪: {missing_cols}")
                self.viz_button.config(state=tk.NORMAL)
                return

            df = df[needed_cols].copy()

            # 纭 繚timestamp鏄痙atetime绫诲瀷
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                self.log("杞 崲timestamp鍒椾负datetime绫诲瀷")
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            self.log(f"澶勭悊鍚庣殑DataFrame褰 姸: {df.shape}")
            self.log(f"鏃堕棿鑼冨洿: {df['timestamp'].min()} 鍒?{df['timestamp'].max()}")

            # 鏇存柊鐘舵 ?
            self.viz_status_label.config(text="姝 湪鍔犺浇Kronos妯 瀷...")
            self.viz_canvas.get_tk_widget().update()

            # 鍔犺浇Kronos妯 瀷锛堜娇鐢 紦瀛橈級
            # 鍦 yInstaller鐜  涓  纭  缃 矾寰?
            if getattr(sys, 'frozen', False):
                # 濡傛灉鏄痚xe锛孠ronos鐩 綍鍦 ys._MEIPASS涓?
                kronos_path = os.path.join(sys._MEIPASS, "Kronos")
            else:
                kronos_path = os.path.join(os.path.dirname(__file__), "Kronos")
            sys.path.append(kronos_path)
            from Kronos.model import Kronos, KronosTokenizer, KronosPredictor

            if self.kronos_tokenizer is None or self.kronos_model is None:
                self.viz_status_label.config(text="姝 湪鍔犺浇Kronos妯 瀷...")
                self.viz_canvas.get_tk_widget().update()

                # 鑾峰彇妯 瀷鐩 綍璺 緞
                if getattr(sys, 'frozen', False):
                    base_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))

                # 妫 鏌?_internal/models 鐩 綍锛堟墦鍖呯幆澧冿級锛屽 鏋滀笉瀛樺湪鍐嶆 鏌?models 鐩 綍
                tokenizer_path = os.path.join(base_dir, "_internal", "models", "kronos-tokenizer-base")
                if not os.path.exists(tokenizer_path):
                    tokenizer_path = os.path.join(base_dir, "models", "kronos-tokenizer-base")

                model_path = os.path.join(base_dir, "_internal", "models", "kronos-small")
                if not os.path.exists(model_path):
                    model_path = os.path.join(base_dir, "models", "kronos-small")

                self.log(f"Tokenizer璺 緞: {tokenizer_path}")
                self.log(f"Model璺 緞: {model_path}")

                if os.path.exists(tokenizer_path) and os.path.exists(model_path):
                    self.viz_status_label.config(text="浠庢湰鍦板姞杞終ronos妯 瀷...")
                    self.log("浠庢湰鍦扮洰褰曞姞杞終ronos妯 瀷")
                else:
                    self.viz_status_label.config(text="妯 瀷鏂囦欢缂哄 ")
                    self.log("閿欒 : 妯 瀷鏂囦欢涓嶅瓨鍦 紝璇风 淇漨odels鐩 綍瀹屾暣")
                    messagebox.showerror(
                        "妯 瀷缂哄 ",
                        "鏈 壘鍒癒ronos妯 瀷鏂囦欢锛乗n\n"
                        "璇风 淇漨odels鐩 綍鍖呭惈浠 笅鍐呭 锛歕n"
                        "- models/kronos-small/\n"
                        "- models/kronos-tokenizer-base/"
                    )
                    return

                try:
                    import json
                    from safetensors.torch import load_file

                    self.log("鎵嬪姩鍔犺浇Kronos Tokenizer...")
                    tokenizer_config_path = os.path.join(tokenizer_path, "config.json")
                    tokenizer_weights_path = os.path.join(tokenizer_path, "model.safetensors")

                    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                        tokenizer_config = json.load(f)

                    self.kronos_tokenizer = KronosTokenizer(**tokenizer_config)
                    tokenizer_state_dict = load_file(tokenizer_weights_path)
                    self.kronos_tokenizer.load_state_dict(tokenizer_state_dict)
                    self.kronos_tokenizer.eval()
                    self.log("Tokenizer鍔犺浇瀹屾垚")

                    self.log("鎵嬪姩鍔犺浇Kronos Model...")
                    model_config_path = os.path.join(model_path, "config.json")
                    model_weights_path = os.path.join(model_path, "model.safetensors")

                    with open(model_config_path, "r", encoding="utf-8") as f:
                        model_config = json.load(f)

                    self.kronos_model = Kronos(**model_config)
                    model_state_dict = load_file(model_weights_path)
                    self.kronos_model.load_state_dict(model_state_dict)
                    self.kronos_model.eval()
                    self.log("Model鍔犺浇瀹屾垚")

                except Exception as e:
                    self.log(f"鎵嬪姩鍔犺浇澶辫触: {e}")
                    self.log("灏濊瘯浣跨敤from_pretrained鏂规硶(浠呮湰鍦?...")
                    try:
                        self.kronos_tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                        self.kronos_model = Kronos.from_pretrained(model_path, local_files_only=True)
                    except Exception as e2:
                        self.log(f"from_pretrained鏈 湴鍔犺浇涔熷 璐? {e2}")
                        self.log("璇锋 鏌  鍨嬫枃浠舵槸鍚 畬鏁达紝鎴栧皾璇曢噸鏂颁笅杞芥 鍨?)"
                        raise

                self.log("Kronos妯 瀷鍔犺浇瀹屾垚")

            # 瀹氫箟Kronos鐨勭壒寰佸垪琛?
            # 瀹樻柟妯 瀷(kronos-*)鐢?涓 熀纭 鐗瑰緛锛岃嚜瀹氫箟妯 瀷鐢?7涓 壒寰?
            OFFICIAL_MODEL_FEATURES = [
                "open", "high", "low", "close", "vol", "amt"
            ]
            CUSTOM_MODEL_FEATURES = [
                "open", "high", "low", "close", "vol", "amt",  # 鍩虹 鏁版嵁
                "MA5", "MA10", "MA20",                           # 绉诲姩骞冲潎绾?
                "BIAS20",                                          # 涔栫 鐜?
                "ATR14", "AMPLITUDE",                             # 娉 姩鎬 寚鏍?
                "AMOUNT_MA5", "AMOUNT_MA10", "VOL_RATIO",         # 鎴愪氦閲忔寚鏍?
                "RSI14", "RSI7",                                  # 鍔 噺鎸囨爣
                "MACD", "MACD_HIST",                              # MACD鎸囨爣
                "PRICE_SLOPE5", "PRICE_SLOPE10",                  # 瓒嬪娍鎸囨爣
                "HIGH5", "LOW5", "HIGH10", "LOW10",              # 鏋佸 兼寚鏍?
                "VOL_BREAKOUT", "VOL_SHRINK"                       # 鎴愪氦閲忕獊鐮?
            ]

            # 鏍规嵁妯 瀷绫诲瀷閫夋嫨鐗瑰緛鍒楄
            # 鐩存帴浠巑odel_var鑾峰彇褰撳墠閫夋嫨鐨勬 鍨嬶紝纭 繚绗 竴娆 姞杞藉氨姝
            model_name = self.model_var.get()
            if model_name.startswith("kronos-"):
                # 瀹樻柟妯 瀷
                FEATURE_LIST = OFFICIAL_MODEL_FEATURES
            else:
                # 鑷 畾涔夋 鍨?
                FEATURE_LIST = CUSTOM_MODEL_FEATURES

            # 鏇存柊kronos_model_name锛岀 淇濅笅娆 垏鎹 椂姝
            self.kronos_model_name = model_name

            if self.kronos_predictor is None:
                self.kronos_predictor = KronosPredictor(
                    self.kronos_model, self.kronos_tokenizer, max_context=512,
                    feature_list=FEATURE_LIST
                )

            predictor = self.kronos_predictor

            # 鏇存柊鐘舵 ?
            self.viz_status_label.config(text="姝 湪鐢熸垚棰勬祴...")
            self.viz_canvas.get_tk_widget().update()

            # 鍑嗗 鏁版嵁 - Kronos妯 瀷闇 瑕佹 纭 殑鍒楀悕
            x_df = df.iloc[:lookback].copy()

            # 纭 繚鏈夊繀瑕佺殑鍒楋紙鍏煎 涓嶅悓鍒楀悕锛?
            # 澶勭悊volume/vol鍒?
            if "volume" in x_df.columns and "vol" not in x_df.columns:
                x_df["vol"] = x_df["volume"]
            elif "vol" in x_df.columns and "volume" not in x_df.columns:
                x_df["volume"] = x_df["vol"]
            elif "vol" not in x_df.columns and "volume" not in x_df.columns:
                # 濡傛灉閮芥病鏈夛紝鍒涘缓榛樿 鍊?
                x_df["vol"] = x_df["close"] * 100
                x_df["volume"] = x_df["vol"]

            # 澶勭悊amount/amt鍒?
            if "amount" in x_df.columns and "amt" not in x_df.columns:
                x_df["amt"] = x_df["amount"]
            elif "amt" in x_df.columns and "amount" not in x_df.columns:
                x_df["amount"] = x_df["amt"]
            elif "amt" not in x_df.columns and "amount" not in x_df.columns:
                # 濡傛灉閮芥病鏈夛紝鐢 垚浜 噺*骞冲潎浠锋牸浼扮畻
                x_df["amt"] = x_df["vol"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
                x_df["amount"] = x_df["amt"]

            # 杞 崲鏃堕棿鎴充负DatetimeIndex锛圞ronos妯 瀷瑕佹眰锛?
            x_timestamp = pd.DatetimeIndex(df.iloc[:lookback]["timestamp"])

            # 纭 畾y_timestamp
            if viz_mode == "棰勬祴鏈 潵":
                # 棰勬祴鏈 潵妯 紡锛氱洿鎺 垱寤烘湭鏉 椂闂存埑
                last_timestamp = x_timestamp[-1]
                freq = self._get_timeframe_freq(timeframe)
                y_timestamp = pd.date_range(
                    start=last_timestamp + pd.Timedelta(freq),
                    periods=pred_len,
                    freq=freq,
                )
                self.log(f"[棰勬祴鏈 潵] 棰勬祴鏈 潵 {pred_len} 涓 懆鏈?)"
            else:
                # 鍥炴祴杩囧幓妯 紡锛氬 鏋滄湁瓒冲 鐨勫巻鍙叉暟鎹 垯浣跨敤鍘嗗彶鏁版嵁锛屽惁鍒欏垱寤烘湭鏉 椂闂存埑
                available_future_data = max(0, len(df) - lookback)
                if available_future_data >= pred_len:
                    # 鏈夎冻澶熺殑鍘嗗彶鏁版嵁浣滀负鏈 潵鏁版嵁
                    y_timestamp = pd.DatetimeIndex(
                        df.iloc[lookback : lookback + pred_len]["timestamp"]
                    )
                    self.log(
                        f"[鍥炴祴杩囧幓] 浣跨敤鍘嗗彶鏁版嵁楠岃瘉锛屾湭鏉 暟鎹 暱搴? {len(y_timestamp)}"
                    )
                elif available_future_data > 0:
                    # 閮 垎鍘嗗彶鏁版嵁 + 閮 垎鏈 潵鏃堕棿鎴?
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

                    # 纭 畾棰戠巼
                    freq = self._get_timeframe_freq(timeframe)
                    future_periods = pred_len - available_future_data
                    future_part = pd.date_range(
                        start=last_timestamp + pd.Timedelta(freq),
                        periods=future_periods,
                        freq=freq,
                    )
                    # 鍚堝苟鍘嗗彶鏁版嵁鍜屾湭鏉 椂闂存埑
                    y_timestamp = pd.DatetimeIndex(
                        list(historical_part) + list(future_part)
                    )
                    self.log(
                        f"[鍥炴祴杩囧幓] 娣峰悎妯 紡: 鍘嗗彶{len(historical_part)} + 鏈 潵{len(future_part)}"
                    )
                else:
                    # 瀹屽叏娌 湁鍘嗗彶鏁版嵁锛屽叏閮 垱寤烘湭鏉 椂闂存埑
                    last_timestamp = x_timestamp[-1]
                    freq = self._get_timeframe_freq(timeframe)
                    y_timestamp = pd.date_range(
                        start=last_timestamp + pd.Timedelta(freq),
                        periods=pred_len,
                        freq=freq,
                    )
                    self.log(f"[鍥炴祴杩囧幓] 鏃犳湭鏉 巻鍙叉暟鎹 紝浣跨敤绾  娴?)"

            # 鍩烘湰璋冭瘯淇 伅
            self.log(f"鏃堕棿鎴充俊鎭? x闀垮害={len(x_timestamp)}, y闀垮害={len(y_timestamp)}")
            self.log(
                f"鏃堕棿鑼冨洿: x[{x_timestamp[0]}]鍒癧{x_timestamp[-1]}], y[{y_timestamp[0]}]鍒癧{y_timestamp[-1]}]"
            )

            # 鐢熸垚棰勬祴鍓嶅厛璁 畻鎶 鏈 寚鏍囩壒寰?
            x_df_with_features = self._calculate_kronos_features(x_df, FEATURE_LIST)

            # 閫夋嫨Kronos闇 瑕佺殑鍒?
            kronos_columns = FEATURE_LIST

            # 鐢熸垚棰勬祴
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

            # 鏇存柊鐘舵 ?
            self.viz_status_label.config(text="姝 湪缁樺埗鍥捐 ...")
            self.viz_canvas.get_tk_widget().update()

            # 娓呯 涔嬪墠鐨勫浘琛?
            self.viz_figure.clear()

            # 鍒涘缓瀛愬浘
            ax1 = self.viz_figure.add_subplot(2, 1, 1)
            ax2 = self.viz_figure.add_subplot(2, 1, 2, sharex=ax1)

            # 鍑嗗 鏃堕棿杞存暟鎹?
            if viz_mode == "棰勬祴鏈 潵":
                # 棰勬祴鏈 潵妯 紡锛氬彧鏄剧 鍘嗗彶鏁版嵁 + 绾  娴?
                history_timestamps = pd.DatetimeIndex(df["timestamp"])
                history_prices = df["close"].values
                history_volumes = df["volume"].values

                # 棰勬祴鏁版嵁鏃堕棿鎴筹紙y_timestamp锛?
                pred_timestamps = y_timestamp[: len(pred_df)]  # 纭 繚闀垮害鍖归厤
                pred_prices = pred_df["close"].values
                pred_volumes = pred_df["volume"].values

                # 缁樺埗鏀剁洏浠?
                ax1.plot(
                    history_timestamps,
                    history_prices,
                    label="鍘嗗彶浠锋牸",
                    color="blue",
                    linewidth=1.5
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="棰勬祴鏈 潵",
                    color="red",
                    linewidth=1.5,
                    linestyle="--"
                )
                ax1.set_ylabel("鏀剁洏浠?(USDT)", fontsize=12)
                ax1.legend(loc="upper left", fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_title(
                    f"Kronos棰勬祴鏈 潵 - {symbol} ({timeframe})",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                # 鍥炴祴杩囧幓妯 紡锛氬巻鍙叉暟鎹 紙鍖呭惈鏈 潵楠岃瘉鏁版嵁锛?
                history_start = max(0, len(df) - lookback - pred_len)
                history_timestamps = pd.DatetimeIndex(
                    df["timestamp"].iloc[history_start:]
                )
                history_prices = df["close"].iloc[history_start:].values
                history_volumes = df["volume"].iloc[history_start:].values

                # 棰勬祴鏁版嵁鏃堕棿鎴筹紙y_timestamp锛?
                pred_timestamps = y_timestamp[: len(pred_df)]  # 纭 繚闀垮害鍖归厤
                pred_prices = pred_df["close"].values
                pred_volumes = pred_df["volume"].values

                # 缁樺埗鏀剁洏浠?
                ax1.plot(
                    history_timestamps,
                    history_prices,
                    label="瀹為檯浠锋牸",
                    color="blue",
                    linewidth=1.5
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="棰勬祴浠锋牸",
                    color="red",
                    linewidth=1.5,
                    linestyle="--"
                )
                ax1.set_ylabel("鏀剁洏浠?(USDT)", fontsize=12)
                ax1.legend(loc="upper left", fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_title(
                    f"Kronos鍥炴祴杩囧幓 - {symbol} ({timeframe})",
                    fontsize=14,
                    fontweight="bold",
                )

            # 缁樺埗鎴愪氦閲?
            if viz_mode == "棰勬祴鏈 潵":
                ax2.plot(
                    history_timestamps,
                    history_volumes,
                    label="鍘嗗彶鎴愪氦閲?,"
                    color="blue",
                    linewidth=1.5,
                )
                ax2.plot(
                    pred_timestamps,
                    pred_volumes,
                    label="棰勬祴鎴愪氦閲?,"
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                )
            else:
                ax2.plot(
                    history_timestamps,
                    history_volumes,
                    label="瀹為檯鎴愪氦閲?,"
                    color="blue",
                    linewidth=1.5,
                )
                ax2.plot(
                    pred_timestamps,
                    pred_volumes,
                    label="棰勬祴鎴愪氦閲?,"
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                )
            ax2.set_ylabel("鎴愪氦閲?, fontsize=12)"
            ax2.set_xlabel("鏃堕棿", fontsize=12)
            ax2.legend(loc="upper left", fontsize=10)
            ax2.grid(True, alpha=0.3)

            # 璋冩暣甯冨眬
            self.viz_figure.tight_layout()

            # 鏇存柊鐢诲竷
            self.viz_canvas.draw()

            # 鎭  鎸夐挳鐘舵 ?
            self.viz_button.config(state=tk.NORMAL)
            if viz_mode == "棰勬祴鏈 潵":
                self.viz_status_label.config(
                    text=f"[棰勬祴鏈 潵] 鍙  鍖栧畬鎴愶紒棰勬祴闀垮害: {pred_len}鍛 湡"
                )
                # 鏇存柊宸 晶灏忓浘琛?
                self._update_small_prediction_chart(history_timestamps, history_prices,
                                                    pred_timestamps, pred_prices, symbol, timeframe)
            else:
                self.viz_status_label.config(
                    text=f"[鍥炴祴杩囧幓] 鍙  鍖栧畬鎴愶紒棰勬祴闀垮害: {pred_len}鍛 湡"
                )

        except Exception as e:
            # 閿欒 澶勭悊
            self.viz_status_label.config(text=f"閿欒 : {str(e)}")
            self.viz_button.config(state=tk.NORMAL)
            import traceback

            traceback.print_exc()
            # 灏嗛敊璇 俊鎭 緭鍑哄埌鏃 織
            self.log(f"鍙  鍖栫敓鎴愰敊璇? {str(e)}")

    def _draw_prediction_chart(self):
        """缁樺埗Kronos棰勬祴K绾胯蛋鍔垮浘琛?"
        try:
            if self.prediction_ax is None:
                return

            # 娓呯 鍥捐
            self.prediction_ax.clear()

            data = self.kronos_analysis_data

            if data.get('history_prices') is None or len(data.get('history_prices', [])) == 0:
                # 娌 湁鍒嗘瀽鏁版嵁鏃舵樉绀虹瓑寰呯姸鎬?
                self.prediction_ax.text(0.5, 0.5, '绛夊緟Kronos鍒嗘瀽...',
                                        ha='center', va='center',
                                        fontsize=14, color='gray',
                                        transform=self.prediction_ax.transAxes)
                self.prediction_ax.set_title('Kronos棰勬祴璧板娍', fontsize=11, fontweight='bold')
                self.prediction_ax.axis('off')
            else:
                # 鏈夊垎鏋愭暟鎹 紝缁樺埗鍘嗗彶鍜岄 娴嬩环鏍?
                self.prediction_ax.set_facecolor('#f8f9fa')

                history_timestamps = data.get('history_timestamps', [])
                history_prices = data.get('history_prices', [])
                pred_timestamps = data.get('pred_timestamps', [])
                pred_prices = data.get('pred_prices', [])
                trend_direction = data.get('trend_direction', 'NEUTRAL')

                # 缁樺埗鍘嗗彶浠锋牸
                self.prediction_ax.plot(history_timestamps, history_prices,
                                        label='鍘嗗彶', color='#3498db', linewidth=1.0)

                # 缁樺埗棰勬祴浠锋牸
                self.prediction_ax.plot(pred_timestamps, pred_prices,
                                        label='棰勬祴', color='#e74c3c', linewidth=1.0)

                # 璁剧疆鏍囬 鍜屾爣绛?
                direction_text = '鈫?涓婃定' if trend_direction == 'LONG' else '鈫?涓嬭穼'
                self.prediction_ax.set_title(f'Kronos棰勬祴璧板娍 ({direction_text})', fontsize=11, fontweight='bold')

                # 闅愯棌X/Y杞寸殑鏁板瓧鍜屾爣绛?
                self.prediction_ax.set_xticks([])
                self.prediction_ax.set_yticks([])

                # 闅愯棌鎵 鏈夎竟妗?
                self.prediction_ax.spines['top'].set_visible(False)
                self.prediction_ax.spines['right'].set_visible(False)
                self.prediction_ax.spines['left'].set_visible(False)
                self.prediction_ax.spines['bottom'].set_visible(False)

                # 鍙 樉绀虹綉鏍?
                self.prediction_ax.grid(True, alpha=0.3)

            # 鏇存柊鐢诲竷锛堜娇鐢 洿澶 殑tight_layout鏉 渶澶 寲鍥捐 鏄剧 闈  锛?
            self.prediction_fig.tight_layout(pad=1.0)
            if hasattr(self, 'prediction_canvas') and self.prediction_canvas:
                self.prediction_canvas.draw()

        except Exception as e:
            print(f"缁樺埗棰勬祴鍥捐 澶辫触: {e}")
            import traceback
            traceback.print_exc()

    def update_kronos_analysis(self, history_timestamps=None, history_prices=None,
                               pred_timestamps=None, pred_prices=None,
                               trend_direction='NEUTRAL', trend_strength=0,
                               pred_change=0, threshold=0.008):
        """鏇存柊Kronos鍒嗘瀽鏁版嵁骞跺埛鏂板浘琛?"

        Args:
            history_timestamps: 鍘嗗彶鏃堕棿鎴虫暟缁?
            history_prices: 鍘嗗彶浠锋牸鏁扮粍
            pred_timestamps: 棰勬祴鏃堕棿鎴虫暟缁?
            pred_prices: 棰勬祴浠锋牸鏁扮粍
            trend_direction: 'LONG' 鎴?'SHORT'
            trend_strength: 瓒嬪娍寮哄害鍊?
            pred_change: 棰勬祴鍙樺寲鍊?(灏忔暟)
            threshold: 闃堝 ?
        ""
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

            # 鍒锋柊鍥捐
            self._draw_prediction_chart()

        except Exception as e:
            print(f"鏇存柊Kronos鍒嗘瀽鏁版嵁澶辫触: {e}")
            import traceback
            traceback.print_exc()

    def _update_small_prediction_chart(self, history_timestamps, history_prices,
                                       pred_timestamps, pred_prices, symbol, timeframe):
        """鏇存柊宸 晶鐨凨ronos棰勬祴灏忓浘琛 紙淇濈暀鏃 柟娉曞吋瀹规  級"""
        pass

    def _get_timeframe_freq(self, timeframe):
        """灏嗘椂闂村懆鏈熷瓧绗 覆杞 崲涓簆andas棰戠巼瀛楃 涓?"
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
            return "5T"  # 榛樿 5鍒嗛挓

    def _calculate_kronos_features(self, df, feature_list):
        """璁 畻Kronos棰勬祴鎵 闇 鐨勭壒寰?- 鏍规嵁feature_list鍐冲畾璁 畻鍝 簺"""
        import numpy as np

        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # 澶勭悊鎴愪氦閲忓垪 - 鍏煎 volume/vol
        if "volume" in df.columns and "vol" not in df.columns:
            df["vol"] = df["volume"]
        elif "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]
        elif "vol" not in df.columns and "volume" not in df.columns:
            df["vol"] = close * 100
            df["volume"] = df["vol"]

        # 澶勭悊鎴愪氦棰濆垪 - 鍏煎 涓嶅悓鏁版嵁婧?
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

        # 纭 繚amt鍒楀瓨鍦?
        if "amt" not in df.columns:
            df["amt"] = amount
        if "amount" not in df.columns:
            df["amount"] = amount

        # 妫 鏌 槸鍚 渶瑕佽 绠楁妧鏈 寚鏍?
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

        # 娓呯悊 NaN 鍊?
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return df

    def load_settings(self):
        try:
            from dotenv import load_dotenv

            # 绛栫暐鍚嶇 鏄犲皠
            strategy_name_map = {
                "trend": "瓒嬪娍鐖嗗彂",
                "range": "闇囪崱濂楀埄",
                "breakout": "娑堟伅绐佺牬",
                "auto": "鑷 姩绛栫暐",
                "time": "鏃堕棿绛栫暐",
            }

            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

            # 浣跨敤UTF-8缂栫爜鍔犺浇锛岄伩鍏峎indows缂栫爜闂
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

            # 杞 崲绛栫暐浠 爜涓烘樉绀哄悕绉?
            strategy = strategy_name_map.get(strategy_code, "瓒嬪娍鐖嗗彂")

            if api_key:
                self.api_key_entry.insert(0, api_key)
            if secret_key:
                self.secret_key_entry.insert(0, secret_key)
            self.leverage_var.set(leverage)
            self.symbol_var.set(symbol)
            self.strategy_var.set(strategy)
            self.model_var.set(model)
            self.min_position_var.set(min_position)

            # 鏍规嵁绛栫暐鍔犺浇榛樿 鍙傛暟锛堝拷鐣?env涓 殑绛栫暐鍙傛暟锛屼娇鐢 唴缃 殑绋冲仴鍙傛暟锛?
            strategy_code_map = {
                "瓒嬪娍鐖嗗彂": "trend",
                "闇囪崱濂楀埄": "range",
                "娑堟伅绐佺牬": "breakout",
                "鑷 姩绛栫暐": "auto",
                "鏃堕棿绛栫暐": "time",
            }
            strategy_key = strategy_code_map.get(strategy, "auto")
            self.reset_to_strategy_defaults(strategy_key)

            # 鍔犺浇纭  娆 暟璁剧疆
            self._load_confirm_counts_settings()
        except Exception as e:
            print(f"鍔犺浇閰嶇疆澶辫触: {e}")

    def save_settings(self):
        try:
            # 绛栫暐鍚嶇 鏄犲皠
            strategy_code_map = {
                "瓒嬪娍鐖嗗彂": "trend",
                "闇囪崱濂楀埄": "range",
                "娑堟伅绐佺牬": "breakout",
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

            # 杞 崲涓虹瓥鐣 唬鐮侊紙閬垮厤涓 枃缂栫爜闂  锛?
            strategy_code = strategy_code_map.get(strategy_display, "trend")

            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

            env_vars = {}
            if os.path.exists(env_path):
                try:
                    # 浣跨敤UTF-8缂栫爜璇诲彇
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if "=" in line and not line.startswith("#"):
                                key, val = line.split("=", 1)
                                env_vars[key] = val
                except UnicodeDecodeError:
                    try:
                        # 濡傛灉UTF-8澶辫触锛屽皾璇旼BK锛圵indows榛樿 锛?
                        with open(env_path, "r", encoding="gbk") as f:
                            for line in f:
                                line = line.strip()
                                if "=" in line and not line.startswith("#"):
                                    key, val = line.split("=", 1)
                                    env_vars[key] = val
                    except Exception as e:
                        print(f"璇诲彇鏃 厤缃 枃浠跺 璐? {e}")

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

            # 浣跨敤UTF-8缂栫爜淇濆瓨
            with open(env_path, "w", encoding="utf-8") as f:
                for key, val in env_vars.items():
                    f.write(f"{key}={val}\n")

            self.log(
                f"鍙傛暟宸蹭繚瀛? 闃堝 ?{threshold}, 鏉犳潌={leverage}x, 鍛 湡={timeframe}, 闂撮殧={interval}绉? AI瓒嬪娍={ai_min_trend}, AI鍋忕 ={ai_min_deviation}, 璧勯噾璐圭巼=[{min_funding}%, {max_funding}%]"
            )
        except Exception as e:
            self.log(f"淇濆瓨閰嶇疆澶辫触: {e}")

    def on_strategy_changed(self, event=None):
        strategy_map = {
            "瓒嬪娍鐖嗗彂": "trend",
            "闇囪崱濂楀埄": "range",
            "娑堟伅绐佺牬": "breakout",
            "鑷 姩绛栫暐": "auto",
            "鏃堕棿绛栫暐": "time",
        }
        strategy_key = strategy_map.get(self.strategy_var.get(), "trend")
        self.reset_to_strategy_defaults(strategy_key)

    def reset_to_strategy_defaults(self, strategy_key):
        strategy_params = {
            "trend": {
                "name": "瓒嬪娍鐖嗗彂",
                "threshold": "0.008",
                "ai_min_trend": "0.010",
                "ai_min_deviation": "0.008",
                "max_funding": "1.0",
                "min_funding": "-1.0",
                "timeframe": "5m",
                "interval": "120",
            },
            "range": {
                "name": "闇囪崱濂楀埄",
                "threshold": "0.005",
                "ai_min_trend": "0.003",
                "ai_min_deviation": "0.005",
                "max_funding": "2.0",
                "min_funding": "-2.0",
                "timeframe": "5m",
                "interval": "60",
            },
            "breakout": {
                "name": "娑堟伅绐佺牬",
                "threshold": "0.015",
                "ai_min_trend": "0.020",
                "ai_min_deviation": "0.015",
                "max_funding": "3.0",
                "min_funding": "-3.0",
                "timeframe": "5m",
                "interval": "180",
            },
            "auto": {
                "name": "鑷 姩绛栫暐",
                "threshold": "0.0047",
                "ai_min_trend": "0.0047",
                "ai_min_deviation": "0.005",
                "max_funding": "3.0",
                "min_funding": "-3.0",
                "timeframe": "5m",
                "interval": "180",
            },
            "time": {
                "name": "鏃堕棿绛栫暐",
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

        self.log(f"宸插垏鎹 瓥鐣? {params['name']}")
        self.log(
            f"鍙傛暟宸茶嚜鍔 姞杞?- 闃堝 ? {params['threshold']}, AI瓒嬪娍: {params['ai_min_trend']}, AI鍋忕 : {params['ai_min_deviation']}"
        )
        self.log(
            f"璧勯噾璐圭巼: [{params['min_funding']}%, {params['max_funding']}%], 鍛 湡: {params['timeframe']}, 闂撮殧: {params['interval']}绉?"
        )

    def toggle_api_visibility(self):
        show = "*" if not self.show_api_var.get() else ""
        self.api_key_entry.config(show=show)
        self.secret_key_entry.config(show=show)

    def log(self, message):
        self.queue.put(("log", message))
        # 鍚屾椂鍐欏叆浜 槗鏃 織鏂囦欢
        if hasattr(self, "trade_logger"):
            self.trade_logger.info(message)

    def start_interval_timer(self, interval_seconds):
        """鍚 姩浜 槗闂撮殧璁 椂鍣?"

        Args:
            interval_seconds: 浜 槗闂撮殧绉掓暟
        ""
        # 鍋滄 鐜版湁璁 椂鍣?
        self.stop_interval_timer()

        # 璁剧疆闂撮殧鍙傛暟
        self.interval_seconds = interval_seconds
        self.remaining_seconds = interval_seconds
        self.is_interval_running = True

        # 鏇存柊UI鏄剧
        self.queue.put(("progress", 0))  # 鍒濆 杩涘害0%锛堥 掑 杩涘害鏉 級
        self.queue.put(("status", f"浜 槗闂撮殧: {interval_seconds}绉?))"

        # 鏇存柊闂撮殧鏍囩
        if hasattr(self, 'interval_label'):
            self.interval_label.config(text=f"浜 槗闂撮殧: {interval_seconds}绉?)"
            self.remaining_time_var.set(f"{interval_seconds} 绉?)"

        # 鍚 姩璁 椂鍣?
        self._update_interval_timer()

    def stop_interval_timer(self):
        """鍋滄 浜 槗闂撮殧璁 椂鍣?"
        self.is_interval_running = False
        if self.interval_timer:
            self.root.after_cancel(self.interval_timer)
            self.interval_timer = None

        # 閲嶇疆UI鏄剧
        self.queue.put(("progress", 0))
        self.queue.put(("status", "绌洪棽"))

        if hasattr(self, 'interval_label'):
            self.interval_label.config(text="浜 槗闂撮殧: 绌洪棽")
            self.remaining_time_var.set("-- 绉?)"

    def _update_interval_timer(self):
        """鏇存柊闂撮殧璁 椂鍣 紙鍐呴儴鏂规硶锛?"
        if not self.is_interval_running:
            return

        # 鍑忓皯鍓 綑鏃堕棿
        self.remaining_seconds -= 1

        # 璁 畻杩涘害鐧惧垎姣旓紙閫掑 锛氫粠0%鍒?00%锛?
        if self.interval_seconds > 0:
            # 宸茶繃鏃堕棿 = 鎬婚棿闅?- 鍓 綑鏃堕棿
            elapsed_seconds = self.interval_seconds - self.remaining_seconds
            progress_percent = (elapsed_seconds / self.interval_seconds) * 100
        else:
            progress_percent = 0

        # 鏇存柊UI
        self.queue.put(("progress", progress_percent))
        self.queue.put(("status", f"涓嬫 鍒嗘瀽: {self.remaining_seconds}绉?))"

        # 鏇存柊闂撮殧鏍囩 鍜屽墿浣欐椂闂?
        if hasattr(self, 'interval_label'):
            self.interval_label.config(text=f"浜 槗闂撮殧: {self.interval_seconds}绉?)"
            self.remaining_time_var.set(f"{self.remaining_seconds} 绉?)"

        # 濡傛灉鏃堕棿鍒帮紝閲嶅惎璁 椂鍣?
        if self.remaining_seconds > 0:
            # 姣?000ms锛?绉掞級鏇存柊涓 娆?
            self.interval_timer = self.root.after(1000, self._update_interval_timer)
        else:
            # 鏃堕棿鍒帮紝杈撳嚭娑堟伅骞跺噯澶囬噸缃?
            self.log(f"浜 槗闂撮殧缁撴潫锛屽紑濮嬫柊涓 杞 垎鏋?..")
            # 杩涘害淇濇寔100%涓 绉掞紝鐒跺悗閲嶇疆
            self.root.after(1000, self._reset_interval_timer)

    def _reset_interval_timer(self):
        """閲嶇疆闂撮殧璁 椂鍣?"
        if not self.is_interval_running:
            return

        # 閲嶇疆鍓 綑鏃堕棿
        self.remaining_seconds = self.interval_seconds

        # 閲嶇疆杩涘害涓?%
        self.queue.put(("progress", 0))

        # 缁 画璁 椂鍣?
        if self.is_interval_running:
            self.interval_timer = self.root.after(1000, self._update_interval_timer)

    def set_progress(self, value):
        self.queue.put(("progress", value))
        # 璋冭瘯鏃 織
        # self.queue.put(("log", f"[璋冭瘯] 璁剧疆杩涘害: {value}"))

    def set_status(self, status):
        self.queue.put(("status", status))

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()

                if msg_type == "log":
                    # 鏄剧 鍦 粓绔 腑
                    self.terminal.insert(tk.END, f"[{self.get_time()}] {data}\n")
                    # 鍙 湁褰撴粴鍔 潯鍦 渶搴曢儴鏃舵墠鑷 姩璺熼殢
                    scroll_position = self.terminal.yview()
                    if scroll_position[1] >= 0.99:
                        self.terminal.see(tk.END)

                    # 妫 娴嬫槸鍚 槸澶氭櫤鑳戒綋绯荤粺鐨勬棩蹇楋紝濡傛灉鏄 紝涔熸樉绀哄湪AI绛栫暐涓 績鍜屽疄鐩樼洃鎺 棩蹇椾腑
                    multi_agent_keywords = [
                        "[澶氭櫤鑳戒綋绯荤粺]", "[FinGPT]", "[绛栫暐鍗忚皟鍣 ",
                        "Kronos鍒嗘瀽", "鑸嗘儏鍒嗘瀽", "淇 彿杩囨护", "鍗忚皟鍣?,"
                        "甯傚満鎯呯华", "椋庨櫓绛夌骇", "浜 槗寤鸿 ", "CoinGecko", "甯佸畨"
                    ]

                    exclude_keywords = [
                        "[Qwen鏂伴椈澶勭悊鍣 ", "[绀句氦濯掍綋鎯呯华]", "qwen_news_processor",
                        "social_sentiment"
                    ]

                    is_multi_agent_log = any(keyword in data for keyword in multi_agent_keywords)
                    is_excluded_log = any(keyword in data for keyword in exclude_keywords)

                    if is_multi_agent_log and not is_excluded_log:
                        # 纭 畾鏃 織绾 埆
                        level = "INFO"
                        if any(keyword in data for keyword in ["鉁?, "鎴愬姛", "瀹屾垚"]):"
                            level = "SUCCESS"
                        elif any(keyword in data for keyword in ["鈿?, "璀 憡", "杩囨护", "椋庨櫓"]):"
                            level = "WARNING"
                        elif any(keyword in data for keyword in ["鉁?, "閿欒 ", "澶辫触"]):"
                            level = "ERROR"

                        # 璁板綍鍒癆I绛栫暐涓 績鍜屽疄鐩樼洃鎺 棩蹇?
                        self._log_live_message(data, level)
                elif msg_type == "progress":
                    self.progress_var.set(data)
                    # 璋冭瘯鏃 織
                    # self.status_label.config(text=f"杩涘害: {data}%")
                elif msg_type == "status":
                    self.status_label.config(text=data)
                elif msg_type == "kronos_analysis":
                    # 鏇存柊Kronos瓒嬪娍鍒嗘瀽鍥捐
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
            messagebox.showerror("閿欒 ", "璇疯緭鍏 PI Key鍜孲ecret Key")
            return

        # 纭 繚鍓嶄竴涓 嚎绋嬪凡瀹屽叏鍋滄
        if self.trading_thread is not None and self.trading_thread.is_alive():
            self.log("绛夊緟鍓嶄竴涓 氦鏄撶嚎绋嬪畬鍏 仠姝?..")
            if self.stop_event is not None:
                self.stop_event.set()
            self.trading_thread.join(timeout=5)
            if self.trading_thread.is_alive():
                self.log("璀 憡: 鍓嶄竴涓 嚎绋嬫湭鑳藉湪5绉掑唴鍋滄 锛屽己鍒跺惎鍔 柊绾跨 ")

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
        self.log("浜 槗绯荤粺鍚 姩")
        self.log("=" * 60)

    def stop_trading(self):
        self._stop_balance_recording()
        if self.stop_event is not None:
            self.stop_event.set()

        self.log("姝 湪鍋滄 浜 槗...")

        # 鎭  榛樿 鍙傛暟
        if hasattr(self, 'strategy') and self.strategy is not None:
            self.log("鎭  榛樿 鍙傛暟...")
            try:
                self.strategy.restore_default_parameters()
            except Exception as e:
                self.log(f"鎭  榛樿 鍙傛暟鏃跺嚭閿? {e}")

        # 绛夊緟绾跨 鐪熸 缁撴潫锛堟渶澶氱瓑寰?0绉掞級
        if self.trading_thread is not None and self.trading_thread.is_alive():
            self.log("绛夊緟浜 槗绾跨 瀹屽叏鍋滄 ...")
            try:
                self.trading_thread.join(timeout=10)
                if self.trading_thread.is_alive():
                    self.log("璀 憡: 浜 槗绾跨 浠嶅湪杩愯 锛屽己鍒剁粓姝?)"
                    # 寮哄埗缁堟  - 璁剧疆鏍囧織
                    self._force_stop_trading = True
            except Exception as e:
                self.log(f"绛夊緟绾跨 鍋滄 鏃跺嚭閿? {e}")

        # 鎭  鏍囧噯杈撳嚭
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        self.stop_interval_timer()
        self.stop_loss_check_timer()
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.set_status("宸插仠姝?)"
        self.log("浜 槗绯荤粺宸插仠姝?)"

    def run_trading(self):
        self.set_progress(0)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector

        try:
            self.log("姝 湪鍒濆 鍖栦氦鏄撶郴缁?..")
            self.set_status("鍔犺浇妯 瀷...")

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
                "瓒嬪娍鐖嗗彂": "trend",
                "闇囪崱濂楀埄": "range",
                "娑堟伅绐佺牬": "breakout",
                "鑷 姩绛栫暐": "auto",
                "鏃堕棿绛栫暐": "time",
            }
            strategy_type = strategy_map.get(self.strategy_var.get(), "trend")

            # 5濂椾氦鏄撶瓥鐣 己鍒朵娇鐢?m鏃堕棿鍛 湡锛堝洜涓烘 鍨嬭 缁冩暟鎹 槸5m锛?
            is_professional_strategy = self.strategy_var.get() in strategy_map
            if is_professional_strategy:
                if timeframe != "5m":
                    self.log(f"鈿狅笍 妫 娴嬪埌閫夋嫨浜唟self.strategy_var.get()}绛栫暐锛屽己鍒朵娇鐢?m鏃堕棿鍛 湡")
                    timeframe = "5m"
                    self.timeframe_var.set("5m")

            self.log(f"浜 槗瀵? {symbol}")
            self.log(f"绛栫暐: {self.strategy_var.get()}")
            self.log(f"妯 瀷: {model_name}")
            self.log(f"鍛 湡: {timeframe}")
            self.log(f"鏉犳潌: {leverage}x")
            self.log(f"鏈 灏忎粨浣? ${min_position:.2f}")
            self.log(f"闂撮殧: {interval}绉?)"
            self.log(f"闃堝 ? {self.threshold_var.get()}")
            self.log(f"AI瓒嬪娍鏈 灏? {ai_min_trend}")
            self.log(f"AI鍋忕 鏈 灏? {ai_min_deviation}")
            self.log(f"璧勯噾璐圭巼: [{min_funding}%, {max_funding}%]")

            BinanceAPI()

            # 鑾峰彇瀹屾暣鐨凙I绛栫暐閰嶇疆
            ai_strategy_config = None
            if hasattr(self, "_get_ai_strategy_config_from_ui"):
                try:
                    ai_strategy_config = self._get_ai_strategy_config_from_ui()
                    self.log(f"鉁?宸茶幏鍙朅I绛栫暐閰嶇疆")
                except Exception as e:
                    self.log(f"鈿?鑾峰彇AI绛栫暐閰嶇疆澶辫触: {e}")

            # 瀹氫箟绾跨 瀹夊叏鐨凨ronos鍒嗘瀽鍥炶皟鍑芥暟
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

            self.log("姝 湪鍒涘缓绛栫暐...")
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

            # 璁剧疆纭  娆 暟鍜屾柊鍙傛暟 - 浠嶢I绛栫暐閰嶇疆闈 澘璇诲彇
            try:
                # 浠嶢I绛栫暐閰嶇疆闈 澘璇诲彇鏈 鏂板弬鏁?
                if hasattr(self, "ai_strategy_config_vars"):
                    entry_count = int(self.ai_strategy_config_vars.get("strategy.entry_confirm_count", self.entry_confirm_count_var).get())
                    reverse_count = int(self.ai_strategy_config_vars.get("strategy.reverse_confirm_count", self.reverse_confirm_count_var).get())
                    consecutive_pred = int(self.ai_strategy_config_vars.get("strategy.require_consecutive_prediction", self.require_consecutive_prediction_var).get())
                    post_entry_hours = float(self.ai_strategy_config_vars.get("strategy.post_entry_hours", self.post_entry_hours_var).get())
                    take_profit_min_pct = float(self.ai_strategy_config_vars.get("strategy.take_profit_min_pct", self.take_profit_min_pct_var).get())
                else:
                    # 闄嶇骇鍒版棫鍙橀噺
                    entry_count = int(self.entry_confirm_count_var.get())
                    reverse_count = int(self.reverse_confirm_count_var.get())
                    consecutive_pred = int(self.require_consecutive_prediction_var.get())
                    post_entry_hours = float(self.post_entry_hours_var.get())
                    take_profit_min_pct = float(self.take_profit_min_pct_var.get())

                # 楠岃瘉鑼冨洿
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

                self.log(f"鍙傛暟璁剧疆: 寮 浠搟entry_count}娆? 骞充粨{reverse_count}娆? 杩炵画棰勬祴{consecutive_pred}娆? 寮 浠撳悗璁 椂{post_entry_hours}灏忔椂, 鏈 灏忔 鐩坽take_profit_min_pct}%")
            except Exception as e:
                self.log(f"璁剧疆鍙傛暟澶辫触: {e}, 浣跨敤榛樿 鍊?)"

            # 璁剧疆鎬 兘鐩戞帶鍣 殑绛栫暐瀹炰緥锛堝 鏋滃瓨鍦 級
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                try:
                    self.performance_monitor.set_strategy_instance(self.strategy)
                    self.log("鎬 兘鐩戞帶鍣 凡杩炴帴鍒扮瓥鐣 疄渚?)"

                    # 濡傛灉鑷 姩鍖栦紭鍖栧凡鍚 敤浣嗘  兘鐩戞帶鏈 惎鍔 紝鐜板湪鍚 姩瀹?
                    if self.is_auto_optimization_enabled and not self.performance_monitor.is_monitoring:
                        self.performance_monitor.start_monitoring()
                        self.log("鎬 兘鐩戞帶鍣 凡鍚 姩")

                except Exception as e:
                    self.log(f"璁剧疆鎬 兘鐩戞帶鍣 瓥鐣 疄渚嬪 璐? {e}")

            # 鑾峰彇鍒濆 璧勯噾锛堝悎绾 粨浣嶆 昏祫閲戯紝鍖呭惈鎸佷粨鐩堜簭锛?
            initial_balance = self.strategy.binance.get_total_balance()
            self.initial_futures_balance = initial_balance if initial_balance else 0.0
            self.optimization_baseline_balance = self.initial_futures_balance
            self.update_fund_display()
            self.log(f"鍒濆 鍚堢害璧勯噾(鎬昏祫閲?: ${self.initial_futures_balance:.2f}")
            self.log(f"AI浼樺寲鍩哄噯璧勯噾: ${self.optimization_baseline_balance:.2f}")

            # 鏈 鍚庝氦鏄撴椂闂翠繚鎸佷负None锛岃 绛栫暐鑷 繁绠 悊

            # 鍚 姩璧勯噾鏇存柊瀹氭椂鍣?
            self.update_fund_timer()
            # 鍚 姩浜忔崯妫 鏌 畾鏃跺櫒
            self.start_loss_check_timer()

            self.set_status("杩愯 涓?..")
            self.start_interval_timer(interval)
            self.log("寮 濮嬩氦鏄撳惊鐜?..")

            self.strategy.run_loop(
                interval_seconds=interval, stop_event=self.stop_event
            )

        except Exception as e:
            self.log(f"閿欒 : {str(e)}")
            import traceback

            self.log(traceback.format_exc())
            self.stop_interval_timer()

        self.set_status("宸插仠姝?)"
        self.set_progress(0)
        self.is_running = False
        self.root.after(0, self.update_button_state)

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def update_button_state(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_fund_display(self):
        """鏇存柊璧勯噾鏄剧 """
        if hasattr(self, "strategy") and self.strategy:
            current_balance = self.strategy.get_total_balance()
            if current_balance:
                self.current_balance_label.config(text=f"${current_balance:.2f}")

                # 璁 畻鐩堜簭
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
        """瀹氭椂鏇存柊璧勯噾鏄剧 """
        if hasattr(self, "is_running") and self.is_running:
            self.update_fund_display()
            # 姣?绉掓洿鏂颁竴娆?
            self.root.after(5000, self.update_fund_timer)

    def start_loss_check_timer(self):
        """鍚 姩浜忔崯妫 鏌 畾鏃跺櫒锛堟瘡5鍒嗛挓妫 鏌 竴娆 級"""
        if self.is_running:
            self.check_loss_and_optimize()
            self.loss_check_timer = self.root.after(300000, self.start_loss_check_timer)

    def stop_loss_check_timer(self):
        """鍋滄 浜忔崯妫 鏌 畾鏃跺櫒"""
        if self.loss_check_timer is not None:
            try:
                self.root.after_cancel(self.loss_check_timer)
            except:
                pass
            self.loss_check_timer = None

    def check_loss_and_optimize(self):
        """妫 鏌 簭鎹熷拰鏃犱氦鏄撴椂闂村苟鍦 渶瑕佹椂瑙 彂AI浼樺寲"""
        if not self.is_running or not hasattr(self, "strategy"):
            return

        try:
            from datetime import datetime, timedelta
            now = datetime.now()
            optimize_triggered = False
            optimization_reason = ""
            optimization_details = None

            # ===============================
            # 1. 妫 鏌 簭鎹?
            # ===============================
            current_balance = self.strategy.get_total_balance()
            if current_balance and self.optimization_baseline_balance > 0:
                # 璁 畻浠庝紭鍖栧熀鍑嗚祫閲戜互鏉 殑浜忔崯
                loss_pct = ((self.optimization_baseline_balance - current_balance) / self.optimization_baseline_balance) * 100

                try:
                    loss_trigger = float(self.loss_trigger_var.get())
                except:
                    loss_trigger = 10.0

                self.log(f"[浜忔崯妫 鏌  鍩哄噯璧勯噾: ${self.optimization_baseline_balance:.2f}, 褰撳墠: ${current_balance:.2f}, 浜忔崯: {loss_pct:.2f}%")

                if loss_pct >= loss_trigger:
                    self.log(f"[浜忔崯瑙 彂] 浜忔崯瓒呰繃{loss_trigger}%锛屽惎鍔 I绛栫暐浼樺寲...")
                    optimize_triggered = True
                    optimization_reason = "loss_trigger"
                    optimization_details = {
                        "loss_pct": loss_pct,
                        "trigger": "浜忔崯瑙 彂"
                    }
                else:
                    self.log(f"[浜忔崯妫 鏌  浜忔崯{loss_pct:.2f}% < 闃堝 納loss_trigger}%锛屾棤闇 浼樺寲")

            # ===============================
            # 2. 妫 鏌 棤浜 槗鏃堕棿锛?2灏忔椂锛?
            # ===============================
            # 妫 鏌 槸鍚 湁鎸佷粨 - 鐢 簬鏃犱氦鏄撲紭鍖?
            has_position = False
            if hasattr(self.strategy, "current_position") and self.strategy.current_position:
                has_position = True

            if not has_position and self.strategy.last_trade_time:
                # 鍙 湁鍦 病鏈夋寔浠撲笖鏈夎繃浜 槗璁板綍鏃讹紝鎵嶆 鏌 窛绂绘渶鍚庝氦鏄撴椂闂?
                time_since_last_trade = now - self.strategy.last_trade_time
                hours_since_last_trade = time_since_last_trade.total_seconds() / 3600
                self.log(f"[鏃犱氦鏄撴 鏌  璺濇渶鍚庝氦鏄? {hours_since_last_trade:.1f}灏忔椂 (闃堝 ? 12灏忔椂)")

                if hours_since_last_trade >= 12.0:
                    self.log(f"[鏃犱氦鏄撹 鍙慮 12灏忔椂鏃犱氦鏄擄紝鍚 姩AI绛栫暐浼樺寲...")
                    optimize_triggered = True
                    optimization_reason = "no_trade"
                    optimization_details = {
                        "hours_since_last_trade": hours_since_last_trade,
                        "last_trade_time": self.strategy.last_trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "trigger": "12灏忔椂鏃犱氦鏄?"
                    }

            # ===============================
            # 3. 濡傛灉鏈変紭鍖栬 鍙戯紝鎵  浼樺寲
            # ===============================
            if optimize_triggered:
                self.trigger_ai_optimization(
                    loss_pct=optimization_details.get("loss_pct", 0.0) if optimization_reason == "loss_trigger" else 0.0,
                    optimization_reason=optimization_reason,
                    optimization_details=optimization_details
                )

        except Exception as e:
            self.log(f"妫 鏌  璐? {e}")
            import traceback
            traceback.print_exc()

    def trigger_ai_optimization(self, loss_pct: float = 0.0, optimization_reason: str = None, optimization_details: dict = None):
        """瑙 彂AI绛栫暐浼樺寲"

        Args:
            loss_pct: 浜忔崯鐧惧垎姣旓紙濡傛灉鏄 簭鎹熻 鍙戯級
            optimization_reason: 浼樺寲鍘熷洜锛?loss_trigger", "no_trade", "regular"锛?"
            optimization_details: 浼樺寲璇 粏淇 伅瀛楀吀
        ""
        try:
            self.log("姝 湪鍚 姩AI绛栫暐浼樺寲...")

            # 濡傛灉娌 湁鎻愪緵浼樺寲鍘熷洜锛屾牴鎹?loss_pct 鎺 柇
            if optimization_reason is None:
                optimization_reason = "loss_trigger" if loss_pct > 0 else "regular"

            # 鍏堟 鏌 槸鍚 湁澶氭櫤鑳戒綋绯荤粺
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                # 浣跨敤澶氭櫤鑳戒綋绯荤粺杩涜 浼樺寲
                self.log("浣跨敤澶氭櫤鑳戒綋绯荤粺浼樺寲鍙傛暟...")
                # 杩欓噷鍙 互娣诲姞鍏蜂綋鐨勪紭鍖栬皟鐢?
            else:
                # 璋冪敤鎬 兘鐩戞帶鍣 殑浼樺寲鍔熻兘
                if hasattr(self, "performance_monitor") and self.performance_monitor:
                    self.log("浣跨敤鎬 兘鐩戞帶鍣 紭鍖栧弬鏁?..")
                    self.performance_monitor.optimize_strategy()

            # 浼樺寲鍚庯紝鏇存柊鍩哄噯璧勯噾涓哄綋鍓嶈祫閲?
            current_balance = self.strategy.get_total_balance()
            if current_balance:
                self.optimization_baseline_balance = current_balance
                self.log(f"AI浼樺寲瀹屾垚锛佹柊鐨勫熀鍑嗚祫閲? ${self.optimization_baseline_balance:.2f}")

        except Exception as e:
            self.log(f"AI浼樺寲瑙 彂澶辫触: {e}")
            import traceback
            traceback.print_exc()

    def _create_system_tab(self, parent):
        """鍒涘缓绯荤粺鐩戞帶鏍囩 椤?"
        # CPU鐩戞帶妗嗘灦
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
            cpu_frame, textvariable=self.cpu_var, font=("寰 蒋闆呴粦", 14, "bold")
        ).pack()

        self.cpu_count_label = ttk.Label(parent, text=f"鏍稿績鏁? {psutil.cpu_count()}")
        self.cpu_count_label.pack(pady=(0, 10))

        # 鍐呭瓨鐩戞帶妗嗘灦
        mem_frame = ttk.LabelFrame(parent, text="鍐呭瓨", padding=10)
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
            mem_frame, textvariable=self.mem_var, font=("寰 蒋闆呴粦", 14, "bold")
        ).pack()

        self.mem_detail_label = ttk.Label(parent, text="")
        self.mem_detail_label.pack(pady=(0, 10))

        # GPU鐩戞帶妗嗘灦
        gpu_frame = ttk.LabelFrame(parent, text="GPU", padding=10)
        gpu_frame.pack(fill=tk.X, padx=5, pady=5)

        self.gpu_var = tk.StringVar(value="妫 娴嬩腑...")
        ttk.Label(
            gpu_frame, textvariable=self.gpu_var, font=("寰 蒋闆呴粦", 14, "bold")
        ).pack()

        self.gpu_detail_label = ttk.Label(parent, text="")
        self.gpu_detail_label.pack(pady=(0, 10))

        # 纾佺洏鐩戞帶妗嗘灦
        disk_frame = ttk.LabelFrame(parent, text="纾佺洏", padding=10)
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
            disk_frame, textvariable=self.disk_var, font=("寰 蒋闆呴粦", 14, "bold")
        ).pack()

        self.disk_detail_label = ttk.Label(parent, text="")
        self.disk_detail_label.pack(pady=(0, 10))

        # 娣诲姞杩涘害鏉 牱寮?
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

        # 鍚 姩绯荤粺鐩戞帶瀹氭椂鍣?
        self._update_system_info()
        self._system_timer = self.root.after(2000, self._update_system_info)

    def _update_system_info(self):
        try:
            # CPU淇 伅
            cpu_percent = psutil.cpu_percent(interval=0.5)
            self.cpu_var.set(f"{cpu_percent:.1f}%")
            self.cpu_progress["value"] = cpu_percent

            # 鍐呭瓨淇 伅
            mem = psutil.virtual_memory()
            self.mem_var.set(f"{mem.percent:.1f}%")
            self.mem_progress["value"] = mem.percent
            self.mem_detail_label.config(
                text=f"宸茬敤: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB"
            )

            # GPU淇 伅
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
                    self.gpu_var.set(f"鏄惧瓨: {gpu_util:.1f}%")
                    self.gpu_detail_label.config(
                        text=f"宸茬敤: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.1f}GB"
                    )
                else:
                    self.gpu_var.set("鏃燝PU")
                    self.gpu_detail_label.config(text="CUDA涓嶅彲鐢?)"
            except ImportError:
                self.gpu_var.set("鏃燩yTorch")
                self.gpu_detail_label.config(text="鏈 畨瑁匬yTorch")
            except Exception as e:
                self.gpu_var.set("GPU閿欒 ")
                self.gpu_detail_label.config(text=str(e)[:30])

            # 纾佺洏淇 伅
            disk = psutil.disk_usage("C:")
            self.disk_var.set(f"{disk.percent:.1f}%")
            self.disk_progress["value"] = disk.percent
            self.disk_detail_label.config(
                text=f"宸茬敤: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB"
            )

        except Exception as e:
            print(f"绯荤粺鐩戞帶鏇存柊閿欒 : {e}")

        # 姣?绉掓洿鏂颁竴娆?
        self._system_timer = self.root.after(2000, self._update_system_info)

    def get_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def _create_training_tab(self, parent):
        """鍒涘缓璁 粌鏍囩 椤?"
        # 璁 粌閰嶇疆妗嗘灦
        config_frame = ttk.LabelFrame(
            parent, text="璁 粌閰嶇疆", style="TFrame", padding=10
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # ==================== 鏁版嵁鑾峰彇閮 垎 ====================
        data_source_frame = ttk.LabelFrame(
            parent, text="鏁版嵁鑾峰彇", style="TFrame", padding=10
        )
        data_source_frame.pack(fill=tk.X, pady=(0, 10))

        # 浜 槗瀵瑰拰鍛 湡
        pair_frame = ttk.Frame(data_source_frame)
        pair_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pair_frame, text="浜 槗瀵?").pack(side=tk.LEFT, padx=(0, 5))
        self.train_symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Label(pair_frame, text="BTCUSDT", font=("TkDefaultFont", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(pair_frame, text="鍛 湡:").pack(side=tk.LEFT, padx=(0, 5))
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

        # 鏃 湡鑼冨洿
        date_frame = ttk.Frame(data_source_frame)
        date_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(date_frame, text="寮 濮嬫棩鏈?").pack(side=tk.LEFT, padx=(0, 5))
        self.train_start_date_var = tk.StringVar(value="")
        ttk.Entry(date_frame, textvariable=self.train_start_date_var, width=12).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(date_frame, text="缁撴潫鏃 湡:").pack(side=tk.LEFT, padx=(0, 5))
        self.train_end_date_var = tk.StringVar(value="")
        ttk.Entry(date_frame, textvariable=self.train_end_date_var, width=12).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Label(date_frame, text="(鏍煎紡: YYYY-MM-DD锛岀暀绌鸿幏鍙栧叏閮?").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # 涓嬭浇鎸夐挳鍜屽 鏁?
        download_frame = ttk.Frame(data_source_frame)
        download_frame.pack(fill=tk.X, pady=(0, 5))
        self.download_data_button = ttk.Button(
            download_frame, text="涓嬭浇甯佸畨鏁版嵁", command=self.download_binance_data
        )
        self.download_data_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(download_frame, text="涓嬭浇澶 暟:").pack(side=tk.LEFT, padx=(20, 5))
        self.train_days_var = tk.StringVar(value="30")
        ttk.Entry(download_frame, textvariable=self.train_days_var, width=6).pack(
            side=tk.LEFT
        )
        ttk.Label(download_frame, text="澶?).pack(side=tk.LEFT)"

        # ==================== 璁 粌閰嶇疆閮 垎 ====================
        train_config_frame = ttk.LabelFrame(
            parent, text="妯 瀷璁 粌", style="TFrame", padding=10
        )
        train_config_frame.pack(fill=tk.X, pady=(0, 10))

        # 妯 瀷閫夋嫨鍜孏PU閫夐
        model_frame = ttk.Frame(train_config_frame)
        model_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(model_frame, text="鍩烘 鍨?").pack(side=tk.LEFT, padx=(0, 10))
        self.train_base_model_var = tk.StringVar(value="Kronos-small")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.train_base_model_var,
            values=["Kronos-mini", "Kronos-small", "Kronos-base"],
            width=15,
            state="readonly",
        )
        model_combo.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="  璁惧 :").pack(side=tk.LEFT, padx=(20, 5))
        self.train_device_var = tk.StringVar(value="GPU (CUDA)")
        device_combo = ttk.Combobox(
            model_frame,
            textvariable=self.train_device_var,
            values=["GPU (CUDA)", "CPU"],
            width=10,
            state="readonly",
        )
        device_combo.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="  (GPU鏇村揩锛屾湁Nvidia鏄惧崱閫塆PU)").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # 璁 粌鏁版嵁鏂囦欢
        ttk.Label(train_config_frame, text="璁 粌鏁版嵁 (CSV):").pack(
            anchor=tk.W, pady=(5, 0)
        )
        data_frame = ttk.Frame(train_config_frame)
        data_frame.pack(fill=tk.X, pady=(0, 5))
        self.train_data_path_var = tk.StringVar(value="training_data/BTCUSDT_5m_with_indicators.csv")
        self.train_data_entry = ttk.Entry(
            data_frame, textvariable=self.train_data_path_var
        )
        self.train_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(data_frame, text="娴忚 ...", command=self.browse_train_data).pack(
            side=tk.RIGHT
        )

        # 閰嶇疆鏂囦欢锛堝彲閫夛級
        ttk.Label(train_config_frame, text="閰嶇疆鏂囦欢 (YAML, 鍙  ?:").pack(anchor=tk.W)
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
            config_file_frame, text="娴忚 ...", command=self.browse_train_config
        ).pack(side=tk.RIGHT)

        # 璁 粌鍙傛暟
        ttk.Label(train_config_frame, text="璁 粌鍙傛暟:").pack(anchor=tk.W, pady=(5, 0))
        params_frame = ttk.Frame(train_config_frame)
        params_frame.pack(fill=tk.X)

        # 绗 竴琛屽弬鏁?
        ttk.Label(params_frame, text="鍥炵湅绐楀彛:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.train_lookback_var = tk.StringVar(value="256")
        ttk.Entry(params_frame, textvariable=self.train_lookback_var, width=8).grid(
            row=0, column=1, padx=(0, 10)
        )

        ttk.Label(params_frame, text="棰勬祴绐楀彛:").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.train_predict_var = tk.StringVar(value="48")
        ttk.Entry(params_frame, textvariable=self.train_predict_var, width=8).grid(
            row=0, column=3, padx=(0, 10)
        )

        ttk.Label(params_frame, text="鎵规 澶 皬:").grid(
            row=0, column=4, sticky=tk.W, padx=(0, 5)
        )
        self.train_batch_var = tk.StringVar(value="64")
        ttk.Entry(params_frame, textvariable=self.train_batch_var, width=8).grid(
            row=0, column=5, padx=(0, 10)
        )

        ttk.Label(params_frame, text="瀛 範鐜?").grid(
            row=0, column=6, sticky=tk.W, padx=(0, 5)
        )
        self.train_lr_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.train_lr_var, width=10).grid(
            row=0, column=7, padx=(0, 5)
        )

        # 绗 簩琛屽弬鏁?
        ttk.Label(params_frame, text="Tokenizer杞 暟:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_epochs_var = tk.StringVar(value="30")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_epochs_var, width=8
        ).grid(row=1, column=1, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="妯 瀷杞 暟:").grid(
            row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_basemodel_epochs_var = tk.StringVar(value="20")
        ttk.Entry(
            params_frame, textvariable=self.train_basemodel_epochs_var, width=8
        ).grid(row=1, column=3, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="Tokenizer瀛 範鐜?").grid(
            row=1, column=4, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_lr_var = tk.StringVar(value="0.0002")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_lr_var, width=10
        ).grid(row=1, column=5, padx=(0, 10), pady=(5, 0))

        ttk.Label(params_frame, text="鏃 織闂撮殧:").grid(
            row=1, column=6, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_log_interval_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.train_log_interval_var, width=8).grid(
            row=1, column=7, padx=(0, 5), pady=(5, 0)
        )

        # 绗 笁琛屽弬鏁?
        ttk.Label(params_frame, text="璁 粌闆嗘瘮渚?").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_ratio_var = tk.StringVar(value="0.9")
        ttk.Entry(params_frame, textvariable=self.train_ratio_var, width=8).grid(
            row=2, column=1, padx=(0, 10), pady=(5, 0)
        )

        ttk.Label(params_frame, text="楠岃瘉闆嗘瘮渚?").grid(
            row=2, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.val_ratio_var = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.val_ratio_var, width=8).grid(
            row=2, column=3, padx=(0, 10), pady=(5, 0)
        )

        ttk.Label(params_frame, text="鏁版嵁鍔犺浇绾跨 :").grid(
            row=2, column=4, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_num_workers_var = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.train_num_workers_var, width=8).grid(
            row=2, column=5, padx=(0, 10), pady=(5, 0)
        )

        # 绗 洓琛屽弬鏁?- 瀹為獙鍚嶇 鍜屼繚瀛樿矾寰?
        ttk.Label(params_frame, text="瀹為獙鍚嶇 :").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_exp_name_var = tk.StringVar(value="custom_model")
        ttk.Entry(params_frame, textvariable=self.train_exp_name_var, width=15).grid(
            row=3, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.W
        )

        ttk.Label(params_frame, text="淇濆瓨璺 緞:").grid(
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
            path_frame, text="娴忚 ", command=self.browse_save_path, width=5
        ).pack(side=tk.LEFT)

        # 绗 簲琛屽弬鏁?- 妯 瀷瀛愭枃浠跺 鍚嶇
        ttk.Label(params_frame, text="Tokenizer鏂囦欢澶?").grid(
            row=4, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_tokenizer_folder_var = tk.StringVar(value="tokenizer")
        ttk.Entry(
            params_frame, textvariable=self.train_tokenizer_folder_var, width=12
        ).grid(row=4, column=1, padx=(0, 10), pady=(5, 0), sticky=tk.W)

        ttk.Label(params_frame, text="妯 瀷鏂囦欢澶?").grid(
            row=4, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.train_basemodel_folder_var = tk.StringVar(value="basemodel")
        ttk.Entry(
            params_frame, textvariable=self.train_basemodel_folder_var, width=12
        ).grid(row=4, column=3, padx=(0, 10), pady=(5, 0), sticky=tk.W)

        ttk.Label(
            params_frame,
            text="(鏈 缁堟 鍨? {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/{妯 瀷鏂囦欢澶箎/best_model)",
        ).grid(row=4, column=4, columnspan=4, sticky=tk.W, pady=(5, 0))

        # 澶嶉 夋 閫夐
        check_frame = ttk.Frame(train_config_frame)
        check_frame.pack(fill=tk.X, pady=(5, 0))
        self.train_tokenizer_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            check_frame, text="璁 粌Tokenizer", variable=self.train_tokenizer_var
        ).pack(side=tk.LEFT, padx=10)
        self.train_basemodel_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            check_frame, text="璁 粌鍩烘 鍨?, variable=self.train_basemodel_check_var"
        ).pack(side=tk.LEFT, padx=10)
        self.train_skip_existing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            check_frame, text="璺宠繃宸插瓨鍦  缁?, variable=self.train_skip_existing_var"
        ).pack(side=tk.LEFT, padx=10)

        # 璁 粌鎺 埗鎸夐挳
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.train_button = ttk.Button(
            control_frame, text="寮 濮嬭 缁?, command=self.start_training"
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.stop_train_button = ttk.Button(
            control_frame,
            text="鍋滄 璁 粌",
            command=self.stop_training,
            state=tk.DISABLED,
        )
        self.stop_train_button.pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=15, fill=tk.Y
        )

        self.save_config_button = ttk.Button(
            control_frame, text="淇濆瓨閰嶇疆", command=self.save_train_config
        )
        self.save_config_button.pack(side=tk.LEFT, padx=5)

        self.load_config_button = ttk.Button(
            control_frame, text="鍔犺浇閰嶇疆", command=self.load_train_config
        )
        self.load_config_button.pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=15, fill=tk.Y
        )

        self.help_button = ttk.Button(
            control_frame, text="甯 姪", command=self.show_train_help
        )
        self.help_button.pack(side=tk.LEFT, padx=5)

        # 璁 粌杩涘害鏉?
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_bar = ttk.Progressbar(
            parent,
            variable=self.train_progress_var,
            maximum=100,
            mode="determinate",
            style="green.Horizontal.TProgressbar",
        )
        self.train_progress_bar.pack(fill=tk.X, pady=(0, 5))

        # 璁 粌鐘舵 佹爣绛?
        self.train_status_label = ttk.Label(parent, text="灏辩华", style="TLabel")
        self.train_status_label.pack(anchor=tk.W, pady=(0, 5))

        # 璁 粌杈撳嚭
        self.train_terminal = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#333333",
            font=("Consolas", 11),
            insertbackground="#333333",
        )
        self.train_terminal.pack(fill=tk.BOTH, expand=True)

        # 璁 粌绾跨 鍜屼簨浠?
        self.train_thread = None
        self.train_stop_event = None

    def _create_training_manager_tab(self, parent):
        """鍒涘缓璁 粌鏂囦欢绠 悊鏍囩 椤?"
        # 涓绘 鏋?
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 鏍囬
        title_label = ttk.Label(
            main_frame,
            text="璁 粌鏂囦欢绠 悊鍣?,"
            font=("寰 蒋闆呴粦", 18, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        # 璇存槑鏂囨湰
        description = (
            "姝 伐鍏风敤浜庢墦鍖呮湰鏈鸿 缁冩枃浠讹紝鎴栧皢鍏朵粬鏈哄櫒鐨勮 缁冩枃浠跺寘瑙 帇鍒版湰鏈恒 俓n"
            "鎵撳寘鏂囦欢鍖呭惈锛氳 缁冩 鍨嬨 侀厤缃 枃浠躲 佽 缁冩暟鎹  俓n"
            "瑙 寘鏃朵細瑕嗙洊鏈 満鐜版湁鏂囦欢锛岃 鍏堝 浠介噸瑕佹暟鎹  ?"
        )
        desc_label = ttk.Label(
            main_frame,
            text=description,
            font=("寰 蒋闆呴粦", 10),
            foreground="#34495e",
            justify=tk.CENTER,
            wraplength=700,
        )
        desc_label.pack(pady=(0, 30))

        # 鎿嶄綔鎸夐挳妗嗘灦
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(0, 30))

        # 鎵撳寘鎸夐挳
        self.pack_button = ttk.Button(
            button_frame,
            text="馃摝 鎵撳寘鏈 満璁 粌鏂囦欢",
            command=self._pack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.pack_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 瑙 寘鎸夐挳
        self.unpack_button = ttk.Button(
            button_frame,
            text="馃摛 瑙 寘璁 粌鏂囦欢鍒版湰鏈?,"
            command=self._unpack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.unpack_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 鐘舵 佸尯鍩?
        status_frame = ttk.LabelFrame(main_frame, text="鎿嶄綔鐘舵 ?, padding="15")"
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # 鐘舵 佹枃鏈?
        self.training_manager_status_text = tk.Text(
            status_frame,
            height=12,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.training_manager_status_text.pack(fill=tk.BOTH, expand=True)

        # 婊氬姩鏉?
        scrollbar = ttk.Scrollbar(
            status_frame, orient=tk.VERTICAL, command=self.training_manager_status_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_manager_status_text.configure(yscrollcommand=scrollbar.set)

        # 搴曢儴淇 伅
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X)

        # 褰撳墠閰嶇疆淇 伅
        self.training_config_info = tk.StringVar(value="姝 湪鍔犺浇閰嶇疆...")
        config_label = ttk.Label(
            info_frame,
            textvariable=self.training_config_info,
            font=("寰 蒋闆呴粦", 9),
            foreground="#7f8c8d",
        )
        config_label.pack(side=tk.LEFT)

        # 鐗堟湰淇 伅
        version_label = ttk.Label(
            info_frame,
            text="鐗堟湰 2.0 路 榛戠尗浜 槗绯荤粺v2.0",
            font=("寰 蒋闆呴粦", 8),
            foreground="#95a5a6",
        )
        version_label.pack(side=tk.RIGHT)

        # 鍔犺浇閰嶇疆
        self._load_training_config()

    def _get_base_dir(self):
        """鑾峰彇鍩虹 鐩 綍璺 緞锛堝吋瀹筆yInstaller鐜  锛?"
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def _load_training_config(self):
        """鍔犺浇璁 粌閰嶇疆鏂囦欢"""
        try:
            base_dir = self._get_base_dir()
            config_path = os.path.join(base_dir, "train_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.training_config = yaml.safe_load(f)

                exp_name = self.training_config.get("exp_name", "鏈 煡")
                save_path = self.training_config.get("save_path", "鏈 煡")
                data_path = self.training_config.get("data_path", "鏈 煡")

                info_text = f"瀹為獙鍚嶇 : {exp_name} | 淇濆瓨璺 緞: {save_path} | 鏁版嵁鏂囦欢: {os.path.basename(data_path)}"
                self.training_config_info.set(info_text)
                self._log_training_manager_message("鉁?閰嶇疆鏂囦欢鍔犺浇鎴愬姛", "SUCCESS")
            else:
                self.training_config = {}
                self.training_config_info.set("鏈 壘鍒伴厤缃 枃浠?train_config.yaml")
                self._log_training_manager_message("鈿狅笍 鏈 壘鍒伴厤缃 枃浠?, "WARNING")"
        except Exception as e:
            self.training_config = {}
            self.training_config_info.set("閰嶇疆鏂囦欢鍔犺浇澶辫触")
            self._log_training_manager_message(f"鉂?閰嶇疆鏂囦欢鍔犺浇澶辫触: {e}", "ERROR")

    def _log_training_manager_message(self, message, level="INFO"):
        """璁板綍娑堟伅鍒拌 缁冩枃浠剁 鐞嗗櫒鐘舵 佸尯鍩?"
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.training_manager_status_text.insert(tk.END, formatted_msg)
        self.training_manager_status_text.see(tk.END)
        self.training_manager_status_text.update()

    def _pack_training_files(self):
        """鎵撳寘鏈 満璁 粌鏂囦欢"""
        try:
            # 妫 鏌 厤缃 枃浠?
            if not hasattr(self, 'training_config') or not self.training_config:
                self._log_training_manager_message(
                    "鉂?璇峰厛纭 繚閰嶇疆鏂囦欢 train_config.yaml 瀛樺湪涓旀湁鏁?, "ERROR
                )
                return

            # 璇 棶淇濆瓨浣嶇疆
            default_filename = f"kronos_training_{self.training_config.get('exp_name', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = filedialog.asksaveasfilename(
                title="淇濆瓨璁 粌鏂囦欢鍖?,"
                defaultextension=".zip",
                initialfile=default_filename,
                filetypes=[("ZIP鍘嬬缉鍖?, "*.zip"), ("鎵 鏈夋枃浠?, "*.*")],
            )

            if not zip_path:
                self._log_training_manager_message("鎿嶄綔宸插彇娑?, "INFO")"
                return

            self._log_training_manager_message(
                f"寮 濮嬫墦鍖呰 缁冩枃浠跺埌: {os.path.basename(zip_path)}", "INFO"
            )

            # 鏀堕泦瑕佹墦鍖呯殑鏂囦欢
            files_to_pack = []

            # 1. 妯 瀷鏂囦欢
            exp_name = self.training_config.get("exp_name")
            save_path = self.training_config.get("save_path")
            if exp_name:
                # 鏌 壘妯 瀷鏂囦欢鐨勫疄闄呬綅缃?
                model_path = self._find_model_files(exp_name, save_path)
                if model_path and os.path.exists(model_path):
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, ".")
                            files_to_pack.append((full_path, rel_path))
                    self._log_training_manager_message(f"鉁?娣诲姞妯 瀷鏂囦欢: {model_path}", "SUCCESS")
                else:
                    self._log_training_manager_message(
                        f"鈿狅笍 鏈 壘鍒版 鍨嬫枃浠讹紝瀹為獙鍚嶇 : {exp_name}", "WARNING"
                    )
                    # 灏濊瘯鎼滅储鎵 鏈夊彲鑳界殑妯 瀷鏂囦欢
                    self._log_training_manager_message("姝 湪鎼滅储妯 瀷鏂囦欢...", "INFO")
                    model_files_found = self._search_all_model_files()
                    if model_files_found:
                        for file_path, rel_path in model_files_found:
                            files_to_pack.append((file_path, rel_path))
                        self._log_training_manager_message(
                            f"鉁?閫氳繃鎼滅储鎵惧埌 {len(model_files_found)} 涓  鍨嬫枃浠?,"
                            "SUCCESS",
                        )

            # 2. 閰嶇疆鏂囦欢
            base_dir = self._get_base_dir()
            config_files = ["train_config.yaml"]
            for config_file in config_files:
                full_config_path = os.path.join(base_dir, config_file)
                if os.path.exists(full_config_path):
                    files_to_pack.append((full_config_path, config_file))
                    self._log_training_manager_message(f"鉁?娣诲姞閰嶇疆鏂囦欢: {config_file}", "SUCCESS")
                else:
                    self._log_training_manager_message(f"鈿狅笍 閰嶇疆鏂囦欢涓嶅瓨鍦? {full_config_path}", "WARNING")

            # 3. 鏁版嵁鏂囦欢
            base_dir = self._get_base_dir()
            data_path = self.training_config.get("data_path")
            if data_path:
                # 灏濊瘯鎵惧埌鏁版嵁鏂囦欢
                data_file_found = False

                # 妫 鏌 粷瀵硅矾寰?
                if os.path.exists(data_path):
                    # 濡傛灉鏄 粷瀵硅矾寰勶紝杞 崲涓虹浉瀵硅矾寰?
                    try:
                        rel_path = os.path.relpath(data_path, base_dir)
                        files_to_pack.append((data_path, rel_path))
                        self._log_training_manager_message(f"鉁?娣诲姞鏁版嵁鏂囦欢: {rel_path}", "SUCCESS")
                        data_file_found = True
                    except ValueError:
                        # 濡傛灉涓嶅湪鍚屼竴椹卞姩鍣 紝浣跨敤鍩烘湰鍚嶇
                        files_to_pack.append((data_path, os.path.basename(data_path)))
                        self._log_training_manager_message(
                            f"鉁?娣诲姞鏁版嵁鏂囦欢: {os.path.basename(data_path)}", "SUCCESS"
                        )
                        data_file_found = True

                # 濡傛灉娌 壘鍒帮紝灏濊瘯鍦 raining_data鐩 綍涓 煡鎵?
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
                            self._log_training_manager_message(f"鉁?鎵惧埌鏁版嵁鏂囦欢: {arc_path}", "SUCCESS")
                            data_file_found = True
                            break

                if not data_file_found:
                    self._log_training_manager_message(f"鈿狅笍 鏁版嵁鏂囦欢涓嶅瓨鍦? {data_path}", "WARNING")
            else:
                self._log_training_manager_message("鈿狅笍 閰嶇疆鏂囦欢涓 湭鎸囧畾鏁版嵁鏂囦欢璺 緞", "WARNING")

            # 鍒涘缓ZIP鏂囦欢
            if not files_to_pack:
                self._log_training_manager_message("鉂?娌 湁鎵惧埌鍙 墦鍖呯殑鏂囦欢", "ERROR")
                return

            import zipfile
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path, arcname in files_to_pack:
                    zipf.write(file_path, arcname)
                    self._log_training_manager_message(f"  鍘嬬缉: {arcname}", "INFO")

            # 娣诲姞鍏冩暟鎹?
            metadata = {
                "鎵撳寘鏃堕棿": datetime.now().isoformat(),
                "瀹為獙鍚嶇 ": self.training_config.get("exp_name"),
                "鏂囦欢鏁伴噺": len(files_to_pack),
                "鐗堟湰": "1.0",
            }

            with zipfile.ZipFile(zip_path, "a") as zipf:
                import json
                zipf.writestr(
                    "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False)
                )

            self._log_training_manager_message(
                f"鉁?鎵撳寘瀹屾垚! 鍏辨墦鍖?{len(files_to_pack)} 涓 枃浠?, "SUCCESS
            )
            self._log_training_manager_message(f"鏂囦欢淇濆瓨鍒? {zip_path}", "INFO")

            messagebox.showinfo(
                "鎵撳寘瀹屾垚",
                f"璁 粌鏂囦欢鎵撳寘鎴愬姛!\n\n淇濆瓨浣嶇疆: {zip_path}\n鏂囦欢鏁伴噺: {len(files_to_pack)}",
            )

        except Exception as e:
            self._log_training_manager_message(f"鉂?鎵撳寘澶辫触: {e}", "ERROR")

    def _unpack_training_files(self):
        """瑙 寘璁 粌鏂囦欢鍒版湰鏈?"
        try:
            # 閫夋嫨ZIP鏂囦欢
            zip_path = filedialog.askopenfilename(
                title="閫夋嫨璁 粌鏂囦欢鍖?,"
                filetypes=[("ZIP鍘嬬缉鍖?, "*.zip"), ("鎵 鏈夋枃浠?, "*.*")],
            )

            if not zip_path:
                self._log_training_manager_message("鎿嶄綔宸插彇娑?, "INFO")"
                return

            self._log_training_manager_message(f"寮 濮嬭 鍖呮枃浠? {os.path.basename(zip_path)}", "INFO")

            # 璇诲彇鍏冩暟鎹?
            metadata = None
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    if "metadata.json" in zipf.namelist():
                        with zipf.open("metadata.json") as f:
                            import json
                            metadata = json.load(f)
                            self._log_training_manager_message(
                                f"鉁?璇诲彇鍏冩暟鎹? {metadata.get('瀹為獙鍚嶇 ', '鏈 煡')}",
                                "SUCCESS",
                            )
            except:
                self._log_training_manager_message("鈿狅笍 鏃犳硶璇诲彇鍏冩暟鎹?, "WARNING")"

            # 纭  瑙 寘
            if metadata:
                confirm_msg = f"鍗冲皢瑙 寘璁 粌鏂囦欢:\n\n瀹為獙鍚嶇 : {metadata.get('瀹為獙鍚嶇 ', '鏈 煡')}\n鎵撳寘鏃堕棿: {metadata.get('鎵撳寘鏃堕棿', '鏈 煡')}\n鏂囦欢鏁伴噺: {metadata.get('鏂囦欢鏁伴噺', '鏈 煡')}\n\n瑙 寘灏嗚 鐩栨湰鏈虹幇鏈夋枃浠讹紝鏄 惁缁 画锛?"
            else:
                confirm_msg = f"鍗冲皢瑙 寘鏂囦欢: {os.path.basename(zip_path)}\n\n瑙 寘灏嗚 鐩栨湰鏈虹幇鏈夋枃浠讹紝鏄 惁缁 画锛?"

            if not messagebox.askyesno("纭  瑙 寘", confirm_msg):
                self._log_training_manager_message("瑙 寘鎿嶄綔宸插彇娑?, "INFO")"
                return

            # 瑙 寘鏂囦欢
            extracted_files = []
            skipped_files = []
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zipf:
                # 鍏堣幏鍙栨枃浠跺垪琛?
                file_list = [f for f in zipf.namelist() if f != "metadata.json"]

                for filename in file_list:
                    try:
                        # 楠岃瘉鏂囦欢璺 緞
                        if not self._validate_zip_path(filename):
                            self._log_training_manager_message(
                                f"  鈿狅笍 璺宠繃涓嶅畨鍏 枃浠? {filename}", "WARNING"
                            )
                            skipped_files.append(filename)
                            continue

                        # 妫 鏌 洰鏍囩洰褰曟槸鍚 湪椤圭洰鑼冨洿鍐?
                        target_path = os.path.join(".", filename)
                        target_dir = os.path.dirname(target_path)

                        # 纭 繚鐩 爣鐩 綍瀛樺湪
                        os.makedirs(target_dir, exist_ok=True)

                        # 鎻愬彇鏂囦欢
                        zipf.extract(filename, ".")
                        extracted_files.append(filename)
                        self._log_training_manager_message(f"  瑙 帇: {filename}", "INFO")
                    except Exception as e:
                        self._log_training_manager_message(f"  鈿狅笍 瑙 帇澶辫触 {filename}: {e}", "WARNING")

            if extracted_files:
                self._log_training_manager_message(
                    f"鉁?瑙 寘瀹屾垚! 鍏辫 鍘?{len(extracted_files)} 涓 枃浠?, "SUCCESS
                )
            else:
                self._log_training_manager_message("鈿狅笍 娌 湁瑙 帇浠讳綍鏂囦欢", "WARNING")

            if skipped_files:
                self._log_training_manager_message(
                    f"鈿狅笍 璺宠繃浜?{len(skipped_files)} 涓 笉瀹夊叏鏂囦欢", "WARNING"
                )

            # 閲嶆柊鍔犺浇閰嶇疆
            self._load_training_config()

            messagebox.showinfo(
                "瑙 寘瀹屾垚", f"璁 粌鏂囦欢瑙 寘鎴愬姛!\n\n瑙 帇鏂囦欢: {len(extracted_files)} 涓?"
            )

        except Exception as e:
            self._log_training_manager_message(f"鉂?瑙 寘澶辫触: {e}", "ERROR")

    def _validate_zip_path(self, filename):
        """楠岃瘉ZIP鏂囦欢涓 殑璺 緞鏄 惁瀹夊叏"""
        # 闃叉 璺 緞閬嶅巻鏀诲嚮
        if ".." in filename or filename.startswith("/") or ":" in filename:
            return False

        # 闃叉 缁濆 璺 緞
        if os.path.isabs(filename):
            return False

        # 闃叉 瑙 帇鍒扮郴缁熺洰褰?
        normalized = os.path.normpath(filename)
        if (
            normalized.startswith("..")
            or normalized == "."
            or normalized.startswith("/")
        ):
            return False

        # 鍙 厑璁哥壒瀹氭墿灞曞悕鐨勬枃浠?
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
        """鏌 壘妯 瀷鏂囦欢鐨勫疄闄呬綅缃?"
        possible_paths = []

        # 1. 閰嶇疆鏂囦欢涓 殑璺 緞
        if save_path and exp_name:
            possible_paths.append(os.path.join(save_path, exp_name))

        # 2. 甯歌 鐨勫祵濂楄矾寰?
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

        # 3. 鎼滅储鏁翠釜椤圭洰鐩 綍
        for root, dirs, files in os.walk("."):
            if exp_name in dirs:
                possible_paths.append(os.path.join(root, exp_name))

        # 妫 鏌 矾寰勬槸鍚 瓨鍦?
        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _search_all_model_files(self):
        """鎼滅储鎵 鏈夋 鍨嬫枃浠讹紙safetensors鍜宑onfig.json锛?"
        model_files = []
        model_extensions = [".safetensors", ".json", ".md"]  # 妯 瀷鐩稿叧鏂囦欢鎵 睍鍚?

        # 鎼滅储鏁翠釜椤圭洰鐩 綍
        for root, dirs, files in os.walk("."):
            # 璺宠繃涓 浜涚洰褰?
            if "__pycache__" in root or ".git" in root or "backup_" in root:
                continue

            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in model_extensions:
                    # 妫 鏌 槸鍚 湪妯 瀷鐩稿叧鐩 綍涓?
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
        """娴忚 璁 粌鏁版嵁鏂囦欢"""
        file_path = filedialog.askopenfilename(
            title="閫夋嫨璁 粌鏁版嵁鏂囦欢",
            filetypes=[("CSV鏂囦欢", "*.csv"), ("鎵 鏈夋枃浠?, "*.*")],"
        )
        if file_path:
            self.train_data_path_var.set(file_path)
            self.log_train(f"宸查 夋嫨鏁版嵁鏂囦欢: {file_path}")

    def browse_save_path(self):
        """娴忚 妯 瀷淇濆瓨璺 緞"""
        dir_path = filedialog.askdirectory(title="閫夋嫨妯 瀷淇濆瓨鐩 綍")
        if dir_path:
            self.train_save_path_var.set(dir_path)
            self.log_train(f"宸查 夋嫨淇濆瓨璺 緞: {dir_path}")

    def browse_train_config(self):
        """娴忚 璁 粌閰嶇疆鏂囦欢"""
        file_path = filedialog.askopenfilename(
            title="閫夋嫨璁 粌閰嶇疆鏂囦欢",
            filetypes=[("YAML鏂囦欢", "*.yaml;*.yml"), ("鎵 鏈夋枃浠?, "*.*")],"
        )
        if file_path:
            self.train_config_path_var.set(file_path)
            self.log_train(f"宸查 夋嫨閰嶇疆鏂囦欢: {file_path}")

    def save_train_config(self):
        """淇濆瓨璁 粌閰嶇疆鍒版枃浠?"
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
                title="淇濆瓨璁 粌閰嶇疆",
                defaultextension=".yaml",
                filetypes=[("YAML鏂囦欢", "*.yaml"), ("鎵 鏈夋枃浠?, "*.*")],"
                initialdir=".",
                initialfile="train_config.yaml",
            )

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                self.log_train(f"閰嶇疆宸蹭繚瀛? {file_path}")
                messagebox.showinfo("鎴愬姛", f"閰嶇疆宸蹭繚瀛樺埌: {file_path}")
        except Exception as e:
            self.log_train(f"淇濆瓨閰嶇疆澶辫触: {e}")
            messagebox.showerror("閿欒 ", f"淇濆瓨閰嶇疆澶辫触: {e}")

    def load_train_config(self):
        """浠庢枃浠跺姞杞借 缁冮厤缃?"
        try:
            import yaml

            file_path = filedialog.askopenfilename(
                title="鍔犺浇璁 粌閰嶇疆",
                filetypes=[("YAML鏂囦欢", "*.yaml;*.yml"), ("鎵 鏈夋枃浠?, "*.*")],"
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

            self.log_train(f"閰嶇疆宸插姞杞? {file_path}")
            messagebox.showinfo("鎴愬姛", "閰嶇疆宸插姞杞?)"
        except Exception as e:
            self.log_train(f"鍔犺浇閰嶇疆澶辫触: {e}")
            messagebox.showerror("閿欒 ", f"鍔犺浇閰嶇疆澶辫触: {e}")

    def show_train_help(self):
        """鏄剧 璁 粌閰嶇疆甯 姪鏂囨 """
        help_text = ""
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺?                        Kronos 妯 瀷璁 粌甯 姪鏂囨                               鈺?
鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

銆愪竴銆佸熀妯 瀷閫夋嫨銆?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  Kronos-mini   鈹?鏈 灏忔 鍨嬶紝閫熷害鏈 蹇 紝绮惧害杈冧綆锛岄 傚悎蹇  熸祴璇?
  Kronos-small  鈹?灏忓瀷妯 瀷锛岄 熷害鍜岀簿搴 钩琛 紝鎺 崘鏃 父浣跨敤
  Kronos-base   鈹?涓 瀷妯 瀷锛岀簿搴 渶楂橈紝閫熷害杈冩參锛岄渶瑕佹洿澶氭樉瀛?

銆愪簩銆佹暟鎹 幏鍙栥 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  浜 槗瀵?  鈹?瑕佷笅杞芥暟鎹 殑浜 槗瀵癸紝濡?BTCUSDT銆丒THUSDT 绛?
  鍛 湡     鈹?K绾挎椂闂村懆鏈燂細
           鈹?  1m, 3m, 5m, 15m, 30m (鍒嗛挓)
           鈹?  1h, 2h, 4h, 6h, 8h, 12h (灏忔椂)
           鈹?  1d (澶?
  鏁版嵁閲?  鈹?涓嬭浇鐨凨绾挎潯鏁帮紝寤鸿  500-2000 鏉?

銆愪笁銆佹 鍨嬭 缁冨弬鏁般 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  鍥炵湅绐楀彛  鈹?鐢  灏戞牴鍘嗗彶K绾挎潵棰勬祴鏈 潵
           鈹?寤鸿 鍊硷細256-1024锛屽 艰秺澶  冭檻鐨勫巻鍙茶秺闀?
           鈹?榛樿 鍊硷細512

  棰勬祴绐楀彛  鈹?棰勬祴鏈 潵澶氬皯鏍筀绾?
           鈹?寤鸿 鍊硷細24-96锛屽 艰秺澶  娴嬭秺杩滀絾瓒婁笉鍑?
           鈹?榛樿 鍊硷細48

  鎵规 澶 皬  鈹?姣忔 璁 粌鍚屾椂澶勭悊鐨勬暟鎹 噺
           鈹?鍊艰秺澶  缁冭秺蹇 紝浣嗛渶瑕佹洿澶氭樉瀛?
           鈹?寤鸿 鍊硷細16-64锛岄粯璁  硷細32

  瀛 範鐜?  鈹?妯 瀷鍙傛暟鏇存柊閫熷害锛岃秺灏忚秺鎱 絾瓒婄簿缁?
           鈹?寤鸿 鍊硷細0.00001-0.001
           鈹?榛樿 鍊硷細0.0001

銆愬洓銆侀珮绾 弬鏁般 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  Tokenizer杞 暟 鈹?鍒嗚瘝鍣  缁冭疆鏁?
               鈹?寤鸿 鍊硷細10-50锛岄粯璁  硷細30

  妯 瀷杞 暟    鈹?鍩烘 鍨嬪井璋冭 缁冭疆鏁?
               鈹?寤鸿 鍊硷細10-50锛岄粯璁  硷細20
               鈹?杞 暟瓒婂 瓒婄簿缁嗭紝浣嗚 缁冩椂闂磋秺闀?

  Tokenizer瀛 範鐜?鈹?鍒嗚瘝鍣 笓鐢  涔犵巼
                 鈹?寤鸿 鍊硷細0.0001-0.0005锛岄粯璁  硷細0.0002

  鏃 織闂撮殧    鈹?澶氬皯鎵规 杈撳嚭涓 娆 棩蹇?
               鈹?寤鸿 鍊硷細10-100锛岄粯璁  硷細50

  璁 粌闆嗘瘮渚? 鈹?鏁版嵁闆嗕腑鐢 簬璁 粌鐨勬瘮渚?
               鈹?寤鸿 鍊硷細0.7-0.95锛岄粯璁  硷細0.9

  楠岃瘉闆嗘瘮渚? 鈹?鏁版嵁闆嗕腑鐢 簬楠岃瘉鐨勬瘮渚?
               鈹?寤鸿 鍊硷細0.05-0.3锛岄粯璁  硷細0.1

  鏁版嵁鍔犺浇绾跨  鈹?骞惰 鍔犺浇鏁版嵁鐨勭嚎绋嬫暟
               鈹?寤鸿 鍊硷細4-8锛岄粯璁  硷細6

銆愪簲銆佸疄楠岄厤缃  ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  瀹為獙鍚嶇  鈹?璁 粌妯 瀷鐨勫悕绉帮紝鐢 簬淇濆瓨妯 瀷鏂囦欢澶?
           鈹?寤鸿 浣跨敤鑻辨枃鍜屼笅鍒掔嚎锛屽  btc_5m_v1

  淇濆瓨璺 緞  鈹?妯 瀷淇濆瓨鐨勬牴鐩 綍
           鈹?榛樿 鍊硷細Kronos/finetune_csv/finetuned

  Tokenizer鏂囦欢澶?鈹?鍒嗚瘝鍣 繚瀛樼殑瀛愭枃浠跺 鍚嶇
                 鈹?榛樿 鍊硷細tokenizer

  妯 瀷鏂囦欢澶?  鈹?棰勬祴妯 瀷淇濆瓨鐨勫瓙鏂囦欢澶瑰悕绉?
              鈹?榛樿 鍊硷細basemodel

  鏈 缁堟 鍨嬭矾寰?= {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/{妯 瀷鏂囦欢澶箎/best_model

  璁 粌Tokenizer 鈹?鏄 惁璁 粌鏂扮殑鍒嗚瘝鍣?
               鈹?棣栨 璁 粌寤鸿 鍕鹃 夛紝鍚庣画寰 皟鍙 彇娑?

  璁 粌鍩烘 鍨? 鈹?鏄 惁璁 粌棰勬祴妯 瀷
              鈹?寤鸿 濮嬬粓鍕鹃 ?

  璺宠繃宸插瓨鍦  缁?鈹?濡傛灉妯 瀷宸插瓨鍦 槸鍚 烦杩?
                鈹?鍕鹃 夊悗涓嶄細瑕嗙洊宸叉湁妯 瀷

銆愬叚銆佽 缁冩暟鎹 牸寮忋 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  CSV鏂囦欢蹇呴 鍖呭惈浠 笅鍒楋細
  timestamps, open, high, low, close, volume

  绀轰緥锛?
  timestamps,open,high,low,close,volume
  2024-01-01 00:00:00,42000,42100,41900,42050,1234.56

銆愪竷銆佽 缁冩祦绋嬨 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  1. 閫夋嫨鎴栦笅杞借 缁冩暟鎹?
  2. 閫夋嫨鍩烘 鍨嬶紙鎺 崘 Kronos-small锛?
  3. 璁剧疆璁 粌鍙傛暟锛堝彲浣跨敤榛樿 鍊硷級
  4. 鐐瑰嚮"寮 濮嬭 缁?"
  5. 绛夊緟璁 粌瀹屾垚

銆愬叓銆佹 鍨嬩繚瀛樹綅缃  ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  鏈 缁堜繚瀛樼粨鏋勶細
  {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/
  鈹溾攢鈹  {Tokenizer鏂囦欢澶箎/best_model    (鍒嗚瘝鍣?
  鈹斺攢鈹  {妯 瀷鏂囦欢澶箎/best_model         (棰勬祴妯 瀷锛屼氦鏄撶敤)

  鈹屸攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?
  鈹? 鍒嗚瘝鍣  鍨?                                                            鈹?
  鈹?   {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/{Tokenizer鏂囦欢澶箎/best_model                鈹?
  鈹?                                                                        鈹?
  鈹? 棰勬祴妯 瀷锛堜氦鏄撲娇鐢 級:                                                   鈹?
  鈹?   {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/{妯 瀷鏂囦欢澶箎/best_model                      鈹?
  鈹斺攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?

  渚嬪 锛?
    淇濆瓨璺 緞 = Kronos/finetune_csv/finetuned
    瀹為獙鍚嶇  = btc_5m_v1
    Tokenizer鏂囦欢澶?= tokenizer
    妯 瀷鏂囦欢澶?= basemodel

    鈫?Kronos/finetune_csv/finetuned/btc_5m_v1/tokenizer/best_model
    鈫?Kronos/finetune_csv/finetuned/btc_5m_v1/basemodel/best_model

銆愪節銆佷娇鐢  缁冨 鐨勬 鍨嬨 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  璁 粌瀹屾垚鍚庝細鐢熸垚涓 釜妯 瀷鐩 綍锛?

  鈹屸攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?
  鈹? 1. Tokenizer鐩 綍 (鍒嗚瘝鍣?                                          鈹?
  鈹?    璺 緞: {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/tokenizer/best_model               鈹?
  鈹?    浣滅敤: 灏咾绾挎暟鎹 紪鐮佷负妯 瀷鍙  鐞嗙殑鏍煎紡                            鈹?
  鈹?                                                                    鈹?
  鈹? 2. Basemodel鐩 綍 (棰勬祴妯 瀷)                                        鈹?
  鈹?    璺 緞: {淇濆瓨璺 緞}/{瀹為獙鍚嶇 }/basemodel/best_model               鈹?
  鈹?    浣滅敤: 杩涜 浠锋牸棰勬祴鍜屼氦鏄撲俊鍙风敓鎴?                                 鈹?
  鈹?    (浜 槗鏃跺彧闇 浣跨敤杩欎釜鐩 綍)                                        鈹?
  鈹斺攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?

  浣跨敤鏂规硶锛?
  璁 粌瀹屾垚鍚庯紝鐐瑰嚮"鍒锋柊妯 瀷鍒楄 "鎸夐挳锛岃 缁冨 鐨勬 鍨嬩細鏄剧 鍦  鍨嬮 夋嫨涓嬫媺妗嗕腑锛岀洿鎺  夋嫨鍗冲彲

銆愬崄銆佸父瑙侀棶棰樸 ?
鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
  Q: 璁 粌寰堟參鎬庝箞鍔烇紵
  A: 鍑忓皬鍥炵湅绐楀彛銆侀 娴嬬獥鍙  佹壒娆  灏忥紝浣跨敤鏇村皬鐨勫熀妯 瀷

  Q: 鏄惧瓨涓嶈冻鎬庝箞鍔烇紵
  A: 鍑忓皬鎵规 澶 皬锛屾垨浣跨敤 Kronos-mini 妯 瀷

  Q: 鎵规 澶 皬鍜屾樉瀛樺 搴斿叧绯伙紵
  A:
     鈹屸攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?
     鈹?鎵规 澶 皬   鈹?鏄惧瓨闇 姹? 鈹?閫傚悎鏄惧崱            鈹?
     鈹溾攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹尖攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹尖攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?
     鈹?  16       鈹? ~2GB    鈹?鎵 鏈夋樉鍗?           鈹?
     鈹?  32       鈹? ~4GB    鈹?鍏 棬绾 樉鍗?        鈹?
     鈹?  64       鈹? ~8GB    鈹?涓  鏄惧崱 (GTX 1660) 鈹?
     鈹? 128       鈹? ~16GB   鈹?楂樼 鏄惧崱 (RTX 3080) 鈹?
     鈹? 256       鈹? ~22GB+  鈹?鏃楄埌鏄惧崱 (RTX 4090) 鈹?
     鈹斺攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹粹攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹粹攢鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹 鈹?
     娉 細鏄惧瓨鍗犵敤杩樹笌鍥炵湅绐楀彛銆侀 娴嬬獥鍙  灏忕浉鍏?

  Q: 璁 粌鏁堟灉涓嶅 鎬庝箞鍔烇紵
  A: 澧炲姞璁 粌鏁版嵁閲忥紝澧炲姞璁 粌杞 暟锛岃皟鏁村 涔犵巼

  Q: 濡備綍浣跨敤璁 粌濂界殑妯 瀷锛?
  A: 妯 瀷淇濆瓨鍚庯紝鍦 氦鏄撻潰鏉跨殑妯 瀷閫夋嫨涓  夋嫨"鑷 畾涔夋 鍨?锛岃緭鍏?basemodel/best_model 璺 緞"
""
        help_window = tk.Toplevel(self.root)
        help_window.title("璁 粌甯 姪鏂囨 ")
        help_window.geometry("750x900")

        text_widget = tk.Text(
            help_window, wrap=tk.WORD, bg="#f5f5f5", font=("Consolas", 11)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(help_window, text="鍏抽棴", command=help_window.destroy).pack(pady=10)

    def log_train(self, message):
        """璁 粌鏃 織杈撳嚭"""
        self.train_terminal.insert(tk.END, f"[{self.get_time()}] {message}\n")
        self.train_terminal.see(tk.END)
        # 鍚屾椂鍐欏叆璁 粌鏃 織鏂囦欢
        if hasattr(self, "train_logger"):
            self.train_logger.info(message)

    def _calculate_all_technical_indicators(self, df):
        """璁 畻鎵 鏈?7涓 妧鏈 寚鏍?"
        import numpy as np
        import pandas as pd

        df = df.copy()

        # 鍩虹 浠锋牸鏁版嵁
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # 瀹夊叏鑾峰彇amount鍒楋紙璁 粌鏃剁敤鐨勬槸 amt 鍜?vol锛?
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

        # 娣诲姞璁 粌鏃剁敤鐨?vol 鍒?
        if "vol" not in df.columns:
            if "volume" in df.columns:
                df["vol"] = df["volume"].values
            else:
                df["vol"] = amount

        # 1. MA5, MA10, MA20
        df["MA5"] = pd.Series(close).rolling(window=5).mean().values
        df["MA10"] = pd.Series(close).rolling(window=10).mean().values
        df["MA20"] = pd.Series(close).rolling(window=20).mean().values

        # 2. 涔栫 鐜?
        df["BIAS20"] = (close / df["MA20"] - 1) * 100

        # 3. ATR(14)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        df["ATR14"] = pd.Series(tr).rolling(window=14).mean().values

        # 4. 鎸 箙
        df["AMPLITUDE"] = (high - low) / close * 100

        # 5. 鎴愪氦棰滿A5, MA10
        df["AMOUNT_MA5"] = pd.Series(amount).rolling(window=5).mean().values
        df["AMOUNT_MA10"] = pd.Series(amount).rolling(window=10).mean().values
        df["VOL_RATIO"] = amount / df["AMOUNT_MA5"]

        # 6. RSI(14) 鍜?RSI(7)
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

        # 7. MACD绾?鍜?MACD鏌?
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        # 8. 浠锋牸鏂滅巼(5鏈? 鍜?(10鏈?
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

        # 9. 杩?鏃?10鏃 渶楂?鏈 浣?
        df["HIGH5"] = pd.Series(high).rolling(window=5).max().values
        df["LOW5"] = pd.Series(low).rolling(window=5).min().values
        df["HIGH10"] = pd.Series(high).rolling(window=10).max().values
        df["LOW10"] = pd.Series(low).rolling(window=10).min().values

        # 10. 鎴愪氦閲忕獊鐮?
        df["VOL_BREAKOUT"] = (amount > df["AMOUNT_MA5"] * 1.5).astype(int).values
        df["VOL_SHRINK"] = (amount < df["AMOUNT_MA5"] * 0.5).astype(int).values

        # 閫夋嫨闇 瑕佺殑27涓 壒寰?
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

        # 纭 繚鎵 鏈夊垪閮藉瓨鍦?
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0.0

        # 淇濈暀鏃堕棿鍒?+ 鐗瑰緛鍒楋紝骞跺幓闄 惈 NaN 鐨勮
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
        """涓嬭浇甯佸畨鍘嗗彶鏁版嵁"""
        try:
            symbol = self.train_symbol_var.get()
            timeframe = self.train_timeframe_var.get()
            days = int(self.train_days_var.get() or 30)

            self.log_train(f"寮 濮嬩笅杞?{symbol} {timeframe} 鏈 杩?{days} 澶 暟鎹?..")
            self.train_status_label.config(text="姝 湪涓嬭浇鏁版嵁...")
            self.download_data_button.config(state=tk.DISABLED)

            # 浣跨敤绾跨 涓嬭浇
            import threading

            download_thread = threading.Thread(
                target=self._download_binance_data_thread,
                args=(symbol, timeframe, days),
            )
            download_thread.daemon = True
            download_thread.start()

        except Exception as e:
            self.log_train(f"涓嬭浇澶辫触: {e}")
            self.train_status_label.config(text="涓嬭浇澶辫触")
            self.download_data_button.config(state=tk.NORMAL)

    def _download_binance_data_thread(self, symbol, timeframe, days):
        """涓嬭浇鏁版嵁鐨勭嚎绋嬶紝鎸夊 鏁拌嚜鍔 垎鎵逛笅杞?"
        try:
            from binance_api import BinanceAPI
            import pandas as pd
            import os

            binance = BinanceAPI()

            # 鏍规嵁鏃堕棿鍛 湡璁 畻姣忓 澶氬皯鏍筀绾?
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
                f"鍛 湡 {timeframe} 姣忓 绾?{candles_per_day} 鏍筀绾匡紝鍏遍渶 {total_candles} 鏉?"
            )

            # 妫 鏌 槸鍚 凡鏈夎 甯佺 鍛 湡鐨勬暟鎹 枃浠?
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "training_data"
            )
            os.makedirs(save_dir, exist_ok=True)
            existing_file = os.path.join(save_dir, f"{symbol}_{timeframe}.csv")

            batch_size = 1500  # 甯佸畨姣忔 鏈 澶?500鏉?
            total_batches = (total_candles + batch_size - 1) // batch_size

            self.log_train(
                f"闇 瑕佷笅杞?{total_candles} 鏉 暟鎹 紝灏嗗垎 {total_batches} 鎵规 涓嬭浇..."
            )

            all_dfs = []
            end_time = None

            for batch in range(total_batches):
                self.log_train(f"涓嬭浇绗?{batch + 1}/{total_batches} 鎵规 ...")

                # 鑾峰彇鍘嗗彶鏁版嵁
                df_batch = binance.get_historical_klines(
                    symbol, timeframe, end_str=end_time, limit=batch_size
                )

                if df_batch is None or len(df_batch) == 0:
                    self.log_train(f"绗?{batch + 1} 鎵规 鑾峰彇澶辫触鎴栨棤鏁版嵁")
                    break

                all_dfs.append(df_batch)

                # 鏇存柊缁撴潫鏃堕棿鐢 簬涓嬩竴鎵癸紙鍙栨渶鏃 殑鏃堕棿鎴筹級
                oldest_timestamp = df_batch["timestamps"].min()
                end_time = int(oldest_timestamp.timestamp() * 1000) - 1

                self.log_train(
                    f"绗?{batch + 1} 鎵规 : 鑾峰彇 {len(df_batch)} 鏉 紝鏈 鏃 椂闂? {oldest_timestamp}"
                )

                # 濡傛灉宸茬粡鑾峰彇瓒冲 鏁版嵁灏卞仠姝?
                if sum(len(d) for d in all_dfs) >= total_candles:
                    break

            if len(all_dfs) == 0:
                self.log_train("鏈 兘鑾峰彇浠讳綍鏁版嵁")
                return

            # 鍚堝苟鎵 鏈夋暟鎹?
            df = pd.concat(all_dfs, ignore_index=True)

            # 鍘婚噸骞舵寜鏃堕棿鎺掑簭
            df = (
                df.drop_duplicates(subset=["timestamps"])
                .sort_values("timestamps")
                .reset_index(drop=True)
            )

            # 濡傛灉鏁版嵁瓒呰繃闇 姹傦紝鍙栨渶鏂扮殑
            if len(df) > total_candles:
                df = df.tail(total_candles).reset_index(drop=True)

            if len(df) < 100:
                self.log_train("鑾峰彇鏁版嵁澶辫触鎴栨暟鎹 笉瓒?)"
                return

            # 淇濆瓨涓篊SV锛堜竴涓 竵绉嶅懆鏈熷彧鏈変竴涓 枃浠讹紝澧為噺绱 姞锛?
            save_path = existing_file

            # 濡傛灉宸叉湁鏂囦欢锛屽悎骞跺 閲忔暟鎹?
            if os.path.exists(save_path):
                self.log_train("妫 娴嬪埌宸叉湁鏁版嵁鏂囦欢锛岃繘琛屽 閲忔洿鏂?..")
                df_existing = pd.read_csv(save_path)
                df_existing["timestamps"] = pd.to_datetime(df_existing["timestamps"])

                # 鍚堝苟鏂版棫鏁版嵁
                df = pd.concat([df_existing, df], ignore_index=True)

                # 鍘婚噸骞舵寜鏃堕棿鎺掑簭
                df = (
                    df.drop_duplicates(subset=["timestamps"])
                    .sort_values("timestamps")
                    .reset_index(drop=True)
                )

                old_count = len(df_existing)
                new_count = len(df)
                self.log_train(f"鍘熸湁 {old_count} 鏉 紝鍚堝苟鍚?{new_count} 鏉?)"

            df.to_csv(save_path, index=False)

            self.log_train(f"鍘熷 鏁版嵁宸蹭繚瀛? {save_path}")
            self.log_train(f"鍘熷 鏁版嵁閲? {len(df)} 鏉?)"
            self.log_train(
                f"鏃堕棿鑼冨洿: {df['timestamps'].min()} 鑷?{df['timestamps'].max()}"
            )

            # 鑷 姩璁 畻鎶 鏈 寚鏍囷紝鐢熸垚甯?7涓 壒寰佺殑璁 粌鏂囦欢
            self.log_train("\n姝 湪璁 畻鎶 鏈 寚鏍?..")
            try:
                # 璋冪敤鎴戜滑鐨勬妧鏈 寚鏍囪 绠楀嚱鏁?
                df_with_indicators = self._calculate_all_technical_indicators(df)

                # 鐢熸垚甯 寚鏍囩殑鏂囦欢鍚?
                save_path_with_indicators = save_path.replace(".csv", "_with_indicators.csv")
                df_with_indicators.to_csv(save_path_with_indicators, index=False)

                self.log_train(f"鉁?鎶 鏈 寚鏍囪 绠楀畬鎴愶紒")
                self.log_train(f"鉁?甯 寚鏍囩殑鏁版嵁宸蹭繚瀛? {save_path_with_indicators}")
                self.log_train(f"鉁?鐗瑰緛鏁? {len(df_with_indicators.columns)}")
                self.log_train(f"鉁?鏈夋晥鏁版嵁閲? {len(df_with_indicators)} 鏉?)"

                # 鑷 姩璁剧疆璁 粌鏁版嵁璺 緞涓哄甫鎸囨爣鐨勬枃浠?
                self.train_data_path_var.set(save_path_with_indicators)
                self.train_status_label.config(text=f"鏁版嵁鍑嗗 瀹屾垚: {len(df_with_indicators)}鏉?鍚 寚鏍?")
            except Exception as e:
                self.log_train(f"鎶 鏈 寚鏍囪 绠楀 璐? {e}")
                import traceback
                self.log_train(traceback.format_exc())
                # 濡傛灉鎶 鏈 寚鏍囪 绠楀 璐 紝灏辩敤鍘熷 鏁版嵁
                self.train_data_path_var.set(save_path)
                self.train_status_label.config(text=f"鏁版嵁涓嬭浇瀹屾垚: {len(df)}鏉?)"

        except Exception as e:
            self.log_train(f"涓嬭浇寮傚父: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_status_label.config(text="涓嬭浇寮傚父")
        finally:
            self.download_data_button.config(state=tk.NORMAL)

    def start_training(self):
        """寮 濮嬭 缁?"
        try:
            data_path = self.train_data_path_var.get().strip()
            config_path = self.train_config_path_var.get().strip()

            if not data_path:
                messagebox.showwarning("璀 憡", "璇峰厛閫夋嫨鏁版嵁鏂囦欢锛?)"
                return

            # 杞 崲涓虹粷瀵硅矾寰?
            if not os.path.isabs(data_path):
                data_path = os.path.abspath(data_path)

            if not os.path.exists(data_path):
                messagebox.showerror("閿欒 ", f"鏁版嵁鏂囦欢涓嶅瓨鍦? {data_path}")
                return

            if not os.path.exists(config_path):
                messagebox.showerror("閿欒 ", f"閰嶇疆鏂囦欢涓嶅瓨鍦? {config_path}")
                return

            # 妫 鏌 暟鎹 噺鏄 惁瓒冲
            import pandas as pd

            try:
                df_temp = pd.read_csv(data_path)
                total_rows = len(df_temp)
                lookback = int(self.train_lookback_var.get())
                predict = int(self.train_predict_var.get())
                window = lookback + predict + 1  # 涓庢暟鎹 泦璁 畻鏂瑰紡涓 鑷?

                train_ratio = float(self.train_ratio_var.get())
                val_ratio = float(self.val_ratio_var.get())

                train_rows = int(total_rows * train_ratio)
                val_rows = int(total_rows * val_ratio)

                train_samples = train_rows - window + 1
                val_samples = val_rows - window + 1

                min_required = int(window * 1.5)

                if total_rows < min_required:
                    messagebox.showerror(
                        "閿欒 ",
                        f"鏁版嵁閲忎笉瓒筹紒褰撳墠: {total_rows} 琛岋紝闇 瑕佽嚦灏?{min_required} 琛孿n寤鸿 锛氬噺灏忓洖鐪嬬獥鍙 拰棰勬祴绐楀彛锛屾垨涓嬭浇鏇村 鏁版嵁",
                    )
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_train_button.config(state=tk.DISABLED)
                    return

                if val_samples <= 0:
                    messagebox.showerror(
                        "閿欒 ",
                        f"楠岃瘉闆嗘暟鎹 笉瓒筹紒\n褰撳墠楠岃瘉闆? {val_rows} 琛岋紝绐楀彛: {window}\n闇 瑕侀獙璇侀泦鑷冲皯鏈?{window} 琛孿n\n寤鸿 锛氬噺灏忛獙璇侀泦姣斾緥鎴栦笅杞芥洿澶氭暟鎹?,"
                    )
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_train_button.config(state=tk.DISABLED)
                    return

                self.log_train(
                    f"鏁版嵁閲忔 鏌? 鎬昏 {total_rows}琛? 璁 粌{train_rows}琛?{train_samples}鏍锋湰), 楠岃瘉{val_rows}琛?{val_samples}鏍锋湰), 绐楀彛{window}"
                )
            except Exception as e:
                self.log_train(f"鏁版嵁妫 鏌  鍛? {e}")

            # 鍒涘缓鍋滄 浜嬩欢
            self.train_stop_event = threading.Event()

            # 绂佺敤鎸夐挳
            self.train_button.config(state=tk.DISABLED)
            self.stop_train_button.config(state=tk.NORMAL)
            self.train_status_label.config(text="姝 湪鍑嗗 璁 粌...")
            self.train_progress_var.set(0)

            # 娓呯 杈撳嚭
            self.train_terminal.delete(1.0, tk.END)

            self.log_train("=" * 60)
            self.log_train("寮 濮嬭 缁?)"
            self.log_train("=" * 60)
            self.log_train(f"鍩烘 鍨? {self.train_base_model_var.get()}")
            self.log_train(f"鏁版嵁鏂囦欢: {data_path}")
            self.log_train(
                f"鍥炵湅绐楀彛: {self.train_lookback_var.get()}, 棰勬祴绐楀彛: {self.train_predict_var.get()}"
            )
            self.log_train(
                f"鎵规 澶 皬: {self.train_batch_var.get()}, 瀛 範鐜? {self.train_lr_var.get()}"
            )
            self.log_train(
                f"Tokenizer杞 暟: {self.train_tokenizer_epochs_var.get()}, 妯 瀷杞 暟: {self.train_basemodel_epochs_var.get()}"
            )
            self.log_train(
                f"璁 粌闆? {self.train_ratio_var.get()}, 楠岃瘉闆? {self.val_ratio_var.get()}"
            )
            self.log_train(f"鏁版嵁璺 緞: {data_path}")
            self.log_train(f"瀹為獙鍚嶇 : {self.train_exp_name_var.get()}")

            # 鍚 姩璁 粌绾跨
            self.train_thread = threading.Thread(
                target=self._training_thread,
                args=(data_path, config_path, self.train_base_model_var.get()),
            )
            self.train_thread.daemon = True
            self.train_thread.start()

        except Exception as e:
            self.log_train(f"鍚 姩璁 粌澶辫触: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

    def _training_thread(self, data_path, config_path, base_model):
        """璁 粌绾跨 """
        try:
            import os
            import sys
            import yaml
            import tempfile

            # 璇诲彇閰嶇疆鏂囦欢
            self.log_train("姝 湪璇诲彇閰嶇疆鏂囦欢...")
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 鍒濆 鍖栭厤缃 粨鏋勶紙濡傛灉涓嶅瓨鍦 級
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

            # 淇 敼閰嶇疆
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
            config["training"]["accumulation_steps"] = 1  # 姊 害绱  姝 暟

            # 瀹為獙閰嶇疆
            if "experiment" not in config:
                config["experiment"] = {}
            config["experiment"]["train_tokenizer"] = self.train_tokenizer_var.get()
            config["experiment"][
                "train_basemodel"
            ] = self.train_basemodel_check_var.get()
            config["experiment"]["skip_existing"] = self.train_skip_existing_var.get()

            # 璁剧疆妯 瀷璺 緞鍜屽疄楠屽悕绉?
            if "model_paths" not in config:
                config["model_paths"] = {}
            config["model_paths"]["exp_name"] = self.train_exp_name_var.get()

            # 璁剧疆妯 瀷淇濆瓨璺 緞
            save_path = self.train_save_path_var.get()
            exp_name = self.train_exp_name_var.get()
            tokenizer_folder = self.train_tokenizer_folder_var.get()
            basemodel_folder = self.train_basemodel_folder_var.get()
            config["model_paths"]["base_save_path"] = f"{save_path}/{exp_name}"
            config["model_paths"][
                "finetuned_tokenizer"
            ] = f"{save_path}/{exp_name}/{tokenizer_folder}/best_model"
            config["model_paths"]["basemodel_save_name"] = basemodel_folder

            # 璁剧疆鍩烘 鍨?
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
            self.log_train(f"浣跨敤鍩烘 鍨? {model_name}")

            # 娣诲姞GPU閰嶇疆
            use_gpu = "GPU" in self.train_device_var.get()
            if "device" not in config:
                config["device"] = {}
            config["device"]["use_cuda"] = use_gpu
            config["device"]["device_id"] = 0
            self.log_train(f"璁惧 : {'GPU (CUDA)' if use_gpu else 'CPU'}")

            # 淇濆瓨涓存椂閰嶇疆鏂囦欢
            temp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            )
            yaml.dump(config, temp_config, allow_unicode=True)
            temp_config.close()

            self.log_train(f"涓存椂閰嶇疆鏂囦欢: {temp_config.name}")

            # 娓呯悊GPU鍐呭瓨
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.log_train("宸叉竻鐞咷PU缂撳瓨")
            except:
                pass

            # 娓呯悊宸插姞杞界殑Kronos妯 瀷锛堥噴鏀綠PU鍐呭瓨锛?
            try:
                if hasattr(self, "strategy") and self.strategy:
                    # 灏濊瘯娓呯悊绛栫暐涓 殑妯 瀷
                    if hasattr(self.strategy, "analyzer"):
                        self.strategy.analyzer = None
                    self.strategy = None
                    self.log_train("宸叉竻鐞嗕氦鏄撴 鍨嬬紦瀛?)"
            except:
                pass

            # 鍒囨崲鍒拌 缁冪洰褰?
            train_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Kronos", "finetune_csv"
            )
            os.chdir(train_dir)

            self.log_train(f"鍒囨崲鍒拌 缁冪洰褰? {train_dir}")

            # 瀵煎叆璁 粌鑴氭湰
            sys.path.insert(0, train_dir)

            # 鑾峰彇璁 粌鍙傛暟
            tokenizer_train = self.train_tokenizer_var.get()
            basemodel_train = self.train_basemodel_check_var.get()

            self.log_train("=" * 60)
            self.log_train("寮 濮嬬湡姝 殑妯 瀷璁 粌...")
            self.log_train(f"璁 粌Tokenizer: {tokenizer_train}")
            self.log_train(f"璁 粌鍩烘 鍨? {basemodel_train}")
            self.log_train("=" * 60)

            # 璋冪敤鐪熸 鐨勮 缁冨嚱鏁帮紙鍦 瓙绾跨 涓 繍琛岋紝鏃 織浼氭樉绀哄湪璁 粌闈 澘锛?
            sys.path.insert(0, train_dir)
            from train_sequential import SequentialTrainer

            trainer = SequentialTrainer(temp_config.name)

            if tokenizer_train:
                self.train_status_label.config(text="姝 湪璁 粌Tokenizer...")
                self.train_progress_var.set(10)
                self.log_train("\n寮 濮嬭 缁僒okenizer...")

                self.log_train(f"閰嶇疆妫 鏌?")
                self.log_train(f"  - data_path: {config['data']['data_path']}")
                self.log_train(
                    f"  - lookback: {config['data']['lookback_window']}, predict: {config['data']['predict_window']}"
                )
                self.log_train(
                    f"  - train_ratio: {config['data']['train_ratio']}, val_ratio: {config['data']['val_ratio']}"
                )
                self.log_train("璁 粌涓?..")

                trainer.train_tokenizer_phase()
                self.train_progress_var.set(50)

            if basemodel_train:
                self.train_status_label.config(text="姝 湪璁 粌棰勬祴妯 瀷...")
                self.log_train("\n寮 濮嬭 缁働redictor...")
                self.log_train("璁 粌涓?..")
                trainer.train_basemodel_phase()

            self.train_progress_var.set(100)
            self.train_status_label.config(text="璁 粌瀹屾垚锛?)"
            self.log_train("\n" + "=" * 60)
            self.log_train("璁 粌瀹屾垚锛?)"
            self.log_train(f"妯 瀷淇濆瓨璺 緞: {save_path}/{exp_name}")
            self.log_train("=" * 60)

            # 鎭  鎸夐挳鐘舵 侊紙涓嶉樆濉濭UI锛?
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

        except Exception as e:
            self.log_train(f"鍚 姩璁 粌澶辫触: {e}")
            import traceback

            self.log_train(traceback.format_exc())
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)

    def stop_training(self):
        """鍋滄 璁 粌"""
        if self.train_stop_event:
            self.train_stop_event.set()
            self.log_train("姝 湪鍋滄 璁 粌...")
            self.train_status_label.config(text="姝 湪鍋滄 ...")

    def on_closing(self):
        # 鍙栨秷鏂伴椈鑷 姩鍒锋柊浠诲姟
        if self.news_auto_refresh_job:
            self.root.after_cancel(self.news_auto_refresh_job)
        self._stop_balance_recording()

        # 娓呯悊淇 彿鏂囦欢
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            signal_file = os.path.join(base_dir, "_splash_close.txt")
            if os.path.exists(signal_file):
                os.remove(signal_file)
        except Exception as e:
            print(f"娓呯悊淇 彿鏂囦欢澶辫触: {e}")

        if self.is_running:
            if messagebox.askokcancel("閫 鍑?, "浜 槗姝 湪杩愯 锛岀 瀹氳 閫 鍑哄悧锛?):
                self.stop_event.set()
                self.is_running = False
                self.root.destroy()
        else:
            self.root.destroy()

    def _create_ai_trading_tab(self, parent):
        main_frame = ttk.Frame(parent, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="馃  AI瀹炵洏绛栫暐", font=("寰 蒋闆呴粦", 18, "bold"), foreground="#2c3e50")
        title_label.pack(pady=(0, 8))

        desc_label = ttk.Label(main_frame, text="瀹炵洏绛栫暐閰嶇疆涓庣 鐞?, font=("寰 蒋闆呴粦", 10), foreground="#7f8c8d")"
        desc_label.pack(pady=(0, 15))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(button_frame, text="馃搵 浜 槗椋庢牸棰勮 :").pack(side=tk.LEFT, padx=(0, 5))
        self.ai_strategy_preset_var = tk.StringVar(value="骞宠 鍨?)"
        preset_combo = ttk.Combobox(
            button_frame,
            textvariable=self.ai_strategy_preset_var,
            values=["婵 杩涜秴鐭 嚎", "瓒嬪娍杩借釜", "骞宠 鍨?, "闇囪崱濂楀埄", "绋冲仴闀跨嚎", "娑堟伅椹卞姩"],"
            state="readonly",
            width=15
        )
        preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._on_ai_strategy_preset_changed)

        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="馃搧 杞藉叆閰嶇疆", command=self._load_ai_strategy_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="馃捑 淇濆瓨閰嶇疆", command=self._save_ai_strategy_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="鈫 笍 閲嶇疆榛樿 ", command=self._reset_ai_strategy_default).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(button_frame, text="鉁?搴旂敤閰嶇疆", command=self._apply_ai_strategy_config, style="Accent.TButton").pack(side=tk.RIGHT, padx=(5, 0))

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

        # 涓簊trategy鍒嗙被鐨勫弬鏁版坊鍔爐race鍥炶皟锛屽悓姝 洿鏂版棫鍙橀噺
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
        self.ai_strategy_notebook.add(frame, text="1. 鍗忚皟鍣 弬鏁?)"

        ttk.Label(frame, text="銆愬崗璋冨櫒鍙傛暟銆?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "coordinator", "min_signal_strength", "鏈 灏忎俊鍙峰己搴?, "float", 0.0, 1.0, "鍙 湁淇 彿寮哄害瓒呰繃姝  兼墠瑙 彂浜 槗")"
        self._create_ai_strategy_param_row(frame, "coordinator", "max_position_size", "鏈 澶 粨浣嶆瘮渚?, "float", 0.0, 1.0, "鍗曟 浜 槗鐨勬渶澶 粨浣嶆瘮渚?)
        self._create_ai_strategy_param_row(frame, "coordinator", "sentiment_weight", "鑸嗘儏淇 彿鏉冮噸", "float", 0.0, 1.0, "FinGPT鑸嗘儏鍒嗘瀽鐨勬潈閲?)"
        self._create_ai_strategy_param_row(frame, "coordinator", "technical_weight", "鎶 鏈 俊鍙锋潈閲?, "float", 0.0, 1.0, "Kronos鎶 鏈 垎鏋愮殑鏉冮噸")"
        self._create_ai_strategy_param_row(frame, "coordinator", "black_swan_threshold", "榛戝 楣呴槇鍊?, "select", None, None, "鏋佺 琛屾儏鏁忔劅搴 槇鍊?)
        self._create_ai_strategy_param_row(frame, "coordinator", "enable_adaptive_filtering", "鑷  傚簲杩囨护", "bool", None, None, "鍚 敤鍔  佸弬鏁拌皟鏁?)"

    def _create_ai_strategy_basic_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="2. 鍩虹 鍙傛暟")

        ttk.Label(frame, text="銆愬熀纭 鍙傛暟銆?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "basic", "POSITION_MULTIPLIER", "浠撲綅鍊嶆暟", "float", 0.1, 10, "浜 槗浠撲綅澶 皬鍊嶆暟锛屾渶灏忎粨浣?00璧?)"
        self._create_ai_strategy_param_row(frame, "basic", "TREND_STRENGTH_THRESHOLD", "瓒嬪娍寮哄害闃堝 ?, "float", 0.001, 0.05, "鍒 柇瓒嬪娍鏈夋晥鐨勯槇鍊?)
        self._create_ai_strategy_param_row(frame, "basic", "LOOKBACK_PERIOD", "鍥炵湅K绾挎暟閲?, "int", 64, 2048, "鎶 鏈 垎鏋愪娇鐢 殑鍘嗗彶鏁版嵁闀垮害")"
        self._create_ai_strategy_param_row(frame, "basic", "PREDICTION_LENGTH", "棰勬祴K绾挎暟閲?, "int", 4, 200, "Kronos棰勬祴鐨勬湭鏉 绾挎暟閲?)
        self._create_ai_strategy_param_row(frame, "basic", "CHECK_INTERVAL", "妫 鏌 棿闅?绉?", "int", 30, 3600, "绯荤粺妫 鏌 氦鏄撲俊鍙风殑闂撮殧")

    def _create_ai_strategy_entry_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="3. 鍏 満杩囨护")

        ttk.Label(frame, text="銆愬叆鍦鸿繃婊 弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "entry", "max_kline_change", "鏈 澶 崟K鍙樺寲", "float", 0.001, 0.05, "闄愬埗鍗曟牴K绾跨殑鏈 澶 定璺屽箙")
        self._create_ai_strategy_param_row(frame, "entry", "max_funding_rate_long", "澶氬 鏈 澶 祫閲戣垂鐜?, "float", -0.05, 0.05, "寮 澶氫粨鏃惰祫閲戣垂鐜囦笂闄?)
        self._create_ai_strategy_param_row(frame, "entry", "min_funding_rate_short", "绌哄 鏈 灏忚祫閲戣垂鐜?, "float", -0.05, 0.05, "寮 绌轰粨鏃惰祫閲戣垂鐜囦笅闄?)
        self._create_ai_strategy_param_row(frame, "entry", "support_buffer", "鏀 拺浣嶇紦鍐?, "float", 1.000, 1.010, "鍦 敮鎾戜綅涓婃柟澶氬皯姣斾緥鍏 満")"
        self._create_ai_strategy_param_row(frame, "entry", "resistance_buffer", "闃诲姏浣嶇紦鍐?, "float", 0.990, 1.000, "鍦 樆鍔涗綅涓嬫柟澶氬皯姣斾緥鍏 満")"

    def _create_ai_strategy_stop_loss_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="4. 姝 崯鍙傛暟")

        ttk.Label(frame, text="銆愭 鎹熷弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "stop_loss", "long_buffer", "澶氬 姝 崯缂撳啿", "float", 0.900, 0.999, "澶氬 姝 崯鐩稿 浜庡叆鍦轰环鐨勬瘮渚?)"
        self._create_ai_strategy_param_row(frame, "stop_loss", "short_buffer", "绌哄 姝 崯缂撳啿", "float", 1.001, 1.100, "绌哄 姝 崯鐩稿 浜庡叆鍦轰环鐨勬瘮渚?)"

    def _create_ai_strategy_take_profit_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="5. 姝 泩鍙傛暟")

        ttk.Label(frame, text="銆愭 鐩堝弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_multiplier_long", "澶氬 绗 竴姝 泩", "float", 1.001, 1.100, "澶氬 绗 竴姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_multiplier_long", "澶氬 绗 簩姝 泩", "float", 1.001, 1.200, "澶氬 绗 簩姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_multiplier_long", "澶氬 绗 笁姝 泩", "float", 1.001, 1.300, "澶氬 绗 笁姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_multiplier_short", "绌哄 绗 竴姝 泩", "float", 0.900, 0.999, "绌哄 绗 竴姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_multiplier_short", "绌哄 绗 簩姝 泩", "float", 0.800, 0.999, "绌哄 绗 簩姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_multiplier_short", "绌哄 绗 笁姝 泩", "float", 0.700, 0.999, "绌哄 绗 笁姝 泩鐩 爣")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp1_position_ratio", "绗 竴姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 竴姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp2_position_ratio", "绗 簩姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 簩姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")
        self._create_ai_strategy_param_row(frame, "take_profit", "tp3_position_ratio", "绗 笁姝 泩浠撲綅", "float", 0.1, 1.0, "杈惧埌绗 笁姝 泩鏃跺钩鎺夌殑浠撲綅姣斾緥")

    def _create_ai_strategy_risk_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="6. 椋庨櫓绠 悊")

        ttk.Label(frame, text="銆愰 闄  鐞嗗弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "risk", "single_trade_risk", "鍗曠瑪椋庨櫓姣斾緥", "float", 0.001, 0.10, "鍗曠瑪浜 槗鏈 澶 簭鎹熸瘮渚?)"
        self._create_ai_strategy_param_row(frame, "risk", "daily_loss_limit", "姣忔棩浜忔崯闄愬埗", "float", 0.01, 0.20, "鍗曟棩绱  浜忔崯杈惧埌姝  煎仠姝 氦鏄?)"
        self._create_ai_strategy_param_row(frame, "risk", "max_consecutive_losses", "鏈 澶 繛缁 簭鎹?, "int", 1, 10, "杩炵画浜忔崯娆 暟杈惧埌姝  兼殏鍋滀氦鏄?)
        self._create_ai_strategy_param_row(frame, "risk", "max_single_position", "鏈 澶 崟绗斾粨浣?, "float", 0.01, 1.0, "鍗曠瑪浜 槗鐨勬渶澶 粨浣嶆瘮渚?)
        self._create_ai_strategy_param_row(frame, "risk", "max_daily_position", "鏈 澶 棩浠撲綅", "float", 0.01, 1.0, "鍗曟棩绱  寮 浠撶殑鏈 澶 粨浣嶆瘮渚?)"

    def _create_ai_strategy_frequency_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="7. 浜 槗棰戠巼")

        ttk.Label(frame, text="銆愪氦鏄撻 鐜囧弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "frequency", "max_daily_trades", "姣忔棩鏈 澶 氦鏄?, "int", 1, 100, "闄愬埗鍗曟棩鏈 澶氫氦鏄撳 灏戞 ")"
        self._create_ai_strategy_param_row(frame, "frequency", "min_trade_interval_minutes", "鏈 灏忛棿闅?鍒?", "int", 1, 60, "涓  浜 槗涔嬮棿鐨勬渶灏忛棿闅?)"
        self._create_ai_strategy_param_row(frame, "frequency", "active_hours_start", "娲昏穬寮 濮嬫椂闂?, "int", 0, 23, "鍙 湪姝 椂闂翠箣鍚庤繘琛屼氦鏄?)
        self._create_ai_strategy_param_row(frame, "frequency", "active_hours_end", "娲昏穬缁撴潫鏃堕棿", "int", 1, 24, "鍙 湪姝 椂闂翠箣鍓嶈繘琛屼氦鏄?)"

    def _create_ai_strategy_position_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="8. 浠撲綅绠 悊")

        ttk.Label(frame, text="銆愪粨浣嶇 鐞嗗弬鏁般 ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "position", "initial_entry_ratio", "鍒濆 鍏 満姣斾緥", "float", 0.1, 1.0, "棣栨 寮 浠撴椂浣跨敤鐩 爣浠撲綅鐨勬瘮渚?)"
        self._create_ai_strategy_param_row(frame, "position", "confirm_interval_kline", "纭  K绾挎暟閲?, "int", 1, 10, "鍒濆 鍏 満鍚庣瓑寰呭 灏戞牴K绾跨 璁 秼鍔?)
        self._create_ai_strategy_param_row(frame, "position", "add_on_profit", "鐩堝埄鍔犱粨", "bool", None, None, "鏄 惁鍦 泩鍒 椂鍔犱粨")
        self._create_ai_strategy_param_row(frame, "position", "add_ratio", "鍔犱粨姣斾緥", "float", 0.1, 0.5, "姣忔 鍔犱粨鍗犵洰鏍囦粨浣嶇殑姣斾緥")
        self._create_ai_strategy_param_row(frame, "position", "max_add_times", "鏈 澶 姞浠撴 鏁?, "int", 1, 10, "鏈 澶氬彲浠 姞浠撳 灏戞 ")"

    def _create_ai_strategy_strategy_tab(self):
        frame = ttk.Frame(self.ai_strategy_notebook, padding="15")
        self.ai_strategy_notebook.add(frame, text="9. 绛栫暐鍙傛暟閰嶇疆")

        ttk.Label(frame, text="銆愮瓥鐣 弬鏁伴厤缃  ?, font=("寰 蒋闆呴粦", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))"

        self._create_ai_strategy_param_row(frame, "strategy", "entry_confirm_count", "寮 浠撶 璁  鏁?, "int", 1, 10, "寮 浠撲俊鍙烽渶瑕佺 璁 殑娆 暟")"
        self._create_ai_strategy_param_row(frame, "strategy", "reverse_confirm_count", "骞充粨纭  娆 暟", "int", 1, 10, "骞充粨淇 彿闇 瑕佺 璁 殑娆 暟")
        self._create_ai_strategy_param_row(frame, "strategy", "require_consecutive_prediction", "杩炵画棰勬祴纭  ", "int", 1, 10, "闇 瑕佽繛缁  灏戞 棰勬祴涓 鑷存墠鎵  ")
        self._create_ai_strategy_param_row(frame, "strategy", "post_entry_hours", "寮 浠撳悗璁 椂(灏忔椂)骞充粨", "float", 0.5, 24, "寮 浠撳悗澶氶暱鏃堕棿鑷 姩骞充粨")
        self._create_ai_strategy_param_row(frame, "strategy", "take_profit_min_pct", "鏈 灏忔 鐩?%)", "float", 0.1, 10, "鏈 灏忕殑姝 泩姣斾緥")

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
        messagebox.showinfo("鎴愬姛", "绛栫暐閰嶇疆宸叉垚鍔熷簲鐢 紒")

    def _load_ai_strategy_config(self):
        file_path = filedialog.askopenfilename(
            title="杞藉叆閰嶇疆",
            filetypes=[("YAML鏂囦欢", "*.yaml"), ("YAML鏂囦欢", "*.yml"), ("鎵 鏈夋枃浠?, "*.*")]"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    self._load_ai_strategy_config_dict(config)
                messagebox.showinfo("鎴愬姛", "閰嶇疆宸茶浇鍏 紒")
            except Exception as e:
                messagebox.showerror("閿欒 ", f"杞藉叆閰嶇疆澶辫触: {e}")

    def _save_ai_strategy_config(self):
        config = self._get_ai_strategy_config_from_ui()
        file_path = filedialog.asksaveasfilename(
            title="淇濆瓨閰嶇疆",
            defaultextension=".yaml",
            filetypes=[("YAML鏂囦欢", "*.yaml"), ("YAML鏂囦欢", "*.yml"), ("鎵 鏈夋枃浠?, "*.*")]"
        )
        if file_path:
            try:
                import yaml
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                messagebox.showinfo("鎴愬姛", "閰嶇疆宸蹭繚瀛橈紒")
            except Exception as e:
                messagebox.showerror("閿欒 ", f"淇濆瓨閰嶇疆澶辫触: {e}")

    def _reset_ai_strategy_default(self):
        if messagebox.askyesno("纭  ", "纭 畾瑕侀噸缃 负榛樿 閰嶇疆鍚楋紵"):
            default_config = self._get_default_ai_strategy_config()
            self._load_ai_strategy_config_dict(default_config)

    def _load_ai_strategy_config_dict(self, config):
        for category, params in config.items():
            for param_name, value in params.items():
                var_key = f"{category}.{param_name}"
                if var_key in self.ai_strategy_config_vars:
                    self.ai_strategy_config_vars[var_key].set(value)

                    # 鍚屾 鍒版棫鐨勭嫭绔嬪彉閲忥紝淇濇寔涓 鑷存 ?
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
            "婵 杩涜秴鐭 嚎": {
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
            "瓒嬪娍杩借釜": {
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
            "骞宠 鍨?: {"
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
            "闇囪崱濂楀埄": {
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
            "绋冲仴闀跨嚎": {
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
            "娑堟伅椹卞姩": {
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
        return presets.get(preset_name, presets["骞宠 鍨?])"

    def _apply_ai_strategy_preset(self, preset_name):
        preset_config = self._get_ai_strategy_preset_config(preset_name)
        current_config = self._get_ai_strategy_config_from_ui()

        for category, params in preset_config.items():
            if category in current_config:
                current_config[category].update(params)

        self._load_ai_strategy_config_dict(current_config)

    def _create_ai_strategy_tab(self, parent):
        """鍒涘缓AI绛栫暐涓 績鏍囩 椤?- 绠 鍖栫増锛屼粎淇濈暀鑷 姩浼樺寲绯荤粺"""
        # 涓绘 鏋?
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 鏍囬
        title_label = ttk.Label(
            main_frame,
            text="馃  AI绛栫暐涓 績",
            font=("寰 蒋闆呴粦", 20, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 5))

        # 璇存槑鏂囧瓧
        desc_label = ttk.Label(
            main_frame,
            text="AI瀹氭湡鍒嗘瀽甯傚満锛岃嚜鍔 紭鍖栨墍鏈変氦鏄撳弬鏁帮紙寮 浠撱 佸钩浠撱 佹 鐩堟 鎹熺瓑锛?,"
            font=("寰 蒋闆呴粦", 11),
            foreground="#7f8c8d"
        )
        desc_label.pack(pady=(0, 20))

        # ==================== 绯荤粺鐘舵 侀潰鏉?====================
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 20))

        # FinGPT鐘舵 ?
        fingpt_status_frame = ttk.Frame(status_frame)
        fingpt_status_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(fingpt_status_frame, text="FinGPT:", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.fingpt_status_label = ttk.Label(
            fingpt_status_frame,
            textvariable=self.fingpt_status_var,
            font=("寰 蒋闆呴粦", 10, "bold"),
            foreground="#27ae60"
        )
        self.fingpt_status_label.pack(side=tk.LEFT)

        # 绛栫暐鍗忚皟鍣 姸鎬?
        coordinator_status_frame = ttk.Frame(status_frame)
        coordinator_status_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(coordinator_status_frame, text="鍗忚皟鍣?", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.coordinator_status_label = ttk.Label(
            coordinator_status_frame,
            textvariable=self.coordinator_status_var,
            font=("寰 蒋闆呴粦", 10, "bold"),
            foreground="#27ae60"
        )
        self.coordinator_status_label.pack(side=tk.LEFT)

        # ==================== AI绛栫暐涓 績鎺 埗闈 澘 ====================
        scheduler_frame = ttk.LabelFrame(main_frame, text="馃  AI绛栫暐涓 績 - 鑷 姩浼樺寲绯荤粺", padding="20")
        scheduler_frame.pack(fill=tk.BOTH, expand=True)

        # 璋冨害鍣 姸鎬佸拰鎺 埗
        scheduler_control_row = ttk.Frame(scheduler_frame)
        scheduler_control_row.pack(fill=tk.X, pady=(0, 15))

        # 宸 晶锛氱姸鎬佹樉绀?
        status_left = ttk.Frame(scheduler_control_row)
        status_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(status_left, text="璋冨害鍣 姸鎬?", font=("寰 蒋闆呴粦", 10)).pack(anchor=tk.W)
        self.ai_scheduler_status_var = tk.StringVar(value="鏈 惎鍔?)"
        scheduler_status_label = ttk.Label(
            status_left,
            textvariable=self.ai_scheduler_status_var,
            font=("寰 蒋闆呴粦", 11, "bold"),
            foreground="#e74c3c"
        )
        scheduler_status_label.pack(anchor=tk.W)

        ttk.Label(status_left, text="涓婃 浼樺寲:", font=("寰 蒋闆呴粦", 10)).pack(anchor=tk.W, pady=(8, 0))
        self.last_optimization_var = tk.StringVar(value="--:--:--")
        ttk.Label(
            status_left,
            textvariable=self.last_optimization_var,
            font=("寰 蒋闆呴粦", 10)
        ).pack(anchor=tk.W)

        # 涓 棿锛氭帶鍒舵寜閽?
        control_middle = ttk.Frame(scheduler_control_row)
        control_middle.pack(side=tk.LEFT, padx=20)

        self.start_scheduler_button = ttk.Button(
            control_middle,
            text="鈻?鍚 姩鑷 姩浼樺寲",
            command=self._start_ai_scheduler,
            style="Accent.TButton"
        )
        self.start_scheduler_button.pack(pady=(0, 8))

        self.stop_scheduler_button = ttk.Button(
            control_middle,
            text="鈴?鍋滄 鑷 姩浼樺寲",
            command=self._stop_ai_scheduler,
            state=tk.DISABLED
        )
        self.stop_scheduler_button.pack()

        # 鍙充晶锛氱珛鍗充紭鍖栧拰璁剧疆
        control_right = ttk.Frame(scheduler_control_row)
        control_right.pack(side=tk.RIGHT)

        ttk.Button(
            control_right,
            text="鈿欙笍 鍙傛暟閰嶇疆",
            command=self._open_strategy_config
        ).pack(pady=(0, 8))

        ttk.Button(
            control_right,
            text="鈿?绔嬪嵆浼樺寲",
            command=self._trigger_optimization_now
        ).pack(pady=(0, 8))

        # 妫 鏌 棿闅旇 缃?
        interval_frame = ttk.Frame(control_right)
        interval_frame.pack()
        ttk.Label(interval_frame, text="妫 鏌 棿闅?鍒嗛挓):", font=("寰 蒋闆呴粦", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.optimization_interval_var = tk.IntVar(value=60)
        interval_spinbox = ttk.Spinbox(
            interval_frame,
            from_=15,
            to=1440,
            textvariable=self.optimization_interval_var,
            width=8,
            font=("寰 蒋闆呴粦", 9)
        )
        interval_spinbox.pack(side=tk.LEFT)

        # 浼樺寲鍙傛暟璇存槑
        params_info_frame = ttk.LabelFrame(scheduler_frame, text="馃搳 AI浼樺寲鍙傛暟璇存槑", padding="10")
        params_info_frame.pack(fill=tk.X, pady=(15, 0))

        params_info_text = """AI绛栫暐涓 績浼氬畾鏈熶紭鍖栦互涓?澶 被鍏?0+浜 槗鍙傛暟锛?"

銆?. 鍗忚皟鍣 弬鏁般 ?
   鈥?min_signal_strength: 鏈 灏忎俊鍙峰己搴?(0.0-1.0)
   鈥?max_position_size: 鏈 澶 粨浣嶆瘮渚?(0.0-1.0)
   鈥?sentiment_weight: 鑸嗘儏淇 彿鏉冮噸 (0.0-1.0)
   鈥?technical_weight: 鎶 鏈 俊鍙锋潈閲?(0.0-1.0)
   鈥?black_swan_threshold: 榛戝 楣呴 闄 槇鍊?(LOW/MEDIUM/HIGH)
   鈥?enable_adaptive_filtering: 鑷  傚簲杩囨护寮 鍏?

銆?. 鍩虹 鍙傛暟銆?
   鈥?POSITION_MULTIPLIER: 浠撲綅鍊嶆暟 (0.1-10)
   鈥?TREND_STRENGTH_THRESHOLD: 瓒嬪娍寮哄害闃堝 ?
   鈥?LOOKBACK_PERIOD: 鍥炵湅K绾挎暟閲?
   鈥?CHECK_INTERVAL: 绛栫暐妫 鏌 棿闅?绉?

銆?. 鍏 満杩囨护鍙傛暟銆?
   鈥?max_kline_change: 鏈 澶 崟鏍筀绾垮彉鍖?
   鈥?max_funding_rate_long: 澶氬 鏈 澶 祫閲戣垂鐜?
   鈥?min_funding_rate_short: 绌哄 鏈 灏忚祫閲戣垂鐜?
   鈥?support_buffer: 鏀 拺浣嶇紦鍐?
   鈥?resistance_buffer: 闃诲姏浣嶇紦鍐?

銆?. 姝 崯鍙傛暟銆?
   鈥?long_buffer: 澶氬 姝 崯缂撳啿
   鈥?short_buffer: 绌哄 姝 崯缂撳啿

銆?. 姝 泩鍙傛暟銆?
   鈥?tp1_multiplier_long: 澶氬 绗 竴姝 泩鍊嶆暟
   鈥?tp2_multiplier_long: 澶氬 绗 簩姝 泩鍊嶆暟
   鈥?tp1_multiplier_short: 绌哄 绗 竴姝 泩鍊嶆暟
   鈥?tp2_multiplier_short: 绌哄 绗 簩姝 泩鍊嶆暟
   鈥?tp1_position_ratio: 绗 竴姝 泩浠撲綅姣斾緥

銆?. 椋庨櫓绠 悊鍙傛暟銆?
   鈥?single_trade_risk: 鍗曠瑪浜 槗椋庨櫓姣斾緥
   鈥?daily_loss_limit: 姣忔棩浜忔崯闄愬埗
   鈥?max_consecutive_losses: 鏈 澶 繛缁 簭鎹熸 鏁?
   鈥?pause_after_losses_minutes: 浜忔崯鍚庢殏鍋滄椂闂?
   鈥?max_single_position: 鏈 澶 崟绗斾粨浣?
   鈥?max_daily_position: 鏈 澶 棩浠撲綅
   鈥?extreme_move_threshold: 鏋佺 娉 姩闃堝 ?

銆?. 浜 槗棰戠巼鍙傛暟銆?
   鈥?max_daily_trades: 姣忔棩鏈 澶 氦鏄撴 鏁?
   鈥?min_trade_interval_minutes: 鏈 灏忎氦鏄撻棿闅?
   鈥?active_hours_start: 娲昏穬寮 濮嬫椂闂?
   鈥?active_hours_end: 娲昏穬缁撴潫鏃堕棿

銆?. 浠撲綅绠 悊鍙傛暟銆?
   鈥?initial_entry_ratio: 鍒濆 鍏 満姣斾緥
   鈥?confirm_interval_kline: 纭  K绾挎暟閲?
""

        params_info_label = ttk.Label(
            params_info_frame,
            text=params_info_text,
            font=("Consolas", 8),
            foreground="#7f8c8d",
            justify=tk.LEFT
        )
        params_info_label.pack(anchor=tk.W)

        # 浼樺寲鍘嗗彶
        history_frame = ttk.LabelFrame(scheduler_frame, text="馃搵 浼樺寲鍘嗗彶", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.optimization_history_text = tk.Text(history_frame, height=10, font=("Consolas", 9), bg="#f8f9fa", relief=tk.FLAT, wrap=tk.WORD)
        self.optimization_history_text.pack(fill=tk.BOTH, expand=True)

        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.optimization_history_text.yview)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.optimization_history_text.configure(yscrollcommand=history_scrollbar.set)

    def _create_live_monitor_tab(self, parent):
        """鍒涘缓瀹炵洏鐩戞帶鏍囩 椤碉紙绠 鍖栫増锛?"
        # 涓绘 鏋?
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 鏍囬
        title_label = ttk.Label(
            main_frame,
            text="瀹炵洏浜 槗鐩戞帶闈 澘",
            font=("寰 蒋闆呴粦", 18, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        # 浜 槗鎵 閰嶇疆妗嗘灦锛堝浐瀹氾級
        exchange_frame = ttk.LabelFrame(main_frame, text="浜 槗鎵 閰嶇疆", padding="15")
        exchange_frame.pack(fill=tk.X, pady=(0, 15))

        # 鍥哄畾鏄剧 浜 槗鎵 鍜屼氦鏄撳
        ttk.Label(exchange_frame, text="浜 槗鎵 :", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(exchange_frame, text="甯佸畨 (Binance)", font=("寰 蒋闆呴粦", 10, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(exchange_frame, text="浜 槗瀵?", font=("寰 蒋闆呴粦", 10)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(exchange_frame, text="BTCUSDT", font=("寰 蒋闆呴粦", 10, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        # 璐 埛淇 伅妗嗘灦
        account_frame = ttk.LabelFrame(main_frame, text="璐 埛淇 伅", padding="15")
        account_frame.pack(fill=tk.X, pady=(0, 15))

        # 璐 埛浣欓 鏄剧
        ttk.Label(
            account_frame,
            text="璐 埛鎬讳綑棰?",
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            account_frame,
            textvariable=self.account_balance_var,
            font=("寰 蒋闆呴粦", 12, "bold"),
            foreground="#27ae60"
        ).pack(side=tk.LEFT)

        # 鏇存柊鎸夐挳
        ttk.Button(
            account_frame,
            text="馃攧 鏇存柊璐 埛淇 伅",
            command=self._update_account_info
        ).pack(side=tk.LEFT, padx=(20, 0))

        # 褰撳墠鎸佷粨妗嗘灦
        position_frame = ttk.LabelFrame(main_frame, text="褰撳墠鎸佷粨", padding="15")
        position_frame.pack(fill=tk.X, pady=(0, 15))

        # 鎸佷粨淇 伅鏄剧
        ttk.Label(
            position_frame,
            text="鎸佷粨鐘舵 ?",
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            position_frame,
            textvariable=self.position_info_var,
            font=("寰 蒋闆呴粦", 10),
            foreground="#34495e"
        ).pack(side=tk.LEFT)

        # 瀹炴椂琛屾儏妗嗘灦
        market_frame = ttk.LabelFrame(main_frame, text="瀹炴椂琛屾儏", padding="15")
        market_frame.pack(fill=tk.X, pady=(0, 15))

        # 浠锋牸鏄剧
        ttk.Label(
            market_frame,
            text="褰撳墠浠锋牸:",
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            market_frame,
            textvariable=self.current_price_var,
            font=("寰 蒋闆呴粦", 12, "bold"),
            foreground="#2980b9"
        ).pack(side=tk.LEFT, padx=(10, 0))

        # 24h娑 穼骞?
        ttk.Label(
            market_frame,
            text="24h娑 穼骞?",
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(20, 10))
        ttk.Label(
            market_frame,
            textvariable=self.price_change_var,
            font=("寰 蒋闆呴粦", 12, "bold"),
            foreground="#e74c3c"
        ).pack(side=tk.LEFT)

        # 鐩戞帶鐘舵 佹 鏋?
        status_frame = ttk.LabelFrame(main_frame, text="鐩戞帶鐘舵 ?, padding="15")"
        status_frame.pack(fill=tk.X, pady=(0, 15))

        # 鐩戞帶鐘舵 佹樉绀?
        ttk.Label(
            status_frame,
            textvariable=self.monitor_status_var,
            font=("寰 蒋闆呴粦", 10),
            foreground="#e74c3c"
        ).pack(side=tk.LEFT, padx=(0, 20))

        # AI绯荤粺鐘舵 佺畝瑕佹樉绀?
        ai_brief_frame = ttk.Frame(status_frame)
        ai_brief_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Label(ai_brief_frame, text="AI绯荤粺:", font=("寰 蒋闆呴粦", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.ai_brief_status_var = tk.StringVar(value="鏈 垵濮嬪寲")
        self.ai_brief_status_label = ttk.Label(
            ai_brief_frame,
            textvariable=self.ai_brief_status_var,
            font=("寰 蒋闆呴粦", 9),
            foreground="#7f8c8d"
        )
        self.ai_brief_status_label.pack(side=tk.LEFT)

        # 浜 槗鏃 織妗嗘灦
        log_frame = ttk.LabelFrame(main_frame, text="浜 槗鏃 織", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # 浜 槗鏃 織鏂囨湰妗?
        self.live_log_text = tk.Text(
            log_frame,
            height=15,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )

        # 浼犵粺婊氬姩鏉 紙鏇村 鏇存槑鏄撅級
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

        # 甯冨眬
        self.live_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 搴曢儴鎸夐挳妗嗘灦
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # 鍚 姩鐩戞帶鎸夐挳
        self.start_monitor_button = ttk.Button(
            button_frame,
            text="鈻?鍚 姩鐩戞帶",
            command=self._start_live_monitoring,
            width=20,
            style="Accent.TButton"
        )
        self.start_monitor_button.pack(side=tk.LEFT, padx=5)

        # 鍋滄 鐩戞帶鎸夐挳
        self.stop_monitor_button = ttk.Button(
            button_frame,
            text="鈴?鍋滄 鐩戞帶",
            command=self._stop_live_monitoring,
            width=20,
            style="TButton"
        )
        self.stop_monitor_button.pack(side=tk.LEFT, padx=5)
        self.stop_monitor_button.config(state=tk.DISABLED)

        # 娓呯 鏃 織鎸夐挳
        ttk.Button(
            button_frame,
            text="馃棏 娓呯 鏃 織",
            command=self._clear_live_log
        ).pack(side=tk.LEFT, padx=5)

    def _update_account_info(self):
        """鏇存柊璐 埛淇 伅"""
        try:
            from binance_api import BinanceAPI
            binance = BinanceAPI()
            total_balance = binance.get_total_balance()

            if total_balance is not None:
                self.account_balance_var.set(f"${total_balance:.2f}")
                self._log_live_message(f"璐 埛浣欓 鏇存柊: ${total_balance:.2f} USDT (鍙 敤+鎸佷粨淇濊瘉閲?", "SUCCESS")
            else:
                self._log_live_message(f"鑾峰彇璐 埛鎬讳綑棰濆 璐 紝灏濊瘯鑾峰彇鍙 敤浣欓 ...", "WARNING")
                balance = binance.get_account_balance()
                if balance:
                    for asset in balance:
                        if asset['asset'] == 'USDT':
                            available = float(asset['availableBalance'])
                            self.account_balance_var.set(f"${available:.2f}")
                            self._log_live_message(f"璐 埛浣欓 鏇存柊: ${available:.2f} USDT", "SUCCESS")
                            break
        except Exception as e:
            self._log_live_message(f"鏇存柊璐 埛淇 伅澶辫触: {e}", "ERROR")

    def _start_live_monitoring(self):
        """鍚 姩瀹炵洏鐩戞帶"""
        try:
            self.is_live_monitoring = True
            self.monitor_status_var.set("杩愯 涓?)"
            self.monitor_status_label.config(foreground="#27ae60")

            self.start_monitor_button.config(state=tk.DISABLED)
            self.stop_monitor_button.config(state=tk.NORMAL)

            self._log_live_message("瀹炵洏鐩戞帶宸插惎鍔?, "SUCCESS")"
            self._log_live_message(f"浜 槗鎵 : {self.live_exchange_var.get()}", "INFO")
            self._log_live_message(f"浜 槗瀵? {self.live_symbol_var.get()}", "INFO")

            # 鍚 姩鐩戞帶绾跨
            import threading
            self.live_monitor_thread = threading.Thread(
                target=self._live_monitoring_thread,
                daemon=True
            )
            self.live_monitor_thread.start()

        except Exception as e:
            self._log_live_message(f"鍚 姩鐩戞帶澶辫触: {e}", "ERROR")

    def _stop_live_monitoring(self):
        """鍋滄 瀹炵洏鐩戞帶"""
        try:
            self.is_live_monitoring = False
            self.monitor_status_var.set("宸插仠姝?)"
            self.monitor_status_label.config(foreground="#e74c3c")

            self.start_monitor_button.config(state=tk.NORMAL)
            self.stop_monitor_button.config(state=tk.DISABLED)

            self._log_live_message("瀹炵洏鐩戞帶宸插仠姝?, "WARNING")"

        except Exception as e:
            self._log_live_message(f"鍋滄 鐩戞帶澶辫触: {e}", "ERROR")

    def _create_balance_chart_tab(self, parent):
        """鍒涘缓璧勯噾鏇茬嚎鏍囩 椤?"
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # 涓绘 鏋?
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 鎺 埗鎸夐挳妗嗘灦
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 鍒锋柊鎸夐挳
        ttk.Button(
            control_frame,
            text="鍒锋柊鍥捐 ",
            command=self._update_balance_chart
        ).pack(side=tk.LEFT, padx=(0, 10))

        # 鏁版嵁缁熻 鏍囩
        self.balance_stats_var = tk.StringVar(value="鏆傛棤鏁版嵁")
        ttk.Label(
            control_frame,
            textvariable=self.balance_stats_var,
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(20, 0))

        # 鍒涘缓3涓 瓙鍥?
        self.balance_figure = Figure(figsize=(12, 12), dpi=100)

        # 24灏忔椂鍥捐 锛堟渶鍚?440涓 偣锛?
        self.balance_ax_24h = self.balance_figure.add_subplot(3, 1, 1)

        # 7澶 浘琛 紙姣忓 0:00浠 悗鐨勭 涓 涓 暟鎹 偣锛?
        self.balance_ax_7d = self.balance_figure.add_subplot(3, 1, 2)

        # 30澶 浘琛 紙姣忓 0:00浠 悗鐨勭 涓 涓 暟鎹 偣锛?
        self.balance_ax_30d = self.balance_figure.add_subplot(3, 1, 3)

        self.balance_chart_canvas = FigureCanvasTkAgg(self.balance_figure, master=main_frame)
        self.balance_chart_canvas.draw()
        self.balance_chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 娣诲姞宸 叿鏍?
        toolbar = NavigationToolbar2Tk(self.balance_chart_canvas, main_frame)
        toolbar.update()

        # 鍒濆 鍖栧浘琛?
        self._update_balance_chart()

    def _create_news_tab(self, parent):
        """鍒涘缓BTC鏂伴椈鏍囩 椤?"
        from datetime import datetime
        import webbrowser

        # 涓绘 鏋?
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 椤堕儴鎺 埗妗嗘灦
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 鍒锋柊鎸夐挳
        ttk.Button(
            control_frame,
            text="鍒锋柊鏂伴椈",
            command=self._refresh_news
        ).pack(side=tk.LEFT, padx=(0, 10))

        # 鏂伴椈鐘舵 佹爣绛?
        self.news_status_var = tk.StringVar(value="鐐瑰嚮鍒锋柊鑾峰彇鏂伴椈")
        ttk.Label(
            control_frame,
            textvariable=self.news_status_var,
            font=("寰 蒋闆呴粦", 10)
        ).pack(side=tk.LEFT, padx=(20, 0))

        # 鍒涘缓鏂伴椈鍒楄 妗嗘灦锛堝甫婊氬姩鏉 級
        news_container = ttk.Frame(main_frame)
        news_container.pack(fill=tk.BOTH, expand=True)

        # 婊氬姩鏉?
        scrollbar = ttk.Scrollbar(news_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鏂伴椈鍒楄 鏂囨湰妗?
        self.news_text = tk.Text(
            news_container,
            wrap=tk.WORD,
            font=("寰 蒋闆呴粦", 10),
            yscrollcommand=scrollbar.set,
            padx=10,
            pady=10
        )
        self.news_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.news_text.yview)

        # 閰嶇疆鏂囨湰鏍囩 鏍峰紡
        self.news_text.tag_config("title", font=("寰 蒋闆呴粦", 12, "bold"), foreground="#2c3e50")
        self.news_text.tag_config("source", font=("寰 蒋闆呴粦", 9), foreground="#7f8c8d")
        self.news_text.tag_config("time", font=("寰 蒋闆呴粦", 9), foreground="#95a5a6")
        self.news_text.tag_config("content", font=("寰 蒋闆呴粦", 10), foreground="#34495e")
        self.news_text.tag_config("link", font=("寰 蒋闆呴粦", 9, "underline"), foreground="#3498db")
        self.news_text.tag_config("separator", foreground="#bdc3c7")

        # 浣挎柊闂绘枃鏈  鍙
        self.news_text.config(state=tk.DISABLED)

    def _refresh_news(self):
        """鍒锋柊鏂伴椈鍒楄 """
        if self.news_crawler is None:
            self.news_status_var.set("鏂伴椈鐖 櫕鏈 垵濮嬪寲")
            return

        self.news_status_var.set("姝 湪鑾峰彇鏂伴椈...")
        self.news_text.config(state=tk.NORMAL)
        self.news_text.delete(1.0, tk.END)

        try:
            # 鑾峰彇鏂伴椈锛堝惎鐢 儏缁 垎鏋愬拰缈昏瘧锛?
            news_list = self.news_crawler.fetch_all_news(force_refresh=True, analyze_sentiment=True, translate=True)
            self.news_list = news_list

            if not news_list:
                self.news_text.insert(tk.END, "鏆傛棤鏂伴椈鏁版嵁\n", "content")
                self.news_status_var.set("鏆傛棤鏂伴椈鏁版嵁")
                self.news_text.config(state=tk.DISABLED)
                return

            # 鏄剧 鏂伴椈
            for i, news in enumerate(news_list):
                # 鏍囬 锛堜紭鍏堟樉绀轰腑鏂囩炕璇戯級
                title_cn = news.get('title_cn')
                title_en = news['title']
                if title_cn and title_cn != title_en:
                    self.news_text.insert(tk.END, f"{title_cn}\n", "title")
                    self.news_text.insert(tk.END, f"{title_en}\n", "source")
                else:
                    self.news_text.insert(tk.END, f"{title_en}\n", "title")

                # 鎯呯华鏍囩
                sentiment = news.get('sentiment', 'neutral')
                sentiment_score = news.get('sentiment_score', 0.0)
                if sentiment == 'positive':
                    sentiment_text = "馃煝 鐪嬪 "
                elif sentiment == 'negative':
                    sentiment_text = "馃敶 鐪嬬 "
                else:
                    sentiment_text = "鈿?涓  ?"
                self.news_text.insert(tk.END, f"鎯呯华: {sentiment_text} (寰楀垎: {sentiment_score:.2f})\n", "source")

                # 鏉 簮鍜屾椂闂?
                source_info = f"鏉 簮: {news['source']}"
                try:
                    pub_time = datetime.fromisoformat(news['published_at'])
                    time_str = pub_time.strftime("%Y-%m-%d %H:%M")
                    source_info += f" | 鏃堕棿: {time_str}"
                except:
                    pass
                self.news_text.insert(tk.END, f"{source_info}\n", "source")

                # 鏂伴椈鍐呭
                content = news['content']
                self.news_text.insert(tk.END, f"銆愭柊闂诲唴瀹广 慭n{content}\n", "content")

                # 閾炬帴
                if news.get('url'):
                    self.news_text.insert(tk.END, f"閾炬帴: {news['url']}\n", "link")

                # 鍒嗛殧绾?
                if i < len(news_list) - 1:
                    self.news_text.insert(tk.END, "\n" + "="*80 + "\n\n", "separator")

            self.news_status_var.set(f"鍏?{len(news_list)} 鏉 柊闂?)"

        except Exception as e:
            self.news_text.insert(tk.END, f"鑾峰彇鏂伴椈澶辫触: {e}\n", "content")
            self.news_status_var.set(f"鑾峰彇鏂伴椈澶辫触: {e}")

        self.news_text.config(state=tk.DISABLED)

    def _start_news_auto_refresh(self):
        """鍚 姩鏂伴椈鑷 姩鍒锋柊锛堟瘡5鍒嗛挓锛?"
        if self.news_crawler:
            self._auto_refresh_news()

    def _auto_refresh_news(self):
        """鑷 姩鍒锋柊鏂伴椈锛堝悗鍙伴潤榛樺埛鏂帮紝鍙 仛鎯呯华鍒嗘瀽缂撳瓨锛?"
        if self.news_crawler:
            try:
                print("[鏂伴椈鐖 櫕] 鑷 姩鍒锋柊鏂伴椈锛堝悗鍙帮級...")
                # 鍚庡彴鍒锋柊锛氬彧鍋氭儏缁 垎鏋愶紝涓嶅仛鏂伴椈鎬荤粨
                self.news_crawler.fetch_all_news(
                    force_refresh=False,
                    analyze_sentiment=True,
                    translate=False
                )
            except Exception as e:
                print(f"[鏂伴椈鐖 櫕] 鑷 姩鍒锋柊澶辫触: {e}")

        # 瀹夋帓涓嬩竴娆 埛鏂帮紙5鍒嗛挓 = 300000姣  锛?
        self.news_auto_refresh_job = self.root.after(300000, self._auto_refresh_news)

    def _update_balance_chart(self):
        """鏇存柊璧勯噾鏇茬嚎鍥?- 鏄剧 3涓 笉鍚屾椂闂磋寖鍥寸殑鍥捐 """
        try:
            if not self.balance_data:
                # 娓呯 鎵 鏈夊浘琛 苟鏄剧 "鏆傛棤鏁版嵁"
                for ax in [self.balance_ax_24h, self.balance_ax_7d, self.balance_ax_30d]:
                    ax.clear()
                    ax.text(0.5, 0.5, "鏆傛棤鏁版嵁", ha='center', va='center',
                           transform=ax.transAxes, fontsize=14, color='gray')
                self.balance_chart_canvas.draw()
                self.balance_stats_var.set("鏆傛棤鏁版嵁")
                return

            import pandas as pd
            from datetime import datetime, timedelta

            df = pd.DataFrame(self.balance_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 1. 24灏忔椂鍥捐 锛堟渶鍚?440涓 偣锛?
            self.balance_ax_24h.clear()
            df_24h = df.tail(1440) if len(df) > 1440 else df

            if len(df_24h) > 0:
                # 鑾峰彇24灏忔椂鍥捐 鐨勫垵濮嬭祫閲戯紙浠庤 鏃堕棿娈靛唴鐨勭 涓 涓 暟鎹 偣锛?
                initial_balance_24h = self._get_initial_balance_for_chart(df_24h)
                self._plot_balance_chart(self.balance_ax_24h, df_24h, initial_balance_24h, "24灏忔椂璧勯噾鏇茬嚎")
            else:
                self.balance_ax_24h.text(0.5, 0.5, "鏆傛棤24灏忔椂鏁版嵁", ha='center', va='center',
                                       transform=self.balance_ax_24h.transAxes, fontsize=12, color='gray')

            # 2. 7澶 浘琛 紙姣忓 0:00浠 悗鐨勭 涓 涓 暟鎹 偣锛?
            self.balance_ax_7d.clear()
            df_7d = self._get_daily_data(df, days=7)

            if len(df_7d) > 0:
                # 鑾峰彇7澶 浘琛 殑鍒濆 璧勯噾锛堜粠璇 椂闂存 鍐呯殑绗 竴涓 暟鎹 偣锛?
                initial_balance_7d = self._get_initial_balance_for_chart(df_7d)
                self._plot_balance_chart(self.balance_ax_7d, df_7d, initial_balance_7d, "7澶 祫閲戞洸绾?)"
            else:
                self.balance_ax_7d.text(0.5, 0.5, "鏆傛棤7澶 暟鎹?, ha='center', va='center',"
                                      transform=self.balance_ax_7d.transAxes, fontsize=12, color='gray')

            # 3. 30澶 浘琛 紙姣忓 0:00浠 悗鐨勭 涓 涓 暟鎹 偣锛?
            self.balance_ax_30d.clear()
            df_30d = self._get_daily_data(df, days=30)

            if len(df_30d) > 0:
                # 鑾峰彇30澶 浘琛 殑鍒濆 璧勯噾锛堜粠璇 椂闂存 鍐呯殑绗 竴涓 暟鎹 偣锛?
                initial_balance_30d = self._get_initial_balance_for_chart(df_30d)
                self._plot_balance_chart(self.balance_ax_30d, df_30d, initial_balance_30d, "30澶 祫閲戞洸绾?)"
            else:
                self.balance_ax_30d.text(0.5, 0.5, "鏆傛棤30澶 暟鎹?, ha='center', va='center',"
                                       transform=self.balance_ax_30d.transAxes, fontsize=12, color='gray')

            self.balance_figure.tight_layout()
            self.balance_chart_canvas.draw()

            # 鏇存柊缁熻 鏁版嵁
            if len(df) >= 2:
                first_balance = df['current_balance'].iloc[0]
                last_balance = df['current_balance'].iloc[-1]
                change = last_balance - first_balance
                change_pct = (change / first_balance * 100) if first_balance > 0 else 0

                self.balance_stats_var.set(
                    f"鎬绘暟鎹 偣: {len(df)} | 鍙樺寲: ${change:+.2f} ({change_pct:+.2f}%)"
                )
            else:
                self.balance_stats_var.set(f"鎬绘暟鎹 偣: {len(df)}")

        except Exception as e:
            print(f"鏇存柊璧勯噾鏇茬嚎鍥惧 璐? {e}")

    def _get_daily_data(self, df, days=7):
        """鑾峰彇鎸囧畾澶 暟鍐呮瘡澶?:00浠 悗鐨勭 涓 涓 暟鎹 偣"""
        try:
            from datetime import datetime, timedelta

            if len(df) == 0:
                return pd.DataFrame()

            # 鑾峰彇鏈 杩慸ays澶 殑鏁版嵁
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_date]

            if len(recent_df) == 0:
                return pd.DataFrame()

            # 鎸夋棩鏈熷垎缁勶紝鍙栨瘡澶 殑绗 竴涓 暟鎹 偣
            recent_df['date'] = recent_df['timestamp'].dt.date
            daily_data = recent_df.groupby('date').first().reset_index()

            return daily_data
        except Exception as e:
            print(f"鑾峰彇姣忔棩鏁版嵁澶辫触: {e}")
            return pd.DataFrame()

    def _get_initial_balance_for_chart(self, df):
        """鑾峰彇鍥捐 瀵瑰簲鐨勫垵濮嬭祫閲?"
        try:
            if len(df) == 0:
                return 0

            # 浼樺厛浣跨敤璁板綍鐨勫垵濮嬭祫閲?
            initial_data = df[df['initial_balance'] > 0]
            if len(initial_data) > 0:
                return initial_data['initial_balance'].iloc[0]

            # 濡傛灉娌 湁璁板綍鍒濆 璧勯噾锛屼娇鐢  鏃堕棿娈靛唴鐨勭 涓 涓 祫閲戝 ?
            return df['current_balance'].iloc[0]

        except Exception as e:
            print(f"鑾峰彇鍒濆 璧勯噾澶辫触: {e}")
            return 0

    def _plot_balance_chart(self, ax, df, initial_balance, title):
        """缁樺埗鍗曚釜璧勯噾鏇茬嚎鍥?"
        try:
            import matplotlib.dates as mdates

            # 鍙 粯鍒跺綋鍓嶈祫閲戠嚎锛堝幓鎺夊垵濮嬭祫閲戠嚎锛?
            line, = ax.plot(
                df['timestamp'],
                df['current_balance'],
                color='green',
                linewidth=2,
                label='褰撳墠璧勯噾'
            )

            # 濉 厖鍖哄煙
            ax.fill_between(
                df['timestamp'],
                df['current_balance'],
                alpha=0.3,
                color='lightgreen'
            )

            # 璁剧疆鍥捐 灞炴 ?
            ax.set_xlabel('鏃堕棿', fontsize=10)
            ax.set_ylabel('璧勯噾 (USDT)', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 璁剧疆Y杞磋寖鍥达紙鍙 熀浜庡綋鍓嶈祫閲戯級
            min_balance = df['current_balance'].min()
            max_balance = df['current_balance'].max()

            margin = (max_balance - min_balance) * 0.05
            if margin == 0:
                margin = max_balance * 0.02 if max_balance > 0 else 1

            ax.set_ylim(min_balance - margin, max_balance + margin)

            # 璁剧疆X杞村埢搴 細鐩存帴浣跨敤鏁版嵁鐐圭殑鏃堕棿锛岃 岄潪鍧囧寑鍒嗗竷
            # 鏍规嵁鏁版嵁閲忛 夋嫨鍚堥 傜殑鍒诲害鏁伴噺
            if len(df) <= 10:
                # 鏁版嵁鐐瑰皯锛屾樉绀烘墍鏈夊埢搴?
                tick_indices = list(range(len(df)))
            else:
                # 鏁版嵁鐐瑰 锛岄 夋嫨绾?-10涓 埢搴?
                num_ticks = min(10, len(df))
                step = max(1, len(df) // num_ticks)
                tick_indices = list(range(0, len(df), step))
                # 纭 繚鏈 鍚庝竴涓 偣涔熸樉绀?
                if tick_indices[-1] != len(df) - 1:
                    tick_indices.append(len(df) - 1)

            # 璁剧疆鍒诲害浣嶇疆鍜屾爣绛?
            tick_dates = df['timestamp'].iloc[tick_indices]
            ax.set_xticks(tick_dates)

            # 鏍煎紡鍖栨棩鏈熸爣绛?
            tick_labels = []
            for date in tick_dates:
                if '24灏忔椂' in title:
                    label = date.strftime('%H:%M')
                else:
                    label = date.strftime('%m-%d')
                tick_labels.append(label)

            ax.set_xticklabels(tick_labels)

            # 鑷 姩璋冩暣鏃 湡鏄剧
            ax.tick_params(axis='x', rotation=45)

        except Exception as e:
            print(f"缁樺埗璧勯噾鏇茬嚎鍥惧 璐? {e}")

    def _live_monitoring_thread(self):
        """瀹炵洏鐩戞帶绾跨 """
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

                    # 鑾峰彇褰撳墠浠锋牸锛堟瘡5绉掓洿鏂帮級
                    ticker = binance.client.futures_symbol_ticker(symbol=symbol)

                    if ticker:
                        price = float(ticker['price'])
                        self.current_price_var.set(f"${price:.2f}")

                        # 鑾峰彇24h娑 穼骞?
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

                    # 鑾峰彇鎸佷粨淇 伅锛堟瘡5绉掓洿鏂帮級
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
                                    f"{side} {abs(pos_amt):.4f} | 鍏 満: ${entry_price:.2f} | 鐩堜簭: ${unrealized_pnl:.2f}"
                                )
                                has_position = True
                                break
                        if not has_position:
                            self.position_info_var.set("鏆傛棤鎸佷粨")
                    except Exception as e:
                        self._log_live_message(f"鑾峰彇鎸佷粨淇 伅澶辫触: {e}", "WARNING")

                    # 鏇存柊璐 埛浣欓 锛堟瘡30绉掓洿鏂帮級
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
                            self._log_live_message(f"鏇存柊璐 埛浣欓 澶辫触: {e}", "WARNING")

                    # 鏇存柊浜 槗姹囨 伙紙姣?0绉掓洿鏂帮級
                    if now - last_stats_update > 10:
                        self._update_trading_stats()
                        last_stats_update = now

                    time.sleep(5)  # 姣?绉掓洿鏂颁竴娆?

                except Exception as e:
                    self._log_live_message(f"鑾峰彇琛屾儏鏁版嵁澶辫触: {e}", "WARNING")
                    time.sleep(5)

        except Exception as e:
            self._log_live_message(f"鐩戞帶绾跨 閿欒 : {e}", "ERROR")

    def _update_trading_stats(self):
        """鏇存柊浜 槗姹囨 荤粺璁?"
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

            # 浠巗trategy涓  鍙栦氦鏄撳巻鍙?
            if hasattr(self, "strategy") and self.strategy and hasattr(self.strategy, "trade_history"):
                for trade in self.strategy.trade_history:
                    trade_time = trade.get("time")
                    if trade_time:
                        trade_date = trade_time.date()
                        pnl_pct = trade.get("pnl_pct", 0.0)

                        # 浠婃棩缁熻
                        if trade_date == today:
                            today_trades += 1
                            today_profit += pnl_pct

                        # 褰撳懆缁熻
                        if trade_date >= week_start:
                            week_trades += 1
                            week_profit += pnl_pct

                        # 褰撴湀缁熻
                        if trade_date >= month_start:
                            month_trades += 1
                            month_profit += pnl_pct

            # 鏇存柊UI鏄剧
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
            self._log_live_message(f"鏇存柊浜 槗姹囨 诲 璐? {e}", "WARNING")

    def _run_fingpt_analysis(self, symbol: str):
        """杩愯 FinGPT鑸嗘儏鍒嗘瀽骞跺湪缁堢 鏄剧 缁撴灉"""
        try:
            if self.fingpt_analyzer is None:
                self._log_live_message("[FinGPT] 鍒嗘瀽鍣 湭鍒濆 鍖栵紝璺宠繃鑸嗘儏鍒嗘瀽", "WARNING")
                return

            self._log_live_message(f"[FinGPT] 姝 湪鍒嗘瀽 {symbol} 甯傚満鎯呯华...", "INFO")

            # 杩愯 鑸嗘儏鍒嗘瀽
            coin_symbol = symbol.replace("USDT", "")
            sentiment_result = self.fingpt_analyzer.analyze_market_sentiment(coin_symbol)

            if sentiment_result:
                # 鏄剧 鎯呯华缁撴灉
                sentiment = sentiment_result.get('overall_sentiment', 'UNKNOWN')
                confidence = sentiment_result.get('confidence', 0)
                risk_level = sentiment_result.get('risk_level', 'LOW')

                # 鏍规嵁鎯呯华璁剧疆棰滆壊绾 埆
                level = "INFO"
                if sentiment == "BEARISH":
                    level = "WARNING"
                elif sentiment == "BULLISH":
                    level = "SUCCESS"

                self._log_live_message(
                    f"[FinGPT] 甯傚満鎯呯华: {sentiment} | 缃 俊搴? {confidence:.1%} | 椋庨櫓: {risk_level}",
                    level
                )

                # 鏄剧 鍏抽敭鎸囨爣
                metrics = sentiment_result.get('metrics', {})
                if metrics:
                    fear_greed = metrics.get('fear_greed_index', 'N/A')
                    news_sentiment = metrics.get('news_sentiment', 'N/A')
                    self._log_live_message(
                        f"[FinGPT] 鎭愭儳璐  鎸囨暟: {fear_greed} | 鏂伴椈鎯呯华: {news_sentiment}",
                        "INFO"
                    )

                # 鏄剧 浜 槗寤鸿
                recommendation = sentiment_result.get('recommendation', '')
                if recommendation:
                    self._log_live_message(f"[FinGPT] 寤鸿 : {recommendation}", "INFO")

                # 鏄剧 椋庨櫓璀 憡
                risk_factors = sentiment_result.get('risk_factors', [])
                if risk_factors and risk_level == "HIGH":
                    self._log_live_message(f"[FinGPT] 鈿狅笍 妫 娴嬪埌楂橀 闄 洜绱?", "WARNING")
                    for factor in risk_factors[:2]:
                        self._log_live_message(f"  鈥?{factor}", "WARNING")

                # 濡傛灉鏈夌瓥鐣 崗璋冨櫒锛屾樉绀轰俊鍙疯繃婊 儏鍐?
                if self.strategy_coordinator is not None:
                    self._log_live_message("[FinGPT] 淇 彿杩囨护鐘舵 ? 娲昏穬", "SUCCESS")
            else:
                self._log_live_message("[FinGPT] 鑸嗘儏鍒嗘瀽杩斿洖绌虹粨鏋?, "WARNING")"

        except Exception as e:
            self._log_live_message(f"[FinGPT] 鑸嗘儏鍒嗘瀽澶辫触: {e}", "ERROR")

    def _log_live_message(self, message, level="INFO"):
        """璁板綍瀹炵洏鐩戞帶鏃 織锛堝悓鏃惰 褰曞埌涓 釜鏃 織鏂囨湰妗嗭級"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        color_map = {
            "INFO": "#34495e",
            "SUCCESS": "#27ae60",
            "WARNING": "#f39c12",
            "ERROR": "#e74c3c"
        }

        color = color_map.get(level, "#34495e")

        # 涓哄疄鐩樼洃鎺 棩蹇楁枃鏈  閰嶇疆棰滆壊鏍囩
        if hasattr(self, 'live_log_text'):
            self.live_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.live_log_text.tag_config("INFO", foreground=color_map["INFO"])
            self.live_log_text.tag_config("SUCCESS", foreground=color_map["SUCCESS"])
            self.live_log_text.tag_config("WARNING", foreground=color_map["WARNING"])
            self.live_log_text.tag_config("ERROR", foreground=color_map["ERROR"])

            # 鎻掑叆鏃 織鍒板疄鐩樼洃鎺 枃鏈
            self.live_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.live_log_text.insert(tk.END, f"{message}\n", level)

            # 鍙 湁褰撴粴鍔 潯鍦 渶搴曢儴鏃舵墠鑷 姩璺熼殢
            scroll_position = self.live_log_text.yview()
            if scroll_position[1] >= 0.99:
                self.live_log_text.see(tk.END)

        # 涓篈I绛栫暐涓 績鏃 織鏂囨湰妗嗛厤缃  鑹叉爣绛?
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.ai_strategy_log_text.tag_config("INFO", foreground=color_map["INFO"])
            self.ai_strategy_log_text.tag_config("SUCCESS", foreground=color_map["SUCCESS"])
            self.ai_strategy_log_text.tag_config("WARNING", foreground=color_map["WARNING"])
            self.ai_strategy_log_text.tag_config("ERROR", foreground=color_map["ERROR"])

            # 鎻掑叆鏃 織鍒癆I绛栫暐涓 績鏂囨湰妗?
            self.ai_strategy_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.ai_strategy_log_text.insert(tk.END, f"{message}\n", level)

            # 鍙 湁褰撴粴鍔 潯鍦 渶搴曢儴鏃舵墠鑷 姩璺熼殢
            scroll_position = self.ai_strategy_log_text.yview()
            if scroll_position[1] >= 0.99:
                self.ai_strategy_log_text.see(tk.END)

    def _clear_live_log(self):
        """娓呯 瀹炵洏鐩戞帶鏃 織锛堝悓鏃舵竻绌轰袱涓 棩蹇楁枃鏈  锛?"
        # 娓呯 瀹炵洏鐩戞帶鏃 織鏂囨湰妗?
        if hasattr(self, 'live_log_text'):
            self.live_log_text.delete(1.0, tk.END)

        # 娓呯 AI绛栫暐涓 績鏃 織鏂囨湰妗?
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.delete(1.0, tk.END)

        # 璁板綍娓呯 鏃 織娑堟伅
        self._log_live_message("鏃 織宸叉竻绌?, "INFO")"

    def _clear_ai_strategy_log(self):
        """娓呯 AI绛栫暐涓 績鏃 織"""
        # 鍙 竻绌篈I绛栫暐涓 績鏃 織鏂囨湰妗?
        if hasattr(self, 'ai_strategy_log_text'):
            self.ai_strategy_log_text.delete(1.0, tk.END)
            # 璁板綍娓呯 鏃 織娑堟伅鍒癆I绛栫暐涓 績鏃 織
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.ai_strategy_log_text.tag_config("timestamp", foreground="#95a5a6")
            self.ai_strategy_log_text.tag_config("INFO", foreground="#34495e")
            self.ai_strategy_log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.ai_strategy_log_text.insert(tk.END, "AI绛栫暐涓 績鏃 織宸叉竻绌篭n", "INFO")
            self.ai_strategy_log_text.see(tk.END)

    def _load_live_config(self):
        """鍔犺浇瀹炵洏鐩戞帶閰嶇疆"""
        try:
            # 鍔犺浇閰嶇疆鏂囦欢
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
        """鏇存柊澶氭櫤鑳戒綋绯荤粺鐘舵 佹樉绀?"
        # 鏇存柊FinGPT鐘舵 侊紙AI绛栫暐涓 績鏍囩 锛?
        try:
            if self.fingpt_analyzer is not None:
                self.fingpt_status_var.set("杩愯 涓?)"
                if hasattr(self, 'fingpt_status_label'):
                    self.fingpt_status_label.config(foreground="#27ae60")
            else:
                self.fingpt_status_var.set("鏈 垵濮嬪寲")
                if hasattr(self, 'fingpt_status_label'):
                    self.fingpt_status_label.config(foreground="#e74c3c")
        except:
            pass

        # 鏇存柊绛栫暐鍗忚皟鍣 姸鎬侊紙AI绛栫暐涓 績鏍囩 锛?
        try:
            if self.strategy_coordinator is not None:
                self.coordinator_status_var.set("杩愯 涓?)"
                if hasattr(self, 'coordinator_status_label'):
                    self.coordinator_status_label.config(foreground="#27ae60")
            else:
                self.coordinator_status_var.set("鏈 垵濮嬪寲")
                if hasattr(self, 'coordinator_status_label'):
                    self.coordinator_status_label.config(foreground="#e74c3c")
        except:
            pass

        # 鏇存柊瀹炵洏鐩戞帶鏍囩 鐨凙I鐘舵 佺畝瑕佹樉绀?
        try:
            if hasattr(self, 'ai_brief_status_var') and hasattr(self, 'ai_brief_status_label'):
                if self.fingpt_analyzer is not None and self.strategy_coordinator is not None:
                    self.ai_brief_status_var.set("杩愯 涓?)"
                    self.ai_brief_status_label.config(foreground="#27ae60")
                elif self.fingpt_analyzer is not None or self.strategy_coordinator is not None:
                    self.ai_brief_status_var.set("閮 垎杩愯 ")
                    self.ai_brief_status_label.config(foreground="#f39c12")
                else:
                    self.ai_brief_status_var.set("鏈 垵濮嬪寲")
                    self.ai_brief_status_label.config(foreground="#7f8c8d")
        except:
            pass

    def _init_ai_scheduler(self):
        """鍒濆 鍖朅I绛栫暐涓 績璋冨害鍣?"
        try:
            # 灏濊瘯瀵煎叆璋冨害鍣?
            from ai_strategy_scheduler import AIStrategyScheduler
            from parameter_integrator import ParameterIntegrator

            # 鍒濆 鍖栧弬鏁伴泦鎴愬櫒
            self.parameter_integrator = ParameterIntegrator()

            # 鍒濆 鍖栬皟搴 櫒
            self.ai_scheduler = AIStrategyScheduler(
                parameter_integrator=self.parameter_integrator,
                strategy_coordinator=self.strategy_coordinator,
                check_interval_minutes=60
            )

            # 璁剧疆鍥炶皟
            self.ai_scheduler.on_optimization_complete = self._on_optimization_complete
            self.ai_scheduler.on_error = self._on_optimization_error

            print("鉁?AI绛栫暐涓 績璋冨害鍣 垵濮嬪寲瀹屾垚")
            return True

        except Exception as e:
            print(f"鉁?AI绛栫暐涓 績璋冨害鍣 垵濮嬪寲澶辫触: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _start_ai_scheduler(self):
        """鍚 姩AI绛栫暐涓 績璋冨害鍣?"
        try:
            # 纭 繚璋冨害鍣 凡鍒濆 鍖?
            if not hasattr(self, "ai_scheduler") or self.ai_scheduler is None:
                if not self._init_ai_scheduler():
                    self._log_ai_strategy_message("鉂?AI绛栫暐涓 績璋冨害鍣 垵濮嬪寲澶辫触", "ERROR")
                    return

            # 璁剧疆妫 鏌 棿闅?
            interval = self.optimization_interval_var.get()
            self.ai_scheduler.check_interval_minutes = interval

            # 鍚 姩璋冨害鍣?
            if self.ai_scheduler.start():
                self.ai_scheduler_status_var.set("杩愯 涓?)"
                self.start_scheduler_button.config(state=tk.DISABLED)
                self.stop_scheduler_button.config(state=tk.NORMAL)

                # 鏇存柊鐘舵 佹爣绛鹃 鑹?
                for widget in self.ai_scheduler_status_var._root.winfo_children():
                    if hasattr(widget, 'configure') and 'foreground' in widget.keys():
                        widget.configure(foreground="#27ae60")

                self._log_ai_strategy_message(f"鉁?AI绛栫暐涓 績璋冨害鍣 凡鍚 姩 (妫 鏌 棿闅? {interval}鍒嗛挓)", "SUCCESS")
                self._log_ai_strategy_message("   馃搳 璋冨害鍣 皢瀹氭湡鍒嗘瀽甯傚満骞惰嚜鍔 紭鍖栦氦鏄撳弬鏁?, "INFO")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鍚 姩AI绛栫暐涓 績璋冨害鍣  璐? {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _stop_ai_scheduler(self):
        """鍋滄 AI绛栫暐涓 績璋冨害鍣?"
        try:
            if hasattr(self, "ai_scheduler") and self.ai_scheduler:
                if self.ai_scheduler.stop():
                    self.ai_scheduler_status_var.set("宸插仠姝?)"
                    self.start_scheduler_button.config(state=tk.NORMAL)
                    self.stop_scheduler_button.config(state=tk.DISABLED)

                    self._log_ai_strategy_message("鈴?AI绛栫暐涓 績璋冨害鍣 凡鍋滄 ", "WARNING")

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鍋滄 AI绛栫暐涓 績璋冨害鍣  璐? {e}", "ERROR")

    def _open_strategy_config(self):
        """鎵撳紑绛栫暐鍙傛暟閰嶇疆瀵硅瘽妗?"
        try:
            # 鑾峰彇褰撳墠閰嶇疆
            current_config = self._get_current_strategy_config()

            # 鎵撳紑瀵硅瘽妗?
            dialog = StrategyConfigDialog(self.root, current_config)
            result = dialog.show()

            if result is not None:
                # 搴旂敤閰嶇疆
                self._apply_strategy_config(result)
                self._log_ai_strategy_message("鉁?绛栫暐閰嶇疆宸插簲鐢?, "SUCCESS")"
                messagebox.showinfo("鎴愬姛", "绛栫暐閰嶇疆宸叉垚鍔熷簲鐢 紒")

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鎵撳紑閰嶇疆瀵硅瘽妗嗗 璐? {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _get_current_strategy_config(self):
        """鑾峰彇褰撳墠绛栫暐閰嶇疆"""
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
            # 浠庣瓥鐣 崗璋冨櫒鑾峰彇閰嶇疆
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                coord_config = getattr(self.strategy_coordinator, "config", {})
                config["coordinator"] = coord_config.copy()
        except Exception:
            pass

        try:
            # 浠巗trategy_config.py璇诲彇榛樿 閰嶇疆
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
        """搴旂敤绛栫暐閰嶇疆"""
        try:
            # 鏇存柊绛栫暐鍗忚皟鍣 厤缃?
            if hasattr(self, "strategy_coordinator") and self.strategy_coordinator:
                if "coordinator" in config:
                    for key, value in config["coordinator"].items():
                        if key in self.strategy_coordinator.config:
                            self.strategy_coordinator.config[key] = value

            # 鏇存柊姝 湪杩愯 鐨勭瓥鐣 疄渚?
            if hasattr(self, "strategy") and self.strategy:
                if hasattr(self.strategy, "update_config"):
                    success = self.strategy.update_config(config)
                    if success:
                        self._log_ai_strategy_message("鉁?绛栫暐瀹炰緥閰嶇疆宸插姩鎬佹洿鏂?, "SUCCESS")"

            # 鏇存柊strategy_config.py鏂囦欢
            self._update_strategy_config_file(config)

            self._log_ai_strategy_message("鉁?绛栫暐閰嶇疆宸叉洿鏂?, "SUCCESS")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?搴旂敤閰嶇疆澶辫触: {e}", "ERROR")
            raise

    def _update_strategy_config_file(self, config):
        """鏇存柊strategy_config.py鏂囦欢"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "strategy_config.py")

            config_content = '''"""绛栫暐閰嶇疆鏂囦欢 - 鐢卞弬鏁伴泦鎴愬櫒鑷 姩鐢熸垚"""'

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
''

            # 濉 厖閰嶇疆鍊?
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

            # 鍐欏叆鏂囦欢
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)

        except Exception as e:
            print(f"鏇存柊閰嶇疆鏂囦欢澶辫触: {e}")
            raise

    def _trigger_optimization_now(self):
        """绔嬪嵆瑙 彂涓 娆 紭鍖?"
        try:
            # 纭 繚璋冨害鍣 凡鍒濆 鍖?
            if not hasattr(self, "ai_scheduler") or self.ai_scheduler is None:
                if not self._init_ai_scheduler():
                    self._log_ai_strategy_message("鉂?AI绛栫暐涓 績璋冨害鍣 垵濮嬪寲澶辫触", "ERROR")
                    return

            self._log_ai_strategy_message("鈿?瑙 彂绔嬪嵆浼樺寲...", "INFO")

            if self.ai_scheduler.trigger_optimization_now():
                self._log_ai_strategy_message("鉁?浼樺寲浠诲姟宸叉彁浜?, "SUCCESS")"
            else:
                self._log_ai_strategy_message("鈿狅笍 璇峰厛鍚 姩璋冨害鍣?, "WARNING")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?瑙 彂浼樺寲澶辫触: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _on_optimization_complete(self, record: Dict):
        """浼樺寲瀹屾垚鍥炶皟"""
        try:
            timestamp = record.get("timestamp", "")
            dt = datetime.fromisoformat(timestamp)

            # 鏇存柊涓婃 浼樺寲鏃堕棿
            self.last_optimization_var.set(dt.strftime("%H:%M:%S"))

            # 娣诲姞鍒板巻鍙茶 褰?
            self._add_optimization_history(record)

            # 璁板綍鏃 織
            self._log_ai_strategy_message(f"鉁?浼樺寲瀹屾垚: {dt.strftime('%Y-%m-%d %H:%M:%S')}", "SUCCESS")

        except Exception as e:
            print(f"浼樺寲瀹屾垚鍥炶皟閿欒 : {e}")

    def _on_optimization_error(self, error: Exception):
        """浼樺寲閿欒 鍥炶皟"""
        try:
            self._log_ai_strategy_message(f"鉂?浼樺寲閿欒 : {error}", "ERROR")
        except Exception as e:
            print(f"浼樺寲閿欒 鍥炶皟閿欒 : {e}")

    def _add_optimization_history(self, record: Dict):
        """娣诲姞浼樺寲鍘嗗彶璁板綍"""
        try:
            timestamp = record.get("timestamp", "")
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")

            params = record.get("optimized_parameters", {})

            history_entry = f"[{time_str}] 浼樺寲瀹屾垚\n"

            if "market_analysis" in params:
                history_entry += f"  甯傚満鍒嗘瀽: {params['market_analysis']}\n"

            if "optimization_reasoning" in params:
                history_entry += f"  浼樺寲鐞嗙敱: {params['optimization_reasoning']}\n"

            history_entry += "-" * 50 + "\n"

            # 娣诲姞鍒版枃鏈
            self.optimization_history_text.insert(tk.END, history_entry)
            self.optimization_history_text.see(tk.END)

        except Exception as e:
            print(f"娣诲姞浼樺寲鍘嗗彶閿欒 : {e}")

    def _initialize_auto_optimization_system(self):
        """鍒濆 鍖栬嚜鍔 寲浼樺寲绯荤粺"""
        try:
            if PerformanceMonitor is None or AutoOptimizationPipeline is None:
                self._log_ai_strategy_message("鈿狅笍 鑷 姩鍖栦紭鍖栨 鍧椾笉鍙 敤", "WARNING")
                return False

            # 鍒濆 鍖栨  兘鐩戞帶鍣?
            if self.performance_monitor is None:
                # 浠嶨UI閫夐 鑾峰彇鍒 柇妯 紡
                mode_map = {
                    "鍥哄畾闃堝 ?: "threshold","
                    "AI鍒 柇": "ai",
                    "娣峰悎妯 紡": "hybrid"
                }
                judgment_mode = mode_map.get(self.optimization_judgment_mode_var.get(), "threshold")

                self.performance_monitor = PerformanceMonitor(
                    strategy_instance=None,  # 绋嶅悗璁剧疆
                    check_interval_seconds=900,  # 榛樿 15鍒嗛挓
                    min_trades_for_analysis=5,
                    performance_history_file="performance_history.json",
                    judgment_mode=judgment_mode
                )

                # 璁剧疆闃堝 肩獊鐮村洖璋?
                self.performance_monitor.on_threshold_breach = self._on_threshold_breach_callback
                self.performance_monitor.on_performance_update = self._on_performance_update_callback

                self._log_ai_strategy_message("鉁?鎬 兘鐩戞帶鍣 垵濮嬪寲瀹屾垚", "SUCCESS")

                # 濡傛灉宸叉湁绛栫暐瀹炰緥锛岀珛鍗宠繛鎺?
                if hasattr(self, 'strategy') and self.strategy:
                    try:
                        self.performance_monitor.set_strategy_instance(self.strategy)
                        self._log_ai_strategy_message("  宸茶繛鎺 埌褰撳墠绛栫暐瀹炰緥", "INFO")
                    except Exception as e:
                        self._log_ai_strategy_message(f"  杩炴帴绛栫暐瀹炰緥澶辫触: {e}", "WARNING")

            # 鍒濆 鍖栬嚜鍔 寲浼樺寲绠 亾
            if self.auto_optimization_pipeline is None:
                self.auto_optimization_pipeline = AutoOptimizationPipeline(
                    symbol="BTCUSDT",
                    config_path="strategy_config.py",
                    optimization_history_file="auto_optimization_history.json"
                )

                # 璁剧疆浼樺寲鍥炶皟
                self.auto_optimization_pipeline.set_callback("optimization_complete", self._on_optimization_complete_callback)
                self.auto_optimization_pipeline.set_callback("optimization_error", self._on_optimization_error_callback)

                self._log_ai_strategy_message("鉁?鑷 姩鍖栦紭鍖栫 閬撳垵濮嬪寲瀹屾垚", "SUCCESS")

            # 鍒濆 鍖栧弬鏁伴泦鎴愬櫒
            if self.parameter_integrator is None:
                self.parameter_integrator = ParameterIntegrator(
                    config_file="strategy_config.py",
                    backup_dir="config_backups"
                )
                self._log_ai_strategy_message("鉁?鍙傛暟闆嗘垚鍣 垵濮嬪寲瀹屾垚", "SUCCESS")

            # 纭 繚鎬 兘鐩戞帶鍣 繛鎺 埌绛栫暐瀹炰緥锛堝 鏋滃瓨鍦 級
            if self.performance_monitor and hasattr(self, 'strategy') and self.strategy:
                try:
                    self.performance_monitor.set_strategy_instance(self.strategy)
                    self._log_ai_strategy_message("鎬 兘鐩戞帶鍣 凡杩炴帴鍒扮瓥鐣 疄渚?, "INFO")"
                except Exception as e:
                    self._log_ai_strategy_message(f"杩炴帴鎬 兘鐩戞帶鍣  璐? {e}", "WARNING")

            return True

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鑷 姩鍖栦紭鍖栫郴缁熷垵濮嬪寲澶辫触: {e}", "ERROR")
            return False

    def _toggle_auto_optimization(self):
        """鍒囨崲鑷 姩鍖栦紭鍖栫姸鎬?"
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("鉂?鑷 姩鍖栦紭鍖栫郴缁熷垵濮嬪寲澶辫触锛屾棤娉曞惎鐢?, "ERROR")"
                return

            if not self.is_auto_optimization_enabled:
                # 鍚 敤鑷 姩鍖栦紭鍖?
                self.is_auto_optimization_enabled = True
                self.auto_optimization_status_var.set("杩愯 涓?)"

                # 鍚 姩鎬 兘鐩戞帶
                if self.performance_monitor:
                    self.performance_monitor.start_monitoring()

                self._log_ai_strategy_message("鉁?鑷 姩鍖栦紭鍖栧凡鍚 敤", "SUCCESS")
                self._log_ai_strategy_message("  绯荤粺灏嗘寔缁 洃鎺 氦鏄撹 鐜板苟鍦 槇鍊肩獊鐮存椂鑷 姩浼樺寲", "INFO")

                # 鏇存柊鎸夐挳鏂囨湰
                # 鎸夐挳鏂囨湰浼氬湪涓嬫 鐣岄潰鏇存柊鏃跺弽鏄?
            else:
                # 绂佺敤鑷 姩鍖栦紭鍖?
                self.is_auto_optimization_enabled = False
                self.auto_optimization_status_var.set("绂佺敤")

                # 鍋滄 鎬 兘鐩戞帶
                if self.performance_monitor:
                    self.performance_monitor.stop_monitoring()

                self._log_ai_strategy_message("鈴革笍 鑷 姩鍖栦紭鍖栧凡绂佺敤", "INFO")

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鍒囨崲鑷 姩鍖栦紭鍖栫姸鎬佸 璐? {e}", "ERROR")

    def _apply_judgment_mode(self):
        """搴旂敤鍒 柇妯 紡鍒囨崲"""
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("鉂?鑷 姩鍖栦紭鍖栫郴缁熸湭鍒濆 鍖栵紝鏃犳硶鍒囨崲妯 紡", "ERROR")
                return

            mode_display = self.optimization_judgment_mode_var.get()
            mode_map = {
                "鍥哄畾闃堝 ?: "threshold","
                "AI鍒 柇": "ai",
                "娣峰悎妯 紡": "hybrid"
            }
            judgment_mode = mode_map.get(mode_display, "threshold")

            # 鍒囨崲妯 紡
            if self.performance_monitor:
                self.performance_monitor.set_judgment_mode(
                    mode=judgment_mode
                )
                self._log_ai_strategy_message(f"鉁?鍒 柇妯 紡宸插垏鎹 负: {mode_display}", "SUCCESS")
            else:
                self._log_ai_strategy_message("鉂?鎬 兘鐩戞帶鍣 湭鍒濆 鍖?, "ERROR")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鍒囨崲鍒 柇妯 紡澶辫触: {e}", "ERROR")

    def _trigger_manual_optimization(self):
        """瑙 彂鎵嬪姩浼樺寲"""
        try:
            if not self._initialize_auto_optimization_system():
                self._log_ai_strategy_message("鉂?鑷 姩鍖栦紭鍖栫郴缁熷垵濮嬪寲澶辫触锛屾棤娉曚紭鍖?, "ERROR")"
                return

            self._log_ai_strategy_message("馃殌 瑙 彂鎵嬪姩浼樺寲...", "INFO")

            # 瑙 彂浼樺寲绠 亾
            if self.auto_optimization_pipeline:
                success = self.auto_optimization_pipeline.trigger_optimization(
                    threshold_breaches=["鎵嬪姩瑙 彂"],
                    performance_data={},
                    force_optimization=True
                )

                if success:
                    self._log_ai_strategy_message("鉁?鎵嬪姩浼樺寲宸插惎鍔 紝璇风瓑寰呭畬鎴?, "SUCCESS")"
                else:
                    self._log_ai_strategy_message("鈿狅笍 鎵嬪姩浼樺寲鍚 姩澶辫触锛堝彲鑳芥 鍦 繍琛岋級", "WARNING")
            else:
                self._log_ai_strategy_message("鉂?鑷 姩鍖栦紭鍖栫 閬撴湭鍒濆 鍖?, "ERROR")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?瑙 彂鎵嬪姩浼樺寲澶辫触: {e}", "ERROR")

    def _on_threshold_breach_callback(self, breaches, performance_data):
        """闃堝 肩獊鐮村洖璋冨嚱鏁?"
        try:
            self._log_ai_strategy_message("鈿狅笍 妫 娴嬪埌鎬 兘闃堝 肩獊鐮达紒", "WARNING")
            self._log_ai_strategy_message(f"  绐佺牬椤? {', '.join(breaches)}", "INFO")

            # 鏄剧 鎬 兘鏁版嵁鎽樿
            if performance_data:
                summary = []
                for key, value in performance_data.items():
                    if isinstance(value, (int, float)):
                        summary.append(f"{key}: {value:.3f}")

                if summary:
                    self._log_ai_strategy_message(f"  褰撳墠鎬 兘: {', '.join(summary[:3])}", "INFO")

            # 瑙 彂鑷 姩鍖栦紭鍖?
            if self.auto_optimization_pipeline and self.is_auto_optimization_enabled:
                self._log_ai_strategy_message("馃攧 鑷 姩瑙 彂浼樺寲娴佺 ...", "INFO")

                success = self.auto_optimization_pipeline.trigger_optimization(
                    threshold_breaches=breaches,
                    performance_data=performance_data,
                    force_optimization=True
                )

                if success:
                    self._log_ai_strategy_message("鉁?鑷 姩鍖栦紭鍖栧凡鍚 姩", "SUCCESS")
                else:
                    self._log_ai_strategy_message("鈿狅笍 鑷 姩鍖栦紭鍖栧惎鍔  璐 紙鍙 兘姝 湪杩愯 锛?, "WARNING")"

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?闃堝 肩獊鐮村 鐞嗗 璐? {e}", "ERROR")

    def _on_performance_update_callback(self, performance_data):
        """鎬 兘鏇存柊鍥炶皟鍑芥暟"""
        # 鍙 互鍦 繖閲屾洿鏂癎UI鏄剧 鎬 兘鎸囨爣
        pass

    def _on_optimization_complete_callback(self, optimization_record):
        """浼樺寲瀹屾垚鍥炶皟鍑芥暟"""
        try:
            if optimization_record.get("success"):
                self._log_ai_strategy_message("鉁?鑷 姩鍖栦紭鍖栧畬鎴愶紒", "SUCCESS")

                # 鏄剧 浼樺寲鎽樿
                report = optimization_record.get("final_report", {})
                if report:
                    performance = report.get("backtest_performance", {})

                    summary = []
                    if "total_return_pct" in performance:
                        summary.append(f"鏀剁泭鐜? {performance['total_return_pct']:.1f}%")
                    if "sharpe_ratio" in performance:
                        summary.append(f"澶忔櫘姣旂巼: {performance['sharpe_ratio']:.2f}")
                    if "win_rate_pct" in performance:
                        summary.append(f"鑳滅巼: {performance['win_rate_pct']:.1f}%")

                    if summary:
                        self._log_ai_strategy_message(f"  浼樺寲缁撴灉: {', '.join(summary)}", "INFO")

                # 灏濊瘯鑷 姩闆嗘垚鍙傛暟
                self._integrate_optimized_parameters(optimization_record)

            else:
                error = optimization_record.get("error", "鏈 煡閿欒 ")
                self._log_ai_strategy_message(f"鉂?鑷 姩鍖栦紭鍖栧 璐? {error}", "ERROR")

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?浼樺寲瀹屾垚鍥炶皟澶勭悊澶辫触: {e}", "ERROR")

    def _on_optimization_error_callback(self, error, optimization_record):
        """浼樺寲閿欒 鍥炶皟鍑芥暟"""
        self._log_ai_strategy_message(f"鉂?鑷 姩鍖栦紭鍖栧彂鐢熼敊璇? {error}", "ERROR")

    def _integrate_optimized_parameters(self, optimization_record):
        """闆嗘垚浼樺寲鍚庣殑鍙傛暟"""
        try:
            if not self.parameter_integrator:
                self._log_ai_strategy_message("鈿狅笍 鍙傛暟闆嗘垚鍣 湭鍒濆 鍖栵紝璺宠繃鍙傛暟闆嗘垚", "WARNING")
                return

            # 浠庝紭鍖栬 褰曚腑鎻愬彇鍙傛暟
            steps = optimization_record.get("steps", [])
            integrated_params = {}

            for step in steps:
                if step.get("description", "").startswith("鏁村悎瀹屾垚"):
                    integrated_params = step.get("integrated_parameters", {})
                    break

            if not integrated_params:
                self._log_ai_strategy_message("鈿狅笍 鏈 壘鍒颁紭鍖栧悗鐨勫弬鏁帮紝璺宠繃闆嗘垚", "WARNING")
                return

            self._log_ai_strategy_message("馃敡 姝 湪闆嗘垚浼樺寲鍚庣殑鍙傛暟...", "INFO")

            # 闆嗘垚鍙傛暟
            result = self.parameter_integrator.integrate_parameters(integrated_params)

            if result.get("success"):
                self._log_ai_strategy_message(f"鉁?鍙傛暟闆嗘垚鎴愬姛: {result.get('parameters_updated', 0)}涓 弬鏁板凡鏇存柊", "SUCCESS")

                # 鐢熸垚鎶 憡
                report = self.parameter_integrator.generate_integration_report(result)
                self._log_ai_strategy_message("=== 鍙傛暟闆嗘垚鎶 憡 ===", "INFO")
                for line in report.split('\n'):
                    if line.strip():
                        self._log_ai_strategy_message(f"  {line}", "INFO")
            else:
                error = result.get("error", "鏈 煡閿欒 ")
                self._log_ai_strategy_message(f"鉂?鍙傛暟闆嗘垚澶辫触: {error}", "ERROR")

        except Exception as e:
            self._log_ai_strategy_message(f"鉂?鍙傛暟闆嗘垚澶辫触: {e}", "ERROR")

    def _log_ai_strategy_message(self, message, level="INFO"):
        """璁板綍AI绛栫暐涓 績娑堟伅"""
        try:
            # 璁板綍鍒癆I绛栫暐涓 績鏃 織
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"

            if hasattr(self, 'ai_strategy_log_text'):
                self.ai_strategy_log_text.insert(tk.END, log_entry)
                self.ai_strategy_log_text.see(tk.END)

                # 鏍规嵁绾 埆搴旂敤棰滆壊
                if level == "ERROR":
                    self.ai_strategy_log_text.tag_add("ERROR", f"end-2l linestart", f"end-2l lineend")
                elif level == "SUCCESS":
                    self.ai_strategy_log_text.tag_add("SUCCESS", f"end-2l linestart", f"end-2l lineend")
                elif level == "WARNING":
                    self.ai_strategy_log_text.tag_add("WARNING", f"end-2l linestart", f"end-2l lineend")

            # 鍚屾椂璁板綍鍒版帶鍒跺彴锛堢敤浜庤皟璇曪級
            print(f"[AI绛栫暐涓 績] {message}")

        except Exception as e:
            print(f"璁板綍AI绛栫暐涓 績娑堟伅澶辫触: {e}")

    def _update_confirm_counts_display(self):
        """鏇存柊纭  娆 暟鏄剧 锛圫pinbox鍊兼敼鍙樻椂璋冪敤锛?"
        try:
            entry_count = self.entry_confirm_count_var.get()
            reverse_count = self.reverse_confirm_count_var.get()
            consecutive_pred = self.require_consecutive_prediction_var.get()
            post_entry_hours = self.post_entry_hours_var.get()
            take_profit_min_pct = self.take_profit_min_pct_var.get()

            # 杩欓噷鍙 互娣诲姞瀹炴椂鏄剧 鏇存柊閫昏緫
            # 鏆傛椂鍙  褰曟棩蹇?
            self._log_ai_strategy_message(
                f"鍙傛暟璁剧疆: 寮 浠搟entry_count}娆? 骞充粨{reverse_count}娆? 杩炵画棰勬祴{consecutive_pred}娆? 寮 浠撳悗璁 椂{post_entry_hours}灏忔椂, 鏈 灏忔 鐩坽take_profit_min_pct}%",
                "INFO"
            )
        except Exception as e:
            self._log_ai_strategy_message(f"鏇存柊鍙傛暟鏄剧 澶辫触: {e}", "ERROR")

    def _apply_confirm_counts(self):
        """搴旂敤纭  娆 暟璁剧疆鍒扮瓥鐣 疄渚?"
        try:
            # 鑾峰彇Spinbox鐨勫 ?
            entry_count = int(self.entry_confirm_count_var.get())
            reverse_count = int(self.reverse_confirm_count_var.get())
            consecutive_pred = int(self.require_consecutive_prediction_var.get())
            post_entry_hours = float(self.post_entry_hours_var.get())
            take_profit_min_pct = float(self.take_profit_min_pct_var.get())

            # 楠岃瘉鑼冨洿
            if not (1 <= entry_count <= 10):
                self._log_ai_strategy_message(f"寮 浠撶 璁  鏁拌秴鍑鸿寖鍥? {entry_count} (1-10)", "ERROR")
                return
            if not (1 <= reverse_count <= 10):
                self._log_ai_strategy_message(f"骞充粨纭  娆 暟瓒呭嚭鑼冨洿: {reverse_count} (1-10)", "ERROR")
                return
            if not (1 <= consecutive_pred <= 10):
                self._log_ai_strategy_message(f"杩炵画棰勬祴纭  娆 暟瓒呭嚭鑼冨洿: {consecutive_pred} (1-10)", "ERROR")
                return
            if not (0.5 <= post_entry_hours <= 24):
                self._log_ai_strategy_message(f"寮 浠撳悗璁 椂瓒呭嚭鑼冨洿: {post_entry_hours} (0.5-24)", "ERROR")
                return
            if not (0.1 <= take_profit_min_pct <= 10):
                self._log_ai_strategy_message(f"鏈 灏忔 鐩堟瘮渚嬭秴鍑鸿寖鍥? {take_profit_min_pct} (0.1-10)", "ERROR")
                return

            # 鏇存柊绛栫暐瀹炰緥锛堝 鏋滃瓨鍦 級
            if hasattr(self, 'strategy') and self.strategy is not None:
                self.strategy.entry_confirm_count = entry_count
                self.strategy.reverse_confirm_count = reverse_count
                self.strategy.require_consecutive_prediction = consecutive_pred
                self.strategy.post_entry_hours = post_entry_hours
                self.strategy.take_profit_min_pct = take_profit_min_pct
                self.strategy.consecutive_entry_count = 0  # 閲嶇疆璁 暟鍣?
                self.strategy.consecutive_reverse_count = 0  # 閲嶇疆璁 暟鍣?
                self.strategy.last_entry_signal = None  # 閲嶇疆淇 彿
                self.strategy.last_reverse_signal = None  # 閲嶇疆淇 彿

                self._log_ai_strategy_message(
                    f"鉁?鍙傛暟宸插簲鐢 埌褰撳墠绛栫暐: 寮 浠搟entry_count}娆? 骞充粨{reverse_count}娆? 杩炵画棰勬祴{consecutive_pred}娆? 寮 浠撳悗璁 椂{post_entry_hours}灏忔椂, 鏈 灏忔 鐩坽take_profit_min_pct}%",
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  娉 剰: 宸查噸缃 繛缁  鏁板櫒鍜屾渶鍚庝俊鍙疯 褰?,"
                    "INFO"
                )
            else:
                self._log_ai_strategy_message(
                    f"鉁?鍙傛暟宸蹭繚瀛? 寮 浠搟entry_count}娆? 骞充粨{reverse_count}娆? 杩炵画棰勬祴{consecutive_pred}娆? 寮 浠撳悗璁 椂{post_entry_hours}灏忔椂, 鏈 灏忔 鐩坽take_profit_min_pct}%",
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  灏嗗湪涓嬫 鍚 姩浜 槗鏃剁敓鏁?,"
                    "INFO"
                )

            # 淇濆瓨璁剧疆鍒伴厤缃 枃浠?
            self._save_confirm_counts_settings(entry_count, reverse_count, consecutive_pred, post_entry_hours, take_profit_min_pct)

        except ValueError as e:
            self._log_ai_strategy_message(f"鏁板 艰浆鎹  璐? {e} (璇疯緭鍏?-10鐨勬暣鏁?", "ERROR")
        except Exception as e:
            self._log_ai_strategy_message(f"搴旂敤纭  娆 暟澶辫触: {e}", "ERROR")

    def _save_confirm_counts_settings(self, entry_count, reverse_count, consecutive_pred, post_entry_hours, take_profit_min_pct):
        """淇濆瓨纭  娆 暟璁剧疆鍒伴厤缃 枃浠?"
        try:
            settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

            # 鍔犺浇鐜版湁璁剧疆
            settings = {}
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

            # 鏇存柊纭  娆 暟璁剧疆
            if 'confirm_counts' not in settings:
                settings['confirm_counts'] = {}

            settings['confirm_counts']['entry_confirm_count'] = entry_count
            settings['confirm_counts']['reverse_confirm_count'] = reverse_count
            settings['confirm_counts']['require_consecutive_prediction'] = consecutive_pred
            settings['confirm_counts']['post_entry_hours'] = post_entry_hours
            settings['confirm_counts']['take_profit_min_pct'] = take_profit_min_pct
            settings['confirm_counts']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 淇濆瓨璁剧疆
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)

            self._log_ai_strategy_message(f"璁剧疆宸蹭繚瀛樺埌閰嶇疆鏂囦欢: {settings_file}", "INFO")

        except Exception as e:
            self._log_ai_strategy_message(f"淇濆瓨璁剧疆澶辫触: {e}", "WARNING")

    def _load_confirm_counts_settings(self):
        """浠庨厤缃 枃浠跺姞杞界 璁  鏁拌 缃?"
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
                        f"宸插姞杞藉弬鏁拌 缃? 寮 浠搟entry_count}娆? 骞充粨{reverse_count}娆? 杩炵画棰勬祴{consecutive_pred}娆? 寮 浠撳悗璁 椂{post_entry_hours}灏忔椂, 鏈 灏忔 鐩坽take_profit_min_pct}%",
                        "INFO"
                    )
                    return True

            return False

        except Exception as e:
            self._log_ai_strategy_message(f"鍔犺浇璁剧疆澶辫触: {e}", "WARNING")
            return False

    def _create_main_backtest_tab(self, parent):
        """鍒涘缓涓诲洖娴嬫爣绛鹃 """
        frame = ttk.Frame(parent, padding="15")
        frame.pack(fill=tk.BOTH, expand=True)

        # 鏍囬
        title_label = ttk.Label(
            frame,
            text="馃搳 绛栫暐鍥炴祴绯荤粺",
            font=("寰 蒋闆呴粦", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(pady=(0, 15))

        # 鍥炴祴鍙傛暟閰嶇疆鍖哄煙
        config_frame = ttk.LabelFrame(frame, text="鈿欙笍 鍥炴祴鍙傛暟閰嶇疆", padding="12")
        config_frame.pack(fill=tk.X, pady=(0, 15))

        # 鏁版嵁鏂囦欢閫夋嫨
        load_row = ttk.Frame(config_frame)
        load_row.pack(fill=tk.X, pady=5)

        ttk.Label(load_row, text="鍘嗗彶鏁版嵁鏂囦欢:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_data_file_var = tk.StringVar(value="")
        data_file_entry = ttk.Entry(load_row, textvariable=self.backtest_data_file_var, width=50)
        data_file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_row, text="馃搨 閫夋嫨鏂囦欢", command=self._select_backtest_data_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_row, text="馃搵 鍒锋柊鍒楄 ", command=self._refresh_backtest_data_list).pack(side=tk.LEFT, padx=5)

        # 鏃堕棿鑼冨洿閫夋嫨
        time_range_frame = ttk.LabelFrame(config_frame, text="馃搮 鏁版嵁鎶撳彇鏃堕棿鑼冨洿", padding="10")
        time_range_frame.pack(fill=tk.X, pady=10)

        # 寮 濮嬫椂闂?
        start_time_row = ttk.Frame(time_range_frame)
        start_time_row.pack(fill=tk.X, pady=5)

        ttk.Label(start_time_row, text="寮 濮嬫椂闂?", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_start_year_var = tk.StringVar(value="")
        self.backtest_start_month_var = tk.StringVar(value="")
        self.backtest_start_day_var = tk.StringVar(value="")

        ttk.Label(start_time_row, text="骞?").pack(side=tk.LEFT, padx=(10, 0))
        year_values = [str(y) for y in range(2020, 2027)]
        start_year_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_year_var, values=year_values, width=6, state="readonly")
        start_year_combobox.pack(side=tk.LEFT)

        ttk.Label(start_time_row, text="鏈?").pack(side=tk.LEFT, padx=(10, 0))
        month_values = [f"{m:02d}" for m in range(1, 13)]
        start_month_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_month_var, values=month_values, width=4, state="readonly")
        start_month_combobox.pack(side=tk.LEFT)

        ttk.Label(start_time_row, text="鏃?").pack(side=tk.LEFT, padx=(10, 0))
        day_values = [f"{d:02d}" for d in range(1, 32)]
        start_day_combobox = ttk.Combobox(start_time_row, textvariable=self.backtest_start_day_var, values=day_values, width=4, state="readonly")
        start_day_combobox.pack(side=tk.LEFT)

        # 缁撴潫鏃堕棿
        end_time_row = ttk.Frame(time_range_frame)
        end_time_row.pack(fill=tk.X, pady=5)

        ttk.Label(end_time_row, text="缁撴潫鏃堕棿:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_end_year_var = tk.StringVar(value="")
        self.backtest_end_month_var = tk.StringVar(value="")
        self.backtest_end_day_var = tk.StringVar(value="")

        ttk.Label(end_time_row, text="骞?").pack(side=tk.LEFT, padx=(10, 0))
        end_year_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_year_var, values=year_values, width=6, state="readonly")
        end_year_combobox.pack(side=tk.LEFT)

        ttk.Label(end_time_row, text="鏈?").pack(side=tk.LEFT, padx=(10, 0))
        end_month_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_month_var, values=month_values, width=4, state="readonly")
        end_month_combobox.pack(side=tk.LEFT)

        ttk.Label(end_time_row, text="鏃?").pack(side=tk.LEFT, padx=(10, 0))
        end_day_combobox = ttk.Combobox(end_time_row, textvariable=self.backtest_end_day_var, values=day_values, width=4, state="readonly")
        end_day_combobox.pack(side=tk.LEFT)

        # 璁剧疆榛樿 鏃堕棿锛堟渶杩?0澶 級
        from datetime import datetime, timedelta
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        self.backtest_start_year_var.set(str(thirty_days_ago.year))
        self.backtest_start_month_var.set(f"{thirty_days_ago.month:02d}")
        self.backtest_start_day_var.set(f"{thirty_days_ago.day:02d}")
        self.backtest_end_year_var.set(str(today.year))
        self.backtest_end_month_var.set(f"{today.month:02d}")
        self.backtest_end_day_var.set(f"{today.day:02d}")

        # 鍙傛暟琛?锛氬垵濮嬭祫閲?
        config_row2 = ttk.Frame(config_frame)
        config_row2.pack(fill=tk.X, pady=5)

        ttk.Label(config_row2, text="鍒濆 璧勯噾(USDT):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_capital_var = tk.StringVar(value="10000")
        capital_entry = ttk.Entry(config_row2, textvariable=self.backtest_capital_var, width=15)
        capital_entry.pack(side=tk.LEFT, padx=5)

        # 鍙傛暟琛?锛氭墜缁 垂鐜?
        config_row3 = ttk.Frame(config_frame)
        config_row3.pack(fill=tk.X, pady=5)

        ttk.Label(config_row3, text="鎵嬬画璐圭巼(%):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_fee_var = tk.StringVar(value="0.1")
        fee_spinbox = ttk.Spinbox(config_row3, from_=0, to=1, increment=0.01, textvariable=self.backtest_fee_var, width=12)
        fee_spinbox.pack(side=tk.LEFT, padx=5)

        # 鍙傛暟琛?锛氭粦鐐?
        config_row4 = ttk.Frame(config_frame)
        config_row4.pack(fill=tk.X, pady=5)

        ttk.Label(config_row4, text="婊戠偣(%):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.backtest_slippage_var = tk.StringVar(value="0.05")
        slippage_spinbox = ttk.Spinbox(config_row4, from_=0, to=1, increment=0.01, textvariable=self.backtest_slippage_var, width=12)
        slippage_spinbox.pack(side=tk.LEFT, padx=5)

        # 鎿嶄綔鎸夐挳鍖?
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))

        self.run_backtest_btn = ttk.Button(
            button_frame,
            text="鈻讹笍 寮 濮嬪洖娴?,"
            command=self._run_backtest,
            style="Accent.TButton",
            width=20
        )
        self.run_backtest_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_backtest_btn = ttk.Button(
            button_frame,
            text="鈴癸笍 鍋滄 鍥炴祴",
            command=self._stop_backtest,
            style="Accent.TButton",
            width=20,
            state="disabled"
        )
        self.stop_backtest_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="馃摜 鎶撳彇鍌 瓨鏁版嵁",
            command=self._download_historical_data,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="馃搵 瀵煎嚭缁撴灉",
            command=self._export_backtest_results,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="馃棏锔?娓呯 缁撴灉",
            command=self._clear_backtest_results,
            width=15
        ).pack(side=tk.LEFT)

        # 杩涘害鏉?
        progress_frame = ttk.Frame(frame)
        progress_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(progress_frame, text="鍥炴祴杩涘害:").pack(side=tk.LEFT, padx=(0, 10))
        self.backtest_progress = ttk.Progressbar(progress_frame, mode="determinate", length=400, style="green.Horizontal.TProgressbar")
        self.backtest_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.backtest_progress_label = ttk.Label(progress_frame, text="0%", width=6)
        self.backtest_progress_label.pack(side=tk.LEFT, padx=10)

        # 鍥炴祴缁撴灉缁熻 鍖?
        stats_frame = ttk.LabelFrame(frame, text="馃搱 鍥炴祴缁撴灉缁熻 ", padding="12")
        stats_frame.pack(fill=tk.X, pady=(0, 15))

        # 缁熻 琛?
        stats_row1 = ttk.Frame(stats_frame)
        stats_row1.pack(fill=tk.X, pady=5)

        self.backtest_total_return_var = tk.StringVar(value="-")
        self.backtest_win_rate_var = tk.StringVar(value="-")
        self.backtest_profit_factor_var = tk.StringVar(value="-")
        self.backtest_max_drawdown_var = tk.StringVar(value="-")

        self._create_stat_item(stats_row1, "鎬绘敹鐩婄巼", self.backtest_total_return_var, "#27ae60")
        self._create_stat_item(stats_row1, "鑳滅巼", self.backtest_win_rate_var, "#3498db")
        self._create_stat_item(stats_row1, "鐩堜簭姣?, self.backtest_profit_factor_var, "#9b59b6")"
        self._create_stat_item(stats_row1, "鏈 澶 洖鎾?, self.backtest_max_drawdown_var, "#e74c3c")"

        # 缁熻 琛?
        stats_row2 = ttk.Frame(stats_frame)
        stats_row2.pack(fill=tk.X, pady=5)

        self.backtest_total_trades_var = tk.StringVar(value="-")
        self.backtest_win_trades_var = tk.StringVar(value="-")
        self.backtest_loss_trades_var = tk.StringVar(value="-")
        self.backtest_avg_profit_var = tk.StringVar(value="-")

        self._create_stat_item(stats_row2, "鎬讳氦鏄撴 鏁?, self.backtest_total_trades_var, "#2c3e50")"
        self._create_stat_item(stats_row2, "鐩堝埄娆 暟", self.backtest_win_trades_var, "#27ae60")
        self._create_stat_item(stats_row2, "浜忔崯娆 暟", self.backtest_loss_trades_var, "#e74c3c")
        self._create_stat_item(stats_row2, "骞冲潎鐩堝埄", self.backtest_avg_profit_var, "#f39c12")

        # 鍥炴祴浜 槗鏃 織鍖?
        log_frame = ttk.LabelFrame(frame, text="馃摑 浜 槗璁板綍", padding="12")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.backtest_log_text = tk.Text(
            log_frame,
            height=12,
            font=("Consolas", 11),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )

        # 浼犵粺婊氬姩鏉 紙鏇村 鏇存槑鏄撅級
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

        # 甯冨眬
        self.backtest_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _create_stat_item(self, parent, label_text, var, color):
        """鍒涘缓缁熻 椤圭粍浠?"
        item_frame = ttk.Frame(parent)
        item_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Label(item_frame, text=label_text, font=("寰 蒋闆呴粦", 9), foreground="#7f8c8d").pack(anchor=tk.W)
        ttk.Label(item_frame, textvariable=var, font=("寰 蒋闆呴粦", 14, "bold"), foreground=color).pack(anchor=tk.W)

    def _run_backtest(self):
        """杩愯 鍥炴祴"""
        import threading
        self.run_backtest_btn.config(state="disabled")
        self.stop_backtest_btn.config(state="normal")
        self.backtest_progress["value"] = 0
        self.backtest_progress_label.config(text="0%")

        try:
            data_file_path = self.backtest_data_file_var.get().strip()
            if not data_file_path:
                messagebox.showwarning("璀 憡", "璇峰厛閫夋嫨鍘嗗彶鏁版嵁鏂囦欢锛?)"
                self.run_backtest_btn.config(state="normal")
                self.stop_backtest_btn.config(state="disabled")
                return

            capital = float(self.backtest_capital_var.get())
            fee_rate = float(self.backtest_fee_var.get()) / 100
            slippage = float(self.backtest_slippage_var.get()) / 100

            self.backtest_log_text.insert(tk.END, f"[寮 濮嬪洖娴媇 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.backtest_log_text.insert(tk.END, f"鏁版嵁鑾峰彇鏂瑰紡: 杞藉叆鍘嗗彶鏁版嵁\n")
            self.backtest_log_text.insert(tk.END, f"鏁版嵁鏂囦欢: {data_file_path}\n")
            self.backtest_log_text.insert(tk.END, f"鍙傛暟: 璧勯噾=${capital:.2f}, 鎵嬬画璐?{fee_rate*100:.2f}%, 婊戠偣={slippage*100:.2f}%\n")

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
            self.backtest_log_text.insert(tk.END, f"[閿欒 ] {e}\n")
            import traceback
            traceback.print_exc()
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")

    def _simulate_backtest(self, initial_capital, fee_rate, slippage, data_file_path):
        """妯 嫙鍥炴祴鏍稿績閫昏緫 - 浣跨敤鐪熷疄绛栫暐鍥炴祴"""
        from datetime import datetime, timedelta
        from binance_api import BinanceAPI
        from professional_strategy import ProfessionalTradingStrategy
        import pandas as pd
        import os

        self.backtest_log_text.insert(tk.END, "鑾峰彇鍘嗗彶鏁版嵁...\n")
        self._update_progress(10)
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("鑾峰彇鍘嗗彶鏁版嵁...")

        df = None

        self.backtest_log_text.insert(tk.END, "[杞藉叆鏁版嵁] 姝 湪浠庢枃浠跺姞杞芥暟鎹?..\n")
        try:
            if not os.path.exists(data_file_path):
                self.backtest_log_text.insert(tk.END, f"[閿欒 ] 鏂囦欢涓嶅瓨鍦? {data_file_path}\n")
                self._update_progress(100)
                self.run_backtest_btn.config(state="normal")
                self.stop_backtest_btn.config(state="disabled")
                return

            df = pd.read_csv(data_file_path, encoding='utf-8')
            self.backtest_log_text.insert(tk.END, f"[杞藉叆鏁版嵁] 鎴愬姛鍔犺浇 {len(df)} 鏉 暟鎹甛n")
            if len(df) > 0 and 'timestamps' in df.columns:
                self.backtest_log_text.insert(tk.END, f"[鏁版嵁] 鏈 鏃 椂闂? {df['timestamps'].iloc[0]}\n")
                self.backtest_log_text.insert(tk.END, f"[鏁版嵁] 鏈 鏅氭椂闂? {df['timestamps'].iloc[-1]}\n")
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[閿欒 ] 杞藉叆鏁版嵁澶辫触: {e}\n")
            import traceback
            self.backtest_log_text.insert(tk.END, f"{traceback.format_exc()}\n")
            self._update_progress(100)
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
            return

        if df is None or len(df) < 100:
            self.backtest_log_text.insert(tk.END, f"K绾挎暟鎹 笉瓒筹紝鏃犳硶鍥炴祴 (df={df is not None}, len={len(df) if df is not None else 0})\n")
            self.backtest_log_text.insert(tk.END, f"[閿欒 ] K绾挎暟鎹 笉瓒筹紝鑷冲皯闇 瑕?00鏉?(褰撳墠: {len(df) if df is not None else 0})\n")
            self._update_progress(100)
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")
            return

        self.backtest_log_text.insert(tk.END, f"鎴愬姛鑾峰彇 {len(df)} 鏉 绾挎暟鎹甛n")
        self.backtest_log_text.insert(tk.END, f"[鏁版嵁] 鑾峰彇鍒?{len(df)} 鏉 绾挎暟鎹甛n")
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info(f"鎴愬姛鑾峰彇 {len(df)} 鏉 绾挎暟鎹?)"

        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        self.backtest_log_text.insert(tk.END, "[寮 濮嬪洖娴媇 姝 湪杩愯 绛栫暐鍥炴祴...\n")

        self._update_progress(20)
        self.backtest_log_text.insert(tk.END, "寮 濮嬭繍琛岀瓥鐣 洖娴?..\n")
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("[寮 濮嬪洖娴媇 姝 湪杩愯 绛栫暐鍥炴祴...")

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
            self.backtest_log_text.insert(tk.END, "[绛栫暐鍒涘缓] 绛栫暐涓嶅瓨鍦 紝姝 湪鍒涘缓...\n")
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
                "瓒嬪娍鐖嗗彂": "trend",
                "闇囪崱濂楀埄": "range",
                "娑堟伅绐佺牬": "breakout",
                "鑷 姩绛栫暐": "auto",
                "鏃堕棿绛栫暐": "time",
            }
            strategy_type = strategy_map.get(self.strategy_var.get(), "trend")

            # 5濂椾氦鏄撶瓥鐣 己鍒朵娇鐢?m鏃堕棿鍛 湡锛堝洜涓烘 鍨嬭 缁冩暟鎹 槸5m锛?
            is_professional_strategy = self.strategy_var.get() in strategy_map
            if is_professional_strategy:
                if timeframe != "5m":
                    self.backtest_log_text.insert(tk.END, f"[绛栫暐鍒涘缓] 妫 娴嬪埌閫夋嫨浜唟self.strategy_var.get()}绛栫暐锛屽己鍒朵娇鐢?m鏃堕棿鍛 湡\n")
                    timeframe = "5m"
                    self.timeframe_var.set("5m")

            BinanceAPI()

            # 鑾峰彇瀹屾暣鐨凙I绛栫暐閰嶇疆
            ai_strategy_config = None
            if hasattr(self, "_get_ai_strategy_config_from_ui"):
                try:
                    ai_strategy_config = self._get_ai_strategy_config_from_ui()
                    self.backtest_log_text.insert(tk.END, "[绛栫暐鍒涘缓] 宸茶幏鍙朅I绛栫暐閰嶇疆\n")
                except Exception as e:
                    self.backtest_log_text.insert(tk.END, f"[绛栫暐鍒涘缓] 鑾峰彇AI绛栫暐閰嶇疆澶辫触: {e}\n")

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

            self.backtest_log_text.insert(tk.END, "[绛栫暐鍒涘缓] 绛栫暐鍒涘缓鎴愬姛锛乗n")
            self.backtest_log_text.insert(tk.END, "[绛栫暐閰嶇疆] 姝 湪鍔犺浇绛栫暐棰勮 閰嶇疆...\n")

            # 鍔犺浇绛栫暐棰勮 鐨勫畬鏁撮厤缃 紙鍖呮嫭姝 泩姝 崯鍙傛暟绛夛級
            from professional_strategy import StrategyProfiles
            self.strategy.strategy_profile = StrategyProfiles.get_profile(strategy_type)
            self.strategy._load_strategy_config()

            self.backtest_log_text.insert(tk.END, "[绛栫暐閰嶇疆] 绛栫暐棰勮 閰嶇疆鍔犺浇瀹屾垚锛乗n")
            if hasattr(self, "backtest_logger"):
                self.backtest_logger.info("[绛栫暐鍒涘缓] 绛栫暐鍒涘缓鎴愬姛锛?)"

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
            self.backtest_log_text.insert(tk.END, "鍥炴祴瀹屾垚锛乗n")
            if hasattr(self, "backtest_logger"):
                self.backtest_logger.info("鍥炴祴瀹屾垚锛?)"

        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[閿欒 ] 鍥炴祴鎵  澶辫触: {e}\n")
            import traceback
            traceback.print_exc()
        finally:
            self.run_backtest_btn.config(state="normal")
            self.stop_backtest_btn.config(state="disabled")

    def _display_real_backtest_results(self, results):
        """鏄剧 鐪熷疄鍥炴祴缁撴灉"""
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
        self.backtest_log_text.insert(tk.END, "馃搳 鍥炴祴缁撴灉缁熻 \n")
        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        self.backtest_log_text.insert(tk.END, f"鍒濆 璧勯噾: ${results['initial_capital']:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"鏈 缁堣祫閲? ${results['final_capital']:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"鎬绘敹鐩婄巼: {total_return:+.2%}\n")
        self.backtest_log_text.insert(tk.END, f"鑳滅巼: {win_rate:.2%}\n")
        self.backtest_log_text.insert(tk.END, f"鐩堜簭姣? {profit_factor:.2f}\n")
        self.backtest_log_text.insert(tk.END, f"鏈 澶 洖鎾? {-max_drawdown:.2%}\n")
        self.backtest_log_text.insert(tk.END, f"鎬讳氦鏄撴 鏁? {total_trades}\n")
        self.backtest_log_text.insert(tk.END, f"鐩堝埄娆 暟: {win_trades}\n")
        self.backtest_log_text.insert(tk.END, f"浜忔崯娆 暟: {loss_trades}\n")
        self.backtest_log_text.insert(tk.END, f"骞冲潎鐩堝埄: {avg_profit:+.2f}%\n")
        self.backtest_log_text.insert(tk.END, f"鎬绘墜缁 垂: ${results['total_fees']:.2f}\n")

        # 鍚屾椂璁板綍鍒板洖娴嬫棩蹇楁枃浠?
        if hasattr(self, "backtest_logger"):
            self.backtest_logger.info("="*80)
            self.backtest_logger.info("馃搳 鍥炴祴缁撴灉缁熻 ")
            self.backtest_logger.info("="*80)
            self.backtest_logger.info(f"鍒濆 璧勯噾: ${results['initial_capital']:.2f}")
            self.backtest_logger.info(f"鏈 缁堣祫閲? ${results['final_capital']:.2f}")
            self.backtest_logger.info(f"鎬绘敹鐩婄巼: {total_return:+.2%}")
            self.backtest_logger.info(f"鑳滅巼: {win_rate:.2%}")
            self.backtest_logger.info(f"鐩堜簭姣? {profit_factor:.2f}")
            self.backtest_logger.info(f"鏈 澶 洖鎾? {-max_drawdown:.2%}")
            self.backtest_logger.info(f"鎬讳氦鏄撴 鏁? {total_trades}")
            self.backtest_logger.info(f"鐩堝埄娆 暟: {win_trades}")
            self.backtest_logger.info(f"浜忔崯娆 暟: {loss_trades}")
            self.backtest_logger.info(f"骞冲潎鐩堝埄: {avg_profit:+.2f}%")
            self.backtest_logger.info(f"鎬绘墜缁 垂: ${results['total_fees']:.2f}")

        self.backtest_log_text.insert(tk.END, "="*80 + "\n\n")

        self.backtest_log_text.insert(tk.END, "[璇 粏浜 槗璁板綍]\n")
        for i, trade in enumerate(results['trades'][:50], 1):
            if trade['type'] == 'OPEN':
                self.backtest_log_text.insert(tk.END, f"{i}. [寮 浠揮 {trade['time']} - {trade['direction']} - 浠锋牸: {trade['price']:.2f} - 鏁伴噺: {trade['size']:.4f}\n")
                if hasattr(self, "backtest_logger"):
                    self.backtest_logger.info(f"{i}. [寮 浠揮 {trade['time']} - {trade['direction']} - 浠锋牸: {trade['price']:.2f} - 鏁伴噺: {trade['size']:.4f}")
            else:
                self.backtest_log_text.insert(tk.END, f"{i}. [骞充粨] {trade['time']} - {trade['direction']} - 鍏 満: {trade['entry_price']:.2f} - 鍑哄満: {trade['exit_price']:.2f} - 鐩堜簭: {trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)\n")
                if hasattr(self, "backtest_logger"):
                    self.backtest_logger.info(f"{i}. [骞充粨] {trade['time']} - {trade['direction']} - 鍏 満: {trade['entry_price']:.2f} - 鍑哄満: {trade['exit_price']:.2f} - 鐩堜簭: {trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)")

        if len(results['trades']) > 50:
            self.backtest_log_text.insert(tk.END, f"... (杩樻湁 {len(results['trades']) - 50} 绗斾氦鏄?\n")

        self.backtest_log_text.insert(tk.END, "="*80 + "\n")
        # 鍙 湁褰撴粴鍔 潯鍦 渶搴曢儴鏃舵墠鑷 姩璺熼殢
        scroll_position = self.backtest_log_text.yview()
        if scroll_position[1] >= 0.9:
            self.backtest_log_text.see(tk.END)

    def _update_progress(self, value):
        """鏇存柊杩涘害鏉?"
        self.backtest_progress["value"] = value
        self.backtest_progress_label.config(text=f"{value}%")
        self.backtest_progress.update()

    def _export_backtest_results(self):
        """瀵煎嚭鍥炴祴缁撴灉"""
        from tkinter import filedialog
        try:
            content = self.backtest_log_text.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("鎻愮 ", "娌 湁鍙  鍑虹殑鍥炴祴缁撴灉锛?)"
                return

            file_path = filedialog.asksaveasfilename(
                title="瀵煎嚭鍥炴祴缁撴灉",
                defaultextension=".txt",
                filetypes=[("鏂囨湰鏂囦欢", "*.txt"), ("鎵 鏈夋枃浠?, "*.*")]"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("鎴愬姛", "鍥炴祴缁撴灉宸插 鍑猴紒")
        except Exception as e:
            messagebox.showerror("閿欒 ", f"瀵煎嚭澶辫触: {e}")

    def _stop_backtest(self):
        """鍋滄 鍥炴祴"""
        if hasattr(self, 'backtest_stop_event'):
            self.backtest_stop_event.set()
            self.backtest_log_text.insert(tk.END, f"[鍋滄 鍥炴祴] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 姝 湪鍋滄 ...\n")
            self.stop_backtest_btn.config(state="disabled")

    def _clear_backtest_results(self):
        """娓呯 鍥炴祴缁撴灉"""
        if messagebox.askyesno("纭  ", "纭 畾瑕佹竻绌哄洖娴嬬粨鏋滃悧锛?):"
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
            self.log("鍥炴祴缁撴灉宸叉竻绌?)"

    def _get_backtest_data_dir(self):
        """鑾峰彇鍥炴祴鏁版嵁瀛樺偍鐩 綍"""
        import os
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    def _select_backtest_data_file(self):
        """閫夋嫨鍘嗗彶鏁版嵁鏂囦欢"""
        from tkinter import filedialog
        data_dir = self._get_backtest_data_dir()
        file_path = filedialog.askopenfilename(
            title="閫夋嫨鍘嗗彶鏁版嵁鏂囦欢",
            initialdir=data_dir,
            filetypes=[("CSV鏂囦欢", "*.csv"), ("鎵 鏈夋枃浠?, "*.*")]"
        )
        if file_path:
            self.backtest_data_file_var.set(file_path)

    def _refresh_backtest_data_list(self):
        """鍒锋柊鍘嗗彶鏁版嵁鏂囦欢鍒楄 """
        import os
        data_dir = self._get_backtest_data_dir()
        try:
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                self.backtest_log_text.insert(tk.END, f"[鏁版嵁鏂囦欢] 鎵惧埌 {len(files)} 涓 巻鍙叉暟鎹 枃浠?\n")
                for f in sorted(files):
                    self.backtest_log_text.insert(tk.END, f"  - {f}\n")
            else:
                self.backtest_log_text.insert(tk.END, "[鏁版嵁鏂囦欢] 鏈 壘鍒板巻鍙叉暟鎹 枃浠禱n")
            self.backtest_log_text.see(tk.END)
        except Exception as e:
            self.backtest_log_text.insert(tk.END, f"[閿欒 ] 鍒锋柊鏂囦欢鍒楄 澶辫触: {e}\n")

    def _download_historical_data(self):
        """鎸夋椂闂磋寖鍥翠笅杞藉巻鍙叉暟鎹 苟淇濆瓨锛屽寘鍚?7涓 妧鏈 寚鏍?"
        import threading
        import os
        from datetime import datetime
        from binance_api import BinanceAPI

        def download_data_thread():
            try:
                self.backtest_log_text.insert(tk.END, "[鎶撳彇鏁版嵁] 寮 濮嬭幏鍙栧巻鍙叉暟鎹?..\n")

                start_year = self.backtest_start_year_var.get().strip()
                start_month = self.backtest_start_month_var.get().strip()
                start_day = self.backtest_start_day_var.get().strip()
                end_year = self.backtest_end_year_var.get().strip()
                end_month = self.backtest_end_month_var.get().strip()
                end_day = self.backtest_end_day_var.get().strip()

                if not (start_year and start_month and start_day and end_year and end_month and end_day):
                    self.backtest_log_text.insert(tk.END, "[閿欒 ] 璇烽 夋嫨瀹屾暣鐨勫紑濮嬫椂闂村拰缁撴潫鏃堕棿锛乗n")
                    return

                start_time_str = f"{start_year}-{start_month}-{start_day} 00:00"
                end_time_str = f"{end_year}-{end_month}-{end_day} 23:59"

                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鏃堕棿鑼冨洿: {start_time_str} 鑷?{end_time_str}\n")

                if not hasattr(self, 'strategy') or not self.strategy:
                    self.backtest_log_text.insert(tk.END, "[閿欒 ] 绛栫暐鏈 垵濮嬪寲锛岃 鍏堣繍琛屼竴娆 洖娴媆n")
                    return

                self.backtest_log_text.insert(tk.END, "[鎶撳彇鏁版嵁] 姝 湪鑾峰彇K绾挎暟鎹?..\n")

                df = self.strategy.binance.get_historical_klines(
                    self.strategy.symbol,
                    self.strategy.timeframe,
                    start_str=start_time_str,
                    end_str=end_time_str
                )

                if df is None or len(df) == 0:
                    self.backtest_log_text.insert(tk.END, "[閿欒 ] 鑾峰彇鏁版嵁澶辫触\n")
                    return

                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鎴愬姛鑾峰彇 {len(df)} 鏉 師濮婯绾挎暟鎹甛n")

                self.backtest_log_text.insert(tk.END, "[鎶撳彇鏁版嵁] 姝 湪璁 畻27涓 妧鏈 寚鏍?..\n")

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

                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鎴愬姛璁 畻27涓 妧鏈 寚鏍囷紝鍓 綑 {len(df)} 鏉 湁鏁堟暟鎹甛n")

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

                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鏁版嵁宸蹭繚瀛樺埌: {file_path}\n")
                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鏂囦欢澶 皬: {os.path.getsize(file_path)} 瀛楄妭\n")
                self.backtest_log_text.insert(tk.END, f"[鎶撳彇鏁版嵁] 鍖呭惈鐗瑰緛: {len(df.columns)} 涓?(鏃堕棿鎴?+ 27涓 妧鏈 寚鏍?\n")

                self.backtest_data_file_var.set(file_path)

                messagebox.showinfo("鎴愬姛", f"鍘嗗彶鏁版嵁宸叉姄鍙栧苟淇濆瓨锛乗n鏂囦欢: {filename}\n鍖呭惈27涓 妧鏈 寚鏍?)"

            except Exception as e:
                self.backtest_log_text.insert(tk.END, f"[閿欒 ] 鎶撳彇鏁版嵁澶辫触: {e}\n")
                import traceback
                self.backtest_log_text.insert(tk.END, f"{traceback.format_exc()}\n")

        thread = threading.Thread(target=download_data_thread)
        thread.daemon = True
        thread.start()

    def _create_help_tab(self, parent):
        """鍒涘缓甯 姪鏍囩 椤?"
        try:
            # 鍒涘缓婊氬姩鏂囨湰妗?
            help_text_frame = ttk.Frame(parent)
            help_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 鍒涘缓婊氬姩鏉?
            scrollbar = ttk.Scrollbar(help_text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # 鍒涘缓鏂囨湰妗?
            help_text = tk.Text(
                help_text_frame,
                wrap=tk.WORD,
                yscrollcommand=scrollbar.set,
                font=("寰 蒋闆呴粦", 14),
                bg="#f8f9fa",
                relief=tk.FLAT,
                padx=10,
                pady=10
            )
            help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=help_text.yview)

            # 鍔犺浇甯 姪鏂囨 鍐呭
            help_file_path = os.path.join(os.path.dirname(__file__), "甯 姪.txt")
            if os.path.exists(help_file_path):
                with open(help_file_path, 'r', encoding='utf-8') as f:
                    help_content = f.read()
                    help_text.insert(tk.END, help_content)
            else:
                help_text.insert(tk.END, "甯 姪鏂囨 鏈 壘鍒帮紝璇风 淇?'甯 姪.txt' 鏂囦欢瀛樺湪銆?)"

            # 璁剧疆鏂囨湰妗嗕负鍙
            help_text.config(state=tk.DISABLED)

        except Exception as e:
            print(f"鍒涘缓甯 姪鏍囩 椤靛 璐? {e}")


def main():
    root = tk.Tk()
    root.withdraw()  # 鍏堥殣钘忕獥鍙?
    KronosTradingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

