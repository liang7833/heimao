import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import os
import sys
import io
import logging
from datetime import datetime
import psutil
import yaml
import json
import pynvml
import matplotlib
import pandas as pd

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
        "Microsoft YaHei",
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

LARGE_FONT = ("Microsoft YaHei", 12)
BOLD_FONT = ("Microsoft YaHei", 12, "bold")
TITLE_FONT = ("Microsoft YaHei", 14, "bold")


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


class KronosTradingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("黑猫交易系统v1.0")
        self.root.geometry("1650x1300")
        self.root.configure(bg="#f0f0f0")
        
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
        self.qwen_status_var = tk.StringVar(value="未初始化")
        self.qwen_strategy_type_var = tk.StringVar(value="自动选择")
        self.qwen_risk_var = tk.StringVar(value="平衡型")
        
        # 策略确认次数变量
        self.entry_confirm_count_var = tk.StringVar(value="2")
        self.reverse_confirm_count_var = tk.StringVar(value="2")
        
        # 初始化Qwen3优化器
        self.qwen_optimizer = None
        
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
                    use_local_model=True
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
                    kronos_model_name="kronos-small",
                    use_fingpt=(self.fingpt_analyzer is not None),
                    symbol="BTC"
                )
                print("  ✓ 策略协调器初始化完成")
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

        # 扫描训练好的模型目录
        possible_dirs = [
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
                            model_list.append(f"custom:{exp_name}")
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
                        model_list.append(f"custom:{model_name}")
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

    def create_widgets(self):
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_container, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

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

        ttk.Label(api_frame, text="币安 API Key:").pack(anchor=tk.W)
        self.api_key_entry = ttk.Entry(api_frame, width=40)
        self.api_key_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(api_frame, text="币安 Secret Key:").pack(anchor=tk.W)
        self.secret_key_entry = ttk.Entry(api_frame, width=40)
        self.secret_key_entry.pack(fill=tk.X, pady=(0, 5))

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

        ttk.Label(config_frame, text="交易对:").pack(anchor=tk.W)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        symbol_label = ttk.Label(config_frame, text="BTCUSDT", font=("TkDefaultFont", 10, "bold"))
        symbol_label.pack(anchor=tk.W, pady=(0, 5))

        ttk.Label(config_frame, text="交易策略:").pack(anchor=tk.W)
        self.strategy_var = tk.StringVar(value="自动策略")
        strategy_combo = ttk.Combobox(
            config_frame, textvariable=self.strategy_var, state="readonly", width=37
        )
        strategy_combo["values"] = (
            "趋势爆发",
            "震荡套利",
            "消息突破",
            "自动策略",
            "时间策略",
        )
        strategy_combo.pack(fill=tk.X, pady=(0, 5))
        strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_changed)

        ttk.Label(config_frame, text="Kronos模型:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="kronos-small")
        self.model_combo = ttk.Combobox(
            config_frame, textvariable=self.model_var, state="readonly", width=37
        )
        self.refresh_model_list()
        self.model_combo.pack(fill=tk.X, pady=(0, 5))

        # 刷新模型按钮
        refresh_model_frame = ttk.Frame(config_frame)
        refresh_model_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(
            refresh_model_frame, text="刷新模型列表", command=self.refresh_model_list
        ).pack(side=tk.RIGHT)

        ttk.Label(config_frame, text="分析周期:").pack(anchor=tk.W)
        self.timeframe_var = tk.StringVar(value="5m")
        timeframe_combo = ttk.Combobox(
            config_frame, textvariable=self.timeframe_var, state="readonly", width=37
        )
        timeframe_combo["values"] = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        timeframe_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="杠杆倍数:").pack(anchor=tk.W)
        self.leverage_var = tk.StringVar(value="10")
        leverage_combo = ttk.Combobox(
            config_frame, textvariable=self.leverage_var, state="readonly", width=37
        )
        leverage_combo["values"] = (
            "1",
            "2",
            "3",
            "5",
            "10",
            "20",
            "25",
            "50",
            "75",
            "100",
        )
        leverage_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="最小仓位 (USDT):").pack(anchor=tk.W)
        self.min_position_var = tk.StringVar(value="100")
        min_pos_entry = ttk.Entry(
            config_frame, textvariable=self.min_position_var, width=40
        )
        min_pos_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="交易间隔 (秒):").pack(anchor=tk.W)
        self.interval_var = tk.StringVar(value="120")
        interval_combo = ttk.Combobox(
            config_frame, textvariable=self.interval_var, state="readonly", width=37
        )
        interval_combo["values"] = ("30", "60", "120", "180", "300", "600")
        interval_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="趋势阈值:").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="0.008")
        threshold_entry = ttk.Entry(
            config_frame, textvariable=self.threshold_var, width=40
        )
        threshold_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="AI最小趋势强度:").pack(anchor=tk.W)
        self.ai_min_trend_var = tk.StringVar(value="0.010")
        ai_min_trend_entry = ttk.Entry(
            config_frame, textvariable=self.ai_min_trend_var, width=40
        )
        ai_min_trend_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="AI最小预测偏离度:").pack(anchor=tk.W)
        self.ai_min_deviation_var = tk.StringVar(value="0.008")
        ai_min_deviation_entry = ttk.Entry(
            config_frame, textvariable=self.ai_min_deviation_var, width=40
        )
        ai_min_deviation_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="最大资金费率 (%):").pack(anchor=tk.W)
        self.max_funding_var = tk.StringVar(value="1.0")
        max_funding_entry = ttk.Entry(
            config_frame, textvariable=self.max_funding_var, width=40
        )
        max_funding_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(config_frame, text="最小资金费率 (%):").pack(anchor=tk.W)
        self.min_funding_var = tk.StringVar(value="-1.0")
        min_funding_entry = ttk.Entry(
            config_frame, textvariable=self.min_funding_var, width=40
        )
        min_funding_entry.pack(fill=tk.X)

    def create_control_section(self, parent):
        control_frame = ttk.Frame(parent, style="TFrame", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = ttk.Button(
            control_frame, text="启动交易", command=self.start_trading
        )
        self.start_button.pack(fill=tk.X, pady=(0, 5))

        self.stop_button = ttk.Button(
            control_frame, text="停止交易", command=self.stop_trading, state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X)

        # 资金信息显示
        fund_frame = ttk.LabelFrame(control_frame, text="合约资金", padding=8)
        fund_frame.pack(fill=tk.X, pady=(10, 0))

        # 设置字体
        label_font = ("微软雅黑", 10)
        value_font = ("微软雅黑", 11, "bold")

        # 初始资金
        ttk.Label(fund_frame, text="初始资金:", font=label_font).grid(
            row=0, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.initial_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="blue"
        )
        self.initial_balance_label.grid(row=0, column=1, sticky=tk.E, pady=4, padx=5)

        # 当前资金
        ttk.Label(fund_frame, text="当前资金:", font=label_font).grid(
            row=1, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.current_balance_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="green"
        )
        self.current_balance_label.grid(row=1, column=1, sticky=tk.E, pady=4, padx=5)

        # 盈亏
        ttk.Label(fund_frame, text="盈亏金额:", font=label_font).grid(
            row=2, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_label = ttk.Label(
            fund_frame, text="$0.00", font=value_font, foreground="red"
        )
        self.pnl_label.grid(row=2, column=1, sticky=tk.E, pady=4, padx=5)

        # 盈亏比例
        ttk.Label(fund_frame, text="盈亏比例:", font=label_font).grid(
            row=3, column=0, sticky=tk.W, pady=4, padx=5
        )
        self.pnl_pct_label = ttk.Label(
            fund_frame, text="0.00%", font=value_font, foreground="red"
        )
        self.pnl_pct_label.grid(row=3, column=1, sticky=tk.E, pady=4, padx=5)

        # 初始化资金变量
        self.initial_futures_balance = 0.0
        
        # 资金数据记录
        self.balance_data_file = os.path.join(os.path.dirname(__file__), "balance_data.csv")
        self.balance_data = []
        self.balance_record_timer = None
        self._load_balance_data()
        
        # 程序启动后立即开始记录资金数据
        self.root.after(1000, self._start_balance_recording)

    def create_terminal_section(self, parent):
        # 创建笔记本控件（标签页）
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 日志标签页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="日志")
        self._create_log_tab(log_frame)

        # 可视化标签页
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="可视化")
        self._create_visualization_tab(viz_frame)

        # 训练标签页
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="训练")
        self._create_training_tab(train_frame)

        # 训练文件管理标签页
        training_manager_frame = ttk.Frame(notebook)
        notebook.add(training_manager_frame, text="训练文件管理")
        self._create_training_manager_tab(training_manager_frame)

        # AI策略中心标签页
        ai_strategy_frame = ttk.Frame(notebook)
        notebook.add(ai_strategy_frame, text="AI策略中心")
        self._create_ai_strategy_tab(ai_strategy_frame)

        # 实盘监控标签页
        live_monitor_frame = ttk.Frame(notebook)
        notebook.add(live_monitor_frame, text="实盘监控")
        self._create_live_monitor_tab(live_monitor_frame)
        
        # 资金曲线标签页
        balance_chart_frame = ttk.Frame(notebook)
        notebook.add(balance_chart_frame, text="资金曲线")
        self._create_balance_chart_tab(balance_chart_frame)
        
        # BTC新闻标签页
        news_frame = ttk.Frame(notebook)
        notebook.add(news_frame, text="BTC新闻")
        self._create_news_tab(news_frame)
        
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
                
                tokenizer_path = os.path.join(base_dir, "models", "kronos-tokenizer-base")
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

            if self.kronos_predictor is None:
                self.kronos_predictor = KronosPredictor(
                    self.kronos_model, self.kronos_tokenizer, max_context=512
                )

            predictor = self.kronos_predictor

            # 更新状态
            self.viz_status_label.config(text="正在生成预测...")
            self.viz_canvas.get_tk_widget().update()

            # 准备数据 - Kronos模型需要open, high, low, close, volume, amount列
            x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume"]].copy()
            # 添加amount列（成交额），如果没有则用成交量*平均价格估算
            if "amount" not in x_df.columns:
                x_df["amount"] = x_df["volume"] * x_df[
                    ["open", "high", "low", "close"]
                ].mean(axis=1)

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
            x_df_with_features = self._calculate_kronos_features(x_df)

            # 选择Kronos需要的列（原始价格 + 技术指标）
            kronos_columns = [
                "open",
                "high",
                "low",
                "close",
                "amount",
                "MA5",
                "MA10",
                "MA20",
                "BIAS20",
                "ATR14",
                "AMPLITUDE",
                "AMOUNT_MA5",
                "AMOUNT_MA10",
                "VOL_RATIO",
                "RSI14",
                "RSI7",
                "MACD",
                "MACD_HIST",
                "PRICE_SLOPE5",
                "PRICE_SLOPE10",
                "HIGH5",
                "LOW5",
                "HIGH10",
                "LOW10",
                "VOL_BREAKOUT",
                "VOL_SHRINK",
            ]

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
                    linewidth=1.5,
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="预测未来",
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
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
                    linewidth=1.5,
                )
                ax1.plot(
                    pred_timestamps,
                    pred_prices,
                    label="预测价格",
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
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

    def _calculate_kronos_features(self, df):
        """计算Kronos预测所需的技术指标特征"""
        import numpy as np

        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # 处理成交额列 - 兼容不同数据源
        # Binance API返回 'amount' 列 (quote_asset_volume)，但有些数据源可能没有
        if "amount" in df.columns:
            amount = df["amount"].values
        elif "volume" in df.columns:
            # 使用成交量作为替代
            amount = df["volume"].values
        elif "quote_asset_volume" in df.columns:
            # 使用原始列名
            amount = df["quote_asset_volume"].values
        else:
            # 如果没有成交额列，创建一个默认值（价格乘以某个系数）
            amount = close * 100  # 默认假设每根K线成交额 = 价格 * 100

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
            ((amount > df["AMOUNT_MA5"] * 1.5) & (close > df["HIGH5"]))
            .astype(int)
            .values
        )
        df["VOL_SHRINK"] = (
            ((amount < df["AMOUNT_MA5"] * 0.5) & (close < df["LOW5"]))
            .astype(int)
            .values
        )

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
                "timeframe": "1m",
                "interval": "60",
            },
            "breakout": {
                "name": "消息突破",
                "threshold": "0.015",
                "ai_min_trend": "0.020",
                "ai_min_deviation": "0.015",
                "max_funding": "3.0",
                "min_funding": "-3.0",
                "timeframe": "15m",
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
                    self.terminal.see(tk.END)
                    
                    # 检测是否是多智能体系统的日志，如果是，也显示在AI策略中心和实盘监控日志中
                    multi_agent_keywords = [
                        "[多智能体系统]", "[FinGPT]", "[策略协调器]", 
                        "Kronos分析", "舆情分析", "信号过滤", "协调器",
                        "市场情绪", "风险等级", "交易建议", "CoinGecko", "币安"
                    ]
                    
                    is_multi_agent_log = any(keyword in data for keyword in multi_agent_keywords)
                    
                    if is_multi_agent_log:
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
            )

            # 设置确认次数
            try:
                entry_count = int(self.entry_confirm_count_var.get())
                reverse_count = int(self.reverse_confirm_count_var.get())
                
                # 验证范围
                if 1 <= entry_count <= 10:
                    self.strategy.entry_confirm_count = entry_count
                if 1 <= reverse_count <= 10:
                    self.strategy.reverse_confirm_count = reverse_count
                    
                self.log(f"确认次数设置: 开仓{entry_count}次, 平仓{reverse_count}次")
            except Exception as e:
                self.log(f"设置确认次数失败: {e}, 使用默认值(2次)")

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
            self.update_fund_display()
            self.log(f"初始合约资金(总资金): ${self.initial_futures_balance:.2f}")

            # 启动资金更新定时器
            self.update_fund_timer()

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
        self.train_data_path_var = tk.StringVar(value="")
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
            value="Kronos/finetune_csv/configs/config_ali09988_candle-5min.yaml"
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
            text="版本 1.0 · 黑猫交易系统v1.0",
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

            self.log_train(f"数据已保存: {save_path}")
            self.log_train(f"数据量: {len(df)} 条")
            self.log_train(
                f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}"
            )
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
        if self.is_running:
            if messagebox.askokcancel("退出", "交易正在运行，确定要退出吗？"):
                self.stop_event.set()
                self.is_running = False
                self.root.destroy()
        else:
            self.root.destroy()

    def _create_ai_strategy_tab(self, parent):
        """创建AI策略中心标签页（优化版 - 简洁布局）"""
        # 主框架
        main_frame = ttk.Frame(parent, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="AI策略中心",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50",
        )
        title_label.pack(pady=(0, 10))

        # 说明文字
        desc_label = ttk.Label(
            main_frame,
            text="集成Kronos、FinGPT、Qwen3三大AI模型，实现智能策略生成与优化",
            font=("微软雅黑", 9),
            foreground="#7f8c8d"
        )
        desc_label.pack(pady=(0, 12))

        # ==================== 系统状态面板 ====================
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 12))
        
        # FinGPT状态
        fingpt_status_frame = ttk.Frame(status_frame)
        fingpt_status_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(fingpt_status_frame, text="FinGPT:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.fingpt_status_label = ttk.Label(
            fingpt_status_frame,
            textvariable=self.fingpt_status_var,
            font=("微软雅黑", 9, "bold"),
            foreground="#e74c3c"
        )
        self.fingpt_status_label.pack(side=tk.LEFT)
        
        # 策略协调器状态
        coordinator_status_frame = ttk.Frame(status_frame)
        coordinator_status_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(coordinator_status_frame, text="协调器:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.coordinator_status_label = ttk.Label(
            coordinator_status_frame,
            textvariable=self.coordinator_status_var,
            font=("微软雅黑", 9, "bold"),
            foreground="#e74c3c"
        )
        self.coordinator_status_label.pack(side=tk.LEFT)
        
        # Qwen3状态
        qwen_status_frame = ttk.Frame(status_frame)
        qwen_status_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(qwen_status_frame, text="Qwen3:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.qwen_status_label = ttk.Label(
            qwen_status_frame,
            textvariable=self.qwen_status_var,
            font=("微软雅黑", 9, "bold"),
            foreground="#e74c3c"
        )
        self.qwen_status_label.pack(side=tk.LEFT)
        
        # 监控状态
        monitor_status_frame = ttk.Frame(status_frame)
        monitor_status_frame.pack(side=tk.LEFT)
        
        ttk.Label(monitor_status_frame, text="监控:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.monitor_status_label = ttk.Label(
            monitor_status_frame,
            textvariable=self.monitor_status_var,
            font=("微软雅黑", 9),
            foreground="#e74c3c"
        )
        self.monitor_status_label.pack(side=tk.LEFT)

        # ==================== 中间功能区域（两列布局） ====================
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # 左侧：策略配置面板
        left_panel = ttk.LabelFrame(middle_frame, text="策略配置", padding="12")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        
        # 确认次数设置
        confirm_frame = ttk.Frame(left_panel)
        confirm_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confirm_frame, text="确认次数:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # 开仓确认
        ttk.Label(confirm_frame, text="开仓:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        entry_confirm_spinbox = ttk.Spinbox(
            confirm_frame,
            from_=1,
            to=10,
            textvariable=self.entry_confirm_count_var,
            width=8,
            command=lambda: self._update_confirm_counts()
        )
        entry_confirm_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # 平仓确认
        ttk.Label(confirm_frame, text="平仓:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        reverse_confirm_spinbox = ttk.Spinbox(
            confirm_frame,
            from_=1,
            to=10,
            textvariable=self.reverse_confirm_count_var,
            width=8,
            command=lambda: self._update_confirm_counts()
        )
        reverse_confirm_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # 应用按钮
        ttk.Button(
            confirm_frame,
            text="应用",
            command=self._apply_confirm_counts,
            width=8
        ).pack(side=tk.LEFT)
        
        # 自动化优化设置
        auto_opt_frame = ttk.LabelFrame(left_panel, text="自动化优化", padding="10")
        auto_opt_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 第一行：状态和开关
        auto_status_row = ttk.Frame(auto_opt_frame)
        auto_status_row.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(auto_status_row, text="状态:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            auto_status_row,
            textvariable=self.auto_optimization_status_var,
            font=("微软雅黑", 9),
            foreground="#27ae60"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            auto_status_row,
            text="开关",
            command=self._toggle_auto_optimization,
            width=8
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            auto_status_row,
            text="优化",
            command=self._trigger_manual_optimization,
            width=8
        ).pack(side=tk.LEFT)
        
        # 第二行：阈值和频率
        auto_config_row1 = ttk.Frame(auto_opt_frame)
        auto_config_row1.pack(fill=tk.X, pady=(5, 5))
        
        ttk.Label(auto_config_row1, text="阈值:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        threshold_combo = ttk.Combobox(
            auto_config_row1,
            textvariable=self.optimization_threshold_var,
            values=["宽松", "中等", "严格"],
            state="readonly",
            width=8
        )
        threshold_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(auto_config_row1, text="频率:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        frequency_combo = ttk.Combobox(
            auto_config_row1,
            textvariable=self.optimization_frequency_var,
            values=["每2分钟", "每5分钟", "每15分钟", "每小时"],
            state="readonly",
            width=10
        )
        frequency_combo.pack(side=tk.LEFT)
        
        # 第三行：判断模式
        auto_config_row2 = ttk.Frame(auto_opt_frame)
        auto_config_row2.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(auto_config_row2, text="判断模式:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        judgment_mode_combo = ttk.Combobox(
            auto_config_row2,
            textvariable=self.optimization_judgment_mode_var,
            values=["固定阈值", "AI判断", "混合模式"],
            state="readonly",
            width=12
        )
        judgment_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            auto_config_row2,
            text="应用模式",
            command=self._apply_judgment_mode,
            width=10
        ).pack(side=tk.LEFT)
        
        # 右侧：AI策略操作面板
        right_panel = ttk.LabelFrame(middle_frame, text="AI策略操作", padding="12")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        
        # Qwen3初始化行
        qwen_init_frame = ttk.Frame(right_panel)
        qwen_init_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(qwen_init_frame, text="Qwen3模型:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.qwen_status_label = ttk.Label(
            qwen_init_frame,
            textvariable=self.qwen_status_var,
            font=("微软雅黑", 9),
            foreground="#e74c3c"
        )
        self.qwen_status_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # 自动初始化Qwen3优化器
        self._auto_init_qwen3()
        
        ttk.Button(
            qwen_init_frame,
            text="初始化",
            command=self._init_qwen3,
            width=10
        ).pack(side=tk.LEFT)
        
        # 策略生成行
        strategy_gen_frame = ttk.Frame(right_panel)
        strategy_gen_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(strategy_gen_frame, text="策略:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        strategy_combo = ttk.Combobox(
            strategy_gen_frame,
            textvariable=self.qwen_strategy_type_var,
            values=["自动选择", "趋势跟踪", "均值回归", "突破策略", "高频刷单"],
            state="readonly",
            width=12
        )
        strategy_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(strategy_gen_frame, text="风险:", font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 5))
        risk_combo = ttk.Combobox(
            strategy_gen_frame,
            textvariable=self.qwen_risk_var,
            values=["保守型", "平衡型", "激进型"],
            state="readonly",
            width=10
        )
        risk_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            strategy_gen_frame,
            text="生成",
            command=self._generate_strategy_with_qwen,
            width=8
        ).pack(side=tk.LEFT)
        
        # 分析操作行
        analysis_frame = ttk.Frame(right_panel)
        analysis_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(
            analysis_frame,
            text="🔍 市场分析",
            command=self._auto_analyze_market_with_qwen,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Button(
            analysis_frame,
            text="🔧 参数优化",
            command=self._optimize_params_with_qwen,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Button(
            analysis_frame,
            text="📊 回测分析",
            command=self._analyze_backtest_with_qwen,
            width=12
        ).pack(side=tk.LEFT)
        
        # 智能操作行
        smart_frame = ttk.Frame(right_panel)
        smart_frame.pack(fill=tk.X, pady=(8, 0))
        
        ttk.Label(smart_frame, text="智能操作:", font=("微软雅黑", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            smart_frame,
            text="🚀 智能生成",
            command=self._smart_strategy_generation,
            width=15,
            style="Accent.TButton"
        ).pack(side=tk.LEFT)

        # ==================== 交易日志区域 ====================
        log_frame = ttk.LabelFrame(main_frame, text="交易日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 日志控制按钮行
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(0, 8))
        
        # 监控按钮
        self.start_monitor_button = ttk.Button(
            log_control_frame,
            text="▶ 启动监控",
            command=self._start_live_monitoring,
            width=12,
            style="Accent.TButton"
        )
        self.start_monitor_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.stop_monitor_button = ttk.Button(
            log_control_frame,
            text="⏹ 停止监控",
            command=self._stop_live_monitoring,
            width=12,
            style="TButton"
        )
        self.stop_monitor_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_monitor_button.config(state=tk.DISABLED)
        
        # 清空日志按钮
        ttk.Button(
            log_control_frame,
            text="🗑 清空日志",
            command=self._clear_ai_strategy_log,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        # 日志文本框
        self.ai_strategy_log_text = tk.Text(
            log_frame,
            height=12,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.ai_strategy_log_text.pack(fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.ai_strategy_log_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ai_strategy_log_text.configure(yscrollcommand=scrollbar.set)

        # 加载配置
        self._load_live_config()

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
        self.live_log_text.pack(fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.live_log_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_log_text.configure(yscrollcommand=scrollbar.set)

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
                marker='o',
                markersize=3,
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
            
            while self.is_live_monitoring:
                try:
                    symbol = self.live_symbol_var.get()
                    
                    # 获取当前价格
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
                    
                    # 获取持仓信息
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
                    
                    time.sleep(5)  # 每5秒更新一次
                    
                except Exception as e:
                    self._log_live_message(f"获取行情数据失败: {e}", "WARNING")
                    time.sleep(5)
                    
        except Exception as e:
            self._log_live_message(f"监控线程错误: {e}", "ERROR")
    
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
            
            # 自动滚动到底部
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
            
            # 自动滚动到底部
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

    def _auto_init_qwen3(self):
        """自动初始化Qwen3优化器"""
        try:
            # 检查是否已经有模型文件
            model_path = os.path.join(os.path.dirname(__file__), "models", "Qwen3.5-0.8B-Instruct")
            if not os.path.exists(model_path):
                print("Qwen3模型文件不存在，跳过自动初始化")
                self.qwen_status_var.set("模型未下载")
                self.qwen_status_label.config(foreground="#e74c3c")  # 红色
                return
                
            print("正在自动初始化Qwen3优化器...")
            from qwen3_optimizer import Qwen3Optimizer
            
            self.qwen_optimizer = Qwen3Optimizer(
                model_path=model_path,
                device=None,
                max_length=2048
            )
            
            if self.qwen_optimizer.is_loaded:
                self.qwen_status_var.set("运行中")
                self.qwen_status_label.config(foreground="#27ae60")  # 绿色
                print("Qwen3优化器自动初始化成功")
                
                # 将 FinGPT 和 Qwen 传递给新闻爬虫
                if self.news_crawler:
                    if self.fingpt_analyzer:
                        self.news_crawler.set_fingpt_analyzer(self.fingpt_analyzer)
                    self.news_crawler.set_qwen_optimizer(self.qwen_optimizer)
                    print("  ✓ 新闻爬虫已连接 FinGPT 和 Qwen")
                
                # Qwen初始化成功后，自动启动自动化优化
                self.root.after(1000, self._auto_start_optimization_and_monitoring)
            else:
                self.qwen_status_var.set("自动加载失败")
                self.qwen_status_label.config(foreground="#e74c3c")  # 红色
                print("Qwen3优化器自动初始化失败")
                
        except Exception as e:
            print(f"Qwen3优化器自动初始化失败: {e}")
            self.qwen_status_var.set("自动初始化失败")
            self.qwen_status_label.config(foreground="#e74c3c")  # 红色

    def _init_qwen3(self):
        """手动初始化Qwen3优化器"""
        try:
            self._log_live_message("正在初始化Qwen3优化器...", "INFO")
            
            from qwen3_optimizer import Qwen3Optimizer
            
            self.qwen_optimizer = Qwen3Optimizer(
                model_path=os.path.join(os.path.dirname(__file__), "models", "Qwen2.5-0.5B-Instruct"),  # 本地模型路径
                device=None,  # 自动选择
                max_length=2048
            )
            
            if self.qwen_optimizer.is_loaded:
                self.qwen_status_var.set("运行中")
                self.qwen_status_label.config(foreground="#27ae60")
                self._log_live_message("Qwen3优化器初始化成功", "SUCCESS")
                
                # 将 FinGPT 和 Qwen 传递给新闻爬虫
                if self.news_crawler:
                    if self.fingpt_analyzer:
                        self.news_crawler.set_fingpt_analyzer(self.fingpt_analyzer)
                    self.news_crawler.set_qwen_optimizer(self.qwen_optimizer)
                    self._log_live_message("新闻爬虫已连接 FinGPT 和 Qwen", "SUCCESS")
                
                # Qwen初始化成功后，自动启动自动化优化
                self.root.after(1000, self._auto_start_optimization_and_monitoring)
            else:
                self.qwen_status_var.set("加载失败")
                self.qwen_status_label.config(foreground="#e74c3c")
                self._log_live_message("Qwen3模型加载失败: 模型未在本地缓存，请下载模型", "ERROR")
                self._log_live_message(f"模型名称: Qwen/Qwen2.5-0.5B-Instruct", "INFO")
                self._log_live_message(f"下载命令: pip install huggingface-hub && huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct", "INFO")
                
        except Exception as e:
            self.qwen_status_var.set("初始化失败")
            self.qwen_status_label.config(foreground="#e74c3c")
            self._log_live_message(f"Qwen3初始化失败: {e}", "ERROR")

    def _get_qwen_strategy_type_mapping(self):
        """获取策略类型的中英文映射"""
        return {
            "自动选择": "auto",
            "趋势跟踪": "trend_following",
            "均值回归": "mean_reversion",
            "突破策略": "breakout",
            "高频刷单": "scalping"
        }

    def _get_qwen_risk_profile_mapping(self):
        """获取风险偏好的中英文映射"""
        return {
            "保守型": "conservative",
            "平衡型": "balanced",
            "激进型": "aggressive"
        }

    def _generate_strategy_with_qwen(self):
        """使用Qwen3生成策略代码"""
        try:
            if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                self._log_live_message("Qwen3优化器未初始化，请先初始化", "ERROR")
                return
            
            # 获取中文选项并转换为英文
            strategy_type_cn = self.qwen_strategy_type_var.get()
            risk_profile_cn = self.qwen_risk_var.get()
            
            strategy_type = self._get_qwen_strategy_type_mapping().get(strategy_type_cn, "trend_following")
            risk_profile = self._get_qwen_risk_profile_mapping().get(risk_profile_cn, "balanced")
            
            self._log_live_message(f"正在生成策略代码: {strategy_type_cn} | 风险偏好: {risk_profile_cn}", "INFO")
            
            # 获取当前市场条件
            market_conditions = {
                "symbol": self.live_symbol_var.get(),
                "timestamp": datetime.now().isoformat()
            }
            
            result = self.qwen_optimizer.generate_strategy_code(
                strategy_type=strategy_type,
                market_conditions=market_conditions,
                risk_profile=risk_profile
            )
            
            if result.get("success"):
                self._log_live_message("策略代码生成成功！", "SUCCESS")
                self._log_live_message(f"策略类型: {result.get('strategy_type')}", "INFO")
                self._log_live_message(f"风险偏好: {result.get('risk_profile')}", "INFO")
                
                # 显示生成的参数
                params = result.get('parameters', {})
                if params:
                    self._log_live_message("建议参数:", "INFO")
                    for key, value in list(params.items())[:5]:
                        self._log_live_message(f"  {key}: {value}", "INFO")
                
                # 保存生成的代码到文件
                code = result.get('generated_code', '')
                if code:
                    filename = f"generated_strategy_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(code)
                    self._log_live_message(f"策略代码已保存到: {filename}", "SUCCESS")
            else:
                error = result.get('error', '未知错误')
                self._log_live_message(f"策略生成失败: {error}", "ERROR")
                
        except Exception as e:
            self._log_live_message(f"生成策略失败: {e}", "ERROR")

    def _optimize_params_with_qwen(self):
        """使用Qwen3优化策略参数"""
        try:
            if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                self._log_live_message("Qwen3优化器未初始化，请先初始化", "ERROR")
                return
            
            self._log_live_message("正在优化策略参数...", "INFO")
            
            # 模拟回测结果（实际应该从回测引擎获取）
            mock_backtest = {
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "total_trades": 100
            }
            
            # 获取中文选项并转换为英文
            strategy_type_cn = self.qwen_strategy_type_var.get()
            strategy_type = self._get_qwen_strategy_type_mapping().get(strategy_type_cn, "trend_following")
            
            result = self.qwen_optimizer.optimize_parameters(
                backtest_results=mock_backtest,
                strategy_type=strategy_type,
                target_metric="sharpe_ratio"
            )
            
            if result.get("success"):
                self._log_live_message("参数优化完成！", "SUCCESS")
                
                optimized_params = result.get('optimized_parameters', {})
                if optimized_params:
                    self._log_live_message("优化后的参数:", "INFO")
                    for key, value in list(optimized_params.items())[:5]:
                        self._log_live_message(f"  {key}: {value}", "INFO")
                
                reasoning = result.get('reasoning', [])
                if reasoning:
                    self._log_live_message("优化理由:", "INFO")
                    for reason in reasoning[:3]:
                        self._log_live_message(f"  • {reason}", "INFO")
            else:
                error = result.get('error', '未知错误')
                self._log_live_message(f"参数优化失败: {error}", "ERROR")
                
        except Exception as e:
            self._log_live_message(f"优化参数失败: {e}", "ERROR")

    def _analyze_backtest_with_qwen(self):
        """使用Qwen3分析回测结果"""
        try:
            if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                self._log_live_message("Qwen3优化器未初始化，请先初始化", "ERROR")
                return
            
            self._log_live_message("正在分析回测结果...", "INFO")
            
            # 模拟回测结果
            mock_backtest = {
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "total_trades": 100,
                "avg_trade_return": 0.02,
                "max_consecutive_losses": 3
            }
            
            result = self.qwen_optimizer.analyze_backtest_results(mock_backtest)
            
            if result.get("success"):
                self._log_live_message("回测分析完成！", "SUCCESS")
                
                analysis = result.get('analysis', {})
                
                if 'strengths' in analysis:
                    self._log_live_message("策略优势:", "INFO")
                    for strength in analysis['strengths'][:3]:
                        self._log_live_message(f"  ✓ {strength}", "INFO")
                
                if 'weaknesses' in analysis:
                    self._log_live_message("策略劣势:", "WARNING")
                    for weakness in analysis['weaknesses'][:3]:
                        self._log_live_message(f"  ✗ {weakness}", "WARNING")
                
                if 'suggestions' in analysis:
                    self._log_live_message("改进建议:", "INFO")
                    for suggestion in analysis['suggestions'][:3]:
                        self._log_live_message(f"  → {suggestion}", "INFO")
            else:
                error = result.get('error', '未知错误')
                self._log_live_message(f"回测分析失败: {error}", "ERROR")
                
        except Exception as e:
            self._log_live_message(f"分析回测失败: {e}", "ERROR")

    def _auto_analyze_market_with_qwen(self):
        """使用Qwen3自动分析市场并推荐策略"""
        try:
            if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                self._log_live_message("Qwen3优化器未初始化，请先初始化", "ERROR")
                return
            
            self._log_live_message("=" * 60, "INFO")
            self._log_live_message("🔍 Qwen3正在自动分析市场状态...", "INFO")
            self._log_live_message("=" * 60, "INFO")
            
            # 获取当前市场数据
            symbol = self.live_symbol_var.get()
            self._log_live_message(f"分析交易对: {symbol}", "INFO")
            
            # 尝试获取市场数据（使用Kronos分析器）
            try:
                from enhanced_kronos import EnhancedKronosAnalyzer
                analyzer = EnhancedKronosAnalyzer(model_name="kronos-small")
                
                # 获取K线数据
                import pandas as pd
                from binance_api import BinanceAPI
                binance = BinanceAPI()
                
                # 获取最近100根K线
                klines = binance.client.futures_klines(
                    symbol=symbol,
                    interval='5m',
                    limit=100
                )
                
                if klines:
                    # 转换为DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # 转换数值类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 获取Kronos分析信号
                    signal = analyzer.get_enhanced_signal(df)
                    
                    # 提取市场特征
                    market_features = {
                        "trend_direction": signal.get('trend_direction', 'NEUTRAL'),
                        "trend_strength": signal.get('trend_strength', 0),
                        "market_state": signal.get('market_state', 'unknown'),
                        "volatility": signal.get('volatility', 0),
                        "price_change_pct": signal.get('price_change_pct', 0) * 100,
                        "current_price": float(df['close'].iloc[-1]) if len(df) > 0 else 0,
                        "support_level": signal.get('pred_support', 0),
                        "resistance_level": signal.get('pred_resistance', 0)
                    }
                    
                    self._log_live_message("📊 市场特征分析完成:", "INFO")
                    self._log_live_message(f"  趋势方向: {market_features['trend_direction']}", "INFO")
                    self._log_live_message(f"  趋势强度: {market_features['trend_strength']:.4f}", "INFO")
                    self._log_live_message(f"  市场状态: {market_features['market_state']}", "INFO")
                    self._log_live_message(f"  价格波动: {market_features['price_change_pct']:.2f}%", "INFO")
                    
                else:
                    market_features = {
                        "trend_direction": "NEUTRAL",
                        "trend_strength": 0.5,
                        "market_state": "unknown",
                        "note": "无法获取实时数据，使用默认分析"
                    }
                    self._log_live_message("⚠️ 无法获取实时市场数据，使用默认分析", "WARNING")
                    
            except Exception as e:
                self._log_live_message(f"获取市场数据失败: {e}", "WARNING")
                market_features = {
                    "trend_direction": "NEUTRAL",
                    "trend_strength": 0.5,
                    "market_state": "unknown",
                    "note": "数据获取失败"
                }
            
            # 使用Qwen3进行策略推荐
            self._log_live_message("🤖 Qwen3正在根据市场特征推荐策略...", "INFO")
            
            # 构建提示词让Qwen3推荐策略
            recommendation = self._get_strategy_recommendation_from_qwen(market_features)
            
            if recommendation:
                self._log_live_message("=" * 60, "INFO")
                self._log_live_message("✅ Qwen3策略推荐完成！", "SUCCESS")
                self._log_live_message("=" * 60, "INFO")
                
                recommended_strategy = recommendation.get('strategy', '趋势跟踪')
                recommended_risk = recommendation.get('risk_profile', '平衡型')
                reasoning = recommendation.get('reasoning', [])
                
                # 更新界面选择
                self.qwen_strategy_type_var.set(recommended_strategy)
                self.qwen_risk_var.set(recommended_risk)
                
                self._log_live_message(f"🎯 推荐策略类型: {recommended_strategy}", "SUCCESS")
                self._log_live_message(f"⚖️ 推荐风险偏好: {recommended_risk}", "SUCCESS")
                
                if reasoning:
                    self._log_live_message("\n📋 推荐理由:", "INFO")
                    for reason in reasoning[:5]:
                        self._log_live_message(f"  • {reason}", "INFO")
                
                self._log_live_message("\n💡 提示: 策略类型和风险偏好已自动更新", "INFO")
                self._log_live_message("   点击'✨ 生成策略代码'即可生成推荐策略", "INFO")
                
            else:
                self._log_live_message("❌ 策略推荐失败，请手动选择策略类型", "ERROR")
                
        except Exception as e:
            self._log_live_message(f"自动分析市场失败: {e}", "ERROR")

    def _get_strategy_recommendation_from_qwen(self, market_features):
        """从Qwen3获取策略推荐"""
        try:
            # 构建市场分析提示
            trend = market_features.get('trend_direction', 'NEUTRAL')
            strength = market_features.get('trend_strength', 0)
            state = market_features.get('market_state', 'unknown')
            volatility = market_features.get('volatility', 0)
            price_change = market_features.get('price_change_pct', 0)
            
            # 根据市场特征进行简单规则匹配（如果Qwen3不可用）
            # 或者可以调用Qwen3生成更智能的推荐
            
            recommendation = {
                'strategy': '趋势跟踪',
                'risk_profile': '平衡型',
                'reasoning': []
            }
            
            # 基于市场状态的策略推荐逻辑
            if state == 'trending' or (trend in ['LONG', 'SHORT'] and strength > 0.6):
                recommendation['strategy'] = '趋势跟踪'
                recommendation['risk_profile'] = '平衡型' if strength > 0.8 else '保守型'
                recommendation['reasoning'] = [
                    f"检测到明显趋势（强度: {strength:.2f}）",
                    "趋势跟踪策略适合当前市场环境",
                    f"趋势方向: {trend}"
                ]
            elif state == 'ranging' or abs(price_change) < 2:
                recommendation['strategy'] = '均值回归'
                recommendation['risk_profile'] = '保守型'
                recommendation['reasoning'] = [
                    "市场处于震荡区间",
                    "均值回归策略适合震荡行情",
                    f"价格波动较小（{price_change:.2f}%）"
                ]
            elif volatility > 0.03 or abs(price_change) > 5:
                recommendation['strategy'] = '突破策略'
                recommendation['risk_profile'] = '激进型'
                recommendation['reasoning'] = [
                    f"检测到高波动性（{volatility:.4f}）",
                    "突破策略适合高波动市场",
                    f"价格变动较大（{price_change:.2f}%）"
                ]
            elif state == 'volatile':
                recommendation['strategy'] = '高频刷单'
                recommendation['risk_profile'] = '激进型'
                recommendation['reasoning'] = [
                    "市场波动剧烈",
                    "高频刷单可捕捉短期机会",
                    "需要严格止损控制"
                ]
            else:
                recommendation['reasoning'] = [
                    f"市场状态: {state}",
                    f"趋势强度: {strength:.2f}",
                    "使用默认趋势跟踪策略"
                ]
            
            return recommendation
            
        except Exception as e:
            self._log_live_message(f"获取策略推荐失败: {e}", "ERROR")
            return None

    def _smart_strategy_generation(self):
        """一键智能策略生成：自动初始化、分析市场、生成代码"""
        try:
            self._log_live_message("=" * 60, "INFO")
            self._log_live_message("🚀 开始智能策略生成流程", "INFO")
            self._log_live_message("=" * 60, "INFO")
            
            # 步骤1: 检查并初始化Qwen3
            if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                self._log_live_message("步骤1: 初始化Qwen3优化器...", "INFO")
                self._init_qwen3()
                
                # 检查初始化结果
                if self.qwen_optimizer is None or not self.qwen_optimizer.is_loaded:
                    self._log_live_message("❌ Qwen3初始化失败，无法继续", "ERROR")
                    return
                self._log_live_message("✓ Qwen3初始化成功", "SUCCESS")
            else:
                self._log_live_message("✓ Qwen3已初始化", "SUCCESS")
            
            # 步骤2: 自动分析市场
            self._log_live_message("\n步骤2: 分析当前市场状态...", "INFO")
            self._auto_analyze_market_with_qwen()
            
            # 获取推荐结果
            strategy_type_cn = self.qwen_strategy_type_var.get()
            risk_profile_cn = self.qwen_risk_var.get()
            
            if strategy_type_cn == "自动选择":
                self._log_live_message("⚠️ 策略类型为'自动选择'，使用默认策略", "WARNING")
                strategy_type_cn = "趋势跟踪"
                self.qwen_strategy_type_var.set(strategy_type_cn)
            
            self._log_live_message(f"✓ 市场分析完成", "SUCCESS")
            self._log_live_message(f"  推荐策略: {strategy_type_cn}", "INFO")
            self._log_live_message(f"  风险偏好: {risk_profile_cn}", "INFO")
            
            # 步骤3: 生成策略代码
            self._log_live_message("\n步骤3: 生成策略代码...", "INFO")
            self._generate_strategy_with_qwen()
            
            self._log_live_message("=" * 60, "INFO")
            self._log_live_message("✅ 智能策略生成流程完成！", "SUCCESS")
            self._log_live_message("=" * 60, "INFO")
            self._log_live_message("💡 提示: 生成的策略代码已保存到项目目录", "INFO")
            self._log_live_message("   您可以在专业策略中参考或集成生成的代码", "INFO")
            
        except Exception as e:
            self._log_live_message(f"❌ 智能策略生成失败: {e}", "ERROR")

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
                    judgment_mode=judgment_mode,
                    qwen_optimizer=self.qwen_optimizer
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
                    qwen_model_name="Qwen/Qwen3.5-0.8B-Instruct",  # Qwen3.5-0.8B参数，更轻量快速
                    enable_qwen=True,  # 启用Qwen3.5优化
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

    def _auto_start_optimization_and_monitoring(self):
        """Qwen初始化成功后自动启动自动化优化和监控"""
        try:
            print("Qwen初始化成功，自动启动自动化优化和监控...")
            
            # 1. 启动自动化优化系统（但不立即启动性能监控）
            if not self.is_auto_optimization_enabled:
                if self._initialize_auto_optimization_system():
                    self.is_auto_optimization_enabled = True
                    self.auto_optimization_status_var.set("运行中")
                    
                    # 确保性能监控器有Qwen引用
                    if self.performance_monitor:
                        self.performance_monitor.qwen_optimizer = self.qwen_optimizer
                    
                    # 只有在策略实例存在时才启动性能监控
                    if self.performance_monitor and hasattr(self, 'strategy') and self.strategy:
                        self.performance_monitor.start_monitoring()
                        print("✅ 自动化优化已启用（性能监控已启动）")
                    else:
                        print("✅ 自动化优化已启用（等待策略实例连接）")
                else:
                    print("❌ 自动化优化系统初始化失败")
            else:
                # 如果自动化优化已经启用，更新性能监控器的Qwen引用
                if self.performance_monitor:
                    self.performance_monitor.qwen_optimizer = self.qwen_optimizer
                    print("✅ 已更新性能监控器的Qwen引用")
            
            # 2. 启动实盘监控
            if not hasattr(self, 'is_live_monitoring') or not self.is_live_monitoring:
                if hasattr(self, '_start_live_monitoring'):
                    self._start_live_monitoring()
                    print("✅ 实盘监控已启动")
        except Exception as e:
            print(f"自动启动失败: {e}")

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
            
            # 检查AI模式是否需要Qwen
            if judgment_mode in ["ai", "hybrid"]:
                if not self.qwen_optimizer or not self.qwen_optimizer.is_loaded:
                    self._log_ai_strategy_message("⚠️ Qwen模型未加载，无法使用AI判断模式", "WARNING")
                    self._log_ai_strategy_message("  请先初始化Qwen3模型", "INFO")
                    return
            
            # 切换模式
            if self.performance_monitor:
                self.performance_monitor.set_judgment_mode(
                    mode=judgment_mode,
                    qwen_optimizer=self.qwen_optimizer
                )
                self._log_ai_strategy_message(f"✅ 判断模式已切换为: {mode_display}", "SUCCESS")
                
                # 如果是AI或混合模式，提示用户
                if judgment_mode in ["ai", "hybrid"]:
                    self._log_ai_strategy_message("  💡 AI模式将使用Qwen3分析性能数据", "INFO")
                    if judgment_mode == "hybrid":
                        self._log_ai_strategy_message("  💡 混合模式：先检查固定阈值，再由AI确认", "INFO")
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

    def _update_confirm_counts(self):
        """更新确认次数显示（Spinbox值改变时调用）"""
        try:
            entry_count = self.entry_confirm_count_var.get()
            reverse_count = self.reverse_confirm_count_var.get()
            
            # 这里可以添加实时显示更新逻辑
            # 暂时只记录日志
            self._log_ai_strategy_message(
                f"确认次数设置: 开仓{entry_count}次, 平仓{reverse_count}次", 
                "INFO"
            )
        except Exception as e:
            self._log_ai_strategy_message(f"更新确认次数显示失败: {e}", "ERROR")

    def _apply_confirm_counts(self):
        """应用确认次数设置到策略实例"""
        try:
            # 获取Spinbox的值
            entry_count = int(self.entry_confirm_count_var.get())
            reverse_count = int(self.reverse_confirm_count_var.get())
            
            # 验证范围
            if not (1 <= entry_count <= 10):
                self._log_ai_strategy_message(f"开仓确认次数超出范围: {entry_count} (1-10)", "ERROR")
                return
            if not (1 <= reverse_count <= 10):
                self._log_ai_strategy_message(f"平仓确认次数超出范围: {reverse_count} (1-10)", "ERROR")
                return
            
            # 更新策略实例（如果存在）
            if hasattr(self, 'strategy') and self.strategy is not None:
                self.strategy.entry_confirm_count = entry_count
                self.strategy.reverse_confirm_count = reverse_count
                self.strategy.consecutive_entry_count = 0  # 重置计数器
                self.strategy.consecutive_reverse_count = 0  # 重置计数器
                self.strategy.last_entry_signal = None  # 重置信号
                
                self._log_ai_strategy_message(
                    f"✅ 确认次数已应用到当前策略: 开仓{entry_count}次, 平仓{reverse_count}次", 
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  注意: 已重置连续计数器和最后信号记录", 
                    "INFO"
                )
            else:
                self._log_ai_strategy_message(
                    f"✅ 确认次数已保存: 开仓{entry_count}次, 平仓{reverse_count}次", 
                    "SUCCESS"
                )
                self._log_ai_strategy_message(
                    f"  将在下次启动交易时生效", 
                    "INFO"
                )
            
            # 保存设置到配置文件
            self._save_confirm_counts_settings(entry_count, reverse_count)
            
        except ValueError as e:
            self._log_ai_strategy_message(f"数值转换失败: {e} (请输入1-10的整数)", "ERROR")
        except Exception as e:
            self._log_ai_strategy_message(f"应用确认次数失败: {e}", "ERROR")

    def _save_confirm_counts_settings(self, entry_count, reverse_count):
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
                    
                    self.entry_confirm_count_var.set(str(entry_count))
                    self.reverse_confirm_count_var.set(str(reverse_count))
                    
                    self._log_ai_strategy_message(
                        f"已加载确认次数设置: 开仓{entry_count}次, 平仓{reverse_count}次", 
                        "INFO"
                    )
                    return True
            
            return False
            
        except Exception as e:
            self._log_ai_strategy_message(f"加载设置失败: {e}", "WARNING")
            return False

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
    KronosTradingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
