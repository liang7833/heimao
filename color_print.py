
"""
彩色打印工具模块
支持Windows PowerShell和其他终端
"""

from colorama import init, Fore, Style, Back

# 初始化colorama
init(autoreset=True)

def print_success(text):
    """绿色成功信息"""
    print(Fore.GREEN + Style.BRIGHT + "✓ " + text + Style.RESET_ALL)

def print_error(text):
    """红色错误信息"""
    print(Fore.RED + Style.BRIGHT + "✗ " + text + Style.RESET_ALL)

def print_warning(text):
    """黄色警告信息"""
    print(Fore.YELLOW + Style.BRIGHT + "⚠ " + text + Style.RESET_ALL)

def print_info(text):
    """蓝色信息"""
    print(Fore.CYAN + Style.BRIGHT + "ℹ " + text + Style.RESET_ALL)

def print_open(text):
    """绿色开仓信息"""
    print(Fore.GREEN + Style.BRIGHT + "🚀 " + text + Style.RESET_ALL)

def print_close(text):
    """红色平仓信息"""
    print(Fore.RED + Style.BRIGHT + "📉 " + text + Style.RESET_ALL)

def print_signal_buy(text):
    """绿色买入信号"""
    print(Fore.GREEN + Style.BRIGHT + text + Style.RESET_ALL)

def print_signal_sell(text):
    """红色卖出信号"""
    print(Fore.RED + Style.BRIGHT + text + Style.RESET_ALL)

def print_signal_neutral(text):
    """白色中性信号"""
    print(Fore.WHITE + Style.DIM + text + Style.RESET_ALL)

def print_trend_up(text):
    """绿色上涨趋势"""
    print(Fore.GREEN + Style.BRIGHT + text + Style.RESET_ALL)

def print_trend_down(text):
    """红色下跌趋势"""
    print(Fore.RED + Style.BRIGHT + text + Style.RESET_ALL)

def print_reverse_signal(text):
    """紫色反转信号"""
    print(Fore.MAGENTA + Style.BRIGHT + text + Style.RESET_ALL)

def print_ai_decision(text):
    """青色AI决策"""
    print(Fore.CYAN + Style.BRIGHT + text + Style.RESET_ALL)

def print_highlight(text):
    """高亮信息（黄色背景）"""
    print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + text + Style.RESET_ALL)
