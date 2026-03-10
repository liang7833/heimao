#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
黑猫交易系统启动器
独立显示启动画面，然后启动主程序
"""

import tkinter as tk
import os
import sys
import subprocess
import time
from threading import Thread

# 确保当前目录在路径中
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)


class SplashScreen:
    """启动画面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        
        # 设置窗口大小600x600
        window_width = 600
        window_height = 600
        
        # 获取屏幕尺寸并计算居中位置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 设置背景
        self.root.configure(bg="#1a1a1a")
        
        # 创建主框架
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 尝试加载图片
        self.image_label = None
        try:
            from PIL import Image, ImageTk
            # 尝试查找图片文件
            possible_images = [
                os.path.join(base_dir, "splash.png"),
                os.path.join(base_dir, "splash.jpg"),
                os.path.join(base_dir, "splash.jpeg"),
                os.path.join(base_dir, "black_cat.png"),
            ]
            
            image_path = None
            for img_path in possible_images:
                if os.path.exists(img_path):
                    image_path = img_path
                    break
            
            if image_path:
                img = Image.open(image_path)
                # 保持图片比例，调整到合适大小
                img.thumbnail((550, 500))
                photo = ImageTk.PhotoImage(img)
                self.image_label = tk.Label(main_frame, image=photo, bg="#1a1a1a")
                self.image_label.image = photo
                self.image_label.pack(pady=(30, 15))
            else:
                tk.Label(main_frame, text="🐱", bg="#1a1a1a", 
                       fg="#ffffff", font=("Arial", 150)).pack(pady=(100, 20))
        except Exception as e:
            print(f"加载启动画面图片失败: {e}")
            tk.Label(main_frame, text="🐱", bg="#1a1a1a", 
                   fg="#ffffff", font=("Arial", 150)).pack(pady=(100, 20))
        
        # 显示文字
        text_label = tk.Label(main_frame, text="黑猫交易系统v2.0", 
                            bg="#1a1a1a", fg="#ffffff", 
                            font=("微软雅黑", 28, "bold"))
        text_label.pack()
        
        # 显示加载提示
        self.loading_label = tk.Label(main_frame, text="正在加载...", 
                                    bg="#1a1a1a", fg="#888888", 
                                    font=("微软雅黑", 12))
        self.loading_label.pack(pady=(10, 0))
        
        self.root.update()
    
    def update_loading(self, text):
        """更新加载提示"""
        try:
            self.loading_label.config(text=text)
            self.root.update()
        except:
            pass
    
    def destroy(self):
        """关闭启动画面"""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


def launch_main_program(splash):
    """启动主程序（支持打包环境）"""
    # 信号文件路径
    signal_file = os.path.join(base_dir, "_splash_close.txt")
    
    # 先清理旧的信号文件
    try:
        if os.path.exists(signal_file):
            os.remove(signal_file)
    except:
        pass
    
    try:
        splash.update_loading("正在初始化主程序...")
        
        # 判断是否是打包后的环境
        if getattr(sys, 'frozen', False):
            # 打包后的exe - 直接在同一进程中运行
            import gui_trading
            
            splash.update_loading("正在准备启动...")
            time.sleep(0.5)
            
            # 创建关闭信号文件
            def close_splash():
                try:
                    with open(signal_file, 'w') as f:
                        f.write('close')
                except:
                    pass
            
            # 运行主程序
            splash.update_loading("启动完成！")
            time.sleep(0.3)
            splash.destroy()
            
            # 直接调用主程序
            gui_trading.main()
            
        else:
            # 开发环境 - 使用subprocess启动
            main_script = os.path.join(base_dir, "gui_trading.py")
            
            # 使用与当前解释器相同的Python来运行
            python_exe = sys.executable
            
            splash.update_loading("启动中...")
            
            # 启动主程序
            process = subprocess.Popen([python_exe, main_script])
            
            splash.update_loading("等待主程序启动...")
            
            # 等待信号文件出现（主程序准备好）
            max_wait = 60  # 最多等待60秒
            wait_count = 0
            while wait_count < max_wait:
                if os.path.exists(signal_file):
                    break
                time.sleep(0.5)
                wait_count += 0.5
                # 更新加载动画
                dots = "." * (int(wait_count) % 4)
                splash.update_loading(f"等待主程序启动{dots}")
            
            # 清理信号文件
            try:
                if os.path.exists(signal_file):
                    os.remove(signal_file)
            except:
                pass
            
            # 关闭启动画面
            splash.update_loading("启动完成！")
            time.sleep(0.3)
            splash.destroy()
            
            # 等待主程序结束
            process.wait()
        
    except Exception as e:
        print(f"启动主程序失败: {e}")
        import traceback
        traceback.print_exc()
        splash.update_loading(f"启动失败: {e}")
        time.sleep(3)
        splash.destroy()
        
        # 清理信号文件
        try:
            if os.path.exists(signal_file):
                os.remove(signal_file)
        except:
            pass


def main():
    """主函数"""
    # 创建启动画面
    splash = SplashScreen()
    
    # 在新线程中启动主程序
    thread = Thread(target=launch_main_program, args=(splash,))
    thread.daemon = True
    thread.start()
    
    # 运行启动画面的主循环
    splash.root.mainloop()


if __name__ == "__main__":
    main()
