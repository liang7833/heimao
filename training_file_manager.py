"""
训练文件管理器 - 打包和解包训练文件
用于在不同机器间迁移Kronos训练文件
"""

import os
import zipfile
import shutil
import yaml
import json
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class TrainingFileManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Kronos训练文件管理器")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # 设置样式
        self.setup_styles()

        # 主框架
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="Kronos训练文件管理器",
            font=("微软雅黑", 18, "bold"),
            foreground="#2c3e50",
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

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
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 30))

        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 30))

        # 打包按钮
        self.pack_button = ttk.Button(
            button_frame,
            text="📦 打包本机训练文件",
            command=self.pack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.pack_button.grid(row=0, column=0, padx=10, pady=10)

        # 解包按钮
        self.unpack_button = ttk.Button(
            button_frame,
            text="📤 解包训练文件到本机",
            command=self.unpack_training_files,
            width=25,
            style="Accent.TButton",
        )
        self.unpack_button.grid(row=0, column=1, padx=10, pady=10)

        # 状态区域
        status_frame = ttk.LabelFrame(main_frame, text="操作状态", padding="15")
        status_frame.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20)
        )

        # 状态文本
        self.status_text = tk.Text(
            status_frame,
            height=12,
            width=80,
            font=("Consolas", 9),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.status_text.grid(row=0, column=0)

        # 滚动条
        scrollbar = ttk.Scrollbar(
            status_frame, orient=tk.VERTICAL, command=self.status_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)

        # 底部信息
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # 当前配置信息
        self.config_info = tk.StringVar(value="正在加载配置...")
        config_label = ttk.Label(
            info_frame,
            textvariable=self.config_info,
            font=("微软雅黑", 9),
            foreground="#7f8c8d",
        )
        config_label.grid(row=0, column=0, sticky=tk.W)

        # 版本信息
        version_label = ttk.Label(
            info_frame,
            text="版本 1.0 · Kronos交易系统",
            font=("微软雅黑", 8),
            foreground="#95a5a6",
        )
        version_label.grid(row=0, column=1, sticky=tk.E)

        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(1, weight=1)

        # 加载配置
        self.load_config()

    def setup_styles(self):
        style = ttk.Style()

        # 设置主题
        style.theme_use("clam")

        # 配置按钮样式
        style.configure(
            "Accent.TButton",
            font=("微软雅黑", 11, "bold"),
            padding=10,
            foreground="white",
            background="#3498db",
            bordercolor="#2980b9",
            focuscolor="none",
        )

        style.map(
            "Accent.TButton",
            background=[("active", "#2980b9")],
            relief=[("pressed", "sunken")],
        )

    def log_message(self, message, level="INFO"):
        """记录消息到状态区域"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"

        # 插入带颜色的文本
        self.status_text.insert(tk.END, formatted_msg)
        self.status_text.see(tk.END)
        self.status_text.update()

    def load_config(self):
        """加载训练配置文件"""
        try:
            config_path = "train_config.yaml"
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)

                exp_name = self.config.get("exp_name", "未知")
                save_path = self.config.get("save_path", "未知")
                data_path = self.config.get("data_path", "未知")

                info_text = f"实验名称: {exp_name} | 保存路径: {save_path} | 数据文件: {os.path.basename(data_path)}"
                self.config_info.set(info_text)
                self.log_message("✓ 配置文件加载成功", "SUCCESS")
            else:
                self.config = {}
                self.config_info.set("未找到配置文件 train_config.yaml")
                self.log_message("⚠️ 未找到配置文件", "WARNING")
        except Exception as e:
            self.config = {}
            self.config_info.set("配置文件加载失败")
            self.log_message(f"❌ 配置文件加载失败: {e}", "ERROR")

    def find_model_files(self, exp_name, save_path):
        """查找模型文件的实际位置"""
        possible_paths = []

        # 1. 配置文件中的路径
        if save_path and exp_name:
            possible_paths.append(os.path.join(save_path, exp_name))

        # 2. 常见的路径（按优先级排序）
        possible_paths.append(
            os.path.join("Kronos", "finetune_csv", "Kronos", "finetune_csv", "finetuned", exp_name)
        )
        possible_paths.append(
            os.path.join("Kronos", "finetune_csv", "finetuned", exp_name)
        )
        possible_paths.append(os.path.join("finetuned", exp_name))

        # 3. 搜索整个项目目录（作为最后手段）
        for root, dirs, files in os.walk("."):
            # 跳过一些不必要的目录
            if "__pycache__" in root or ".git" in root or "backup_" in root:
                continue
            if exp_name in dirs:
                path = os.path.join(root, exp_name)
                # 确保找到的目录包含模型文件
                if any(os.path.exists(os.path.join(path, f)) for f in ["config.json", "model.safetensors"]):
                    possible_paths.append(path)

        # 检查路径是否存在
        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def search_all_model_files(self):
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

    def pack_training_files(self):
        """打包本机训练文件"""
        try:
            # 检查配置文件
            if not self.config:
                self.log_message(
                    "❌ 请先确保配置文件 train_config.yaml 存在且有效", "ERROR"
                )
                return

            # 询问保存位置
            default_filename = f"kronos_training_{self.config.get('exp_name', 'unknown')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = filedialog.asksaveasfilename(
                title="保存训练文件包",
                defaultextension=".zip",
                initialfile=default_filename,
                filetypes=[("ZIP压缩包", "*.zip"), ("所有文件", "*.*")],
            )

            if not zip_path:
                self.log_message("操作已取消", "INFO")
                return

            self.log_message(
                f"开始打包训练文件到: {os.path.basename(zip_path)}", "INFO"
            )

            # 收集要打包的文件
            files_to_pack = []

            # 1. 模型文件
            exp_name = self.config.get("exp_name")
            save_path = self.config.get("save_path")
            if exp_name:
                # 查找模型文件的实际位置
                model_path = self.find_model_files(exp_name, save_path)
                if model_path and os.path.exists(model_path):
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, ".")
                            files_to_pack.append((full_path, rel_path))
                    self.log_message(f"✓ 添加模型文件: {model_path}", "SUCCESS")
                else:
                    self.log_message(
                        f"⚠️ 未找到模型文件，实验名称: {exp_name}", "WARNING"
                    )
                    # 尝试搜索所有可能的模型文件
                    self.log_message("正在搜索模型文件...", "INFO")
                    model_files_found = self.search_all_model_files()
                    if model_files_found:
                        for file_path, rel_path in model_files_found:
                            files_to_pack.append((file_path, rel_path))
                        self.log_message(
                            f"✓ 通过搜索找到 {len(model_files_found)} 个模型文件",
                            "SUCCESS",
                        )

            # 2. 配置文件
            config_files = ["train_config.yaml"]
            for config_file in config_files:
                if os.path.exists(config_file):
                    files_to_pack.append((config_file, config_file))
                    self.log_message(f"✓ 添加配置文件: {config_file}", "SUCCESS")
                else:
                    self.log_message(f"⚠️ 配置文件不存在: {config_file}", "WARNING")

            # 3. 数据文件
            data_path = self.config.get("data_path")
            if data_path:
                # 尝试找到数据文件
                data_file_found = False
                data_file_name = os.path.basename(data_path)

                # 可能的目录列表（按优先级）
                possible_dirs = [
                    os.path.dirname(data_path) if os.path.isabs(data_path) else "",
                    "training_data",
                    "data",
                    "Kronos/finetune_csv/data",
                    "."
                ]

                # 移除空字符串
                possible_dirs = [d for d in possible_dirs if d]

                for data_dir in possible_dirs:
                    test_path = os.path.join(data_dir, data_file_name)
                    if os.path.exists(test_path):
                        # 转换为相对路径
                        try:
                            rel_path = os.path.relpath(test_path, ".")
                            files_to_pack.append((test_path, rel_path))
                            self.log_message(f"✓ 添加数据文件: {rel_path}", "SUCCESS")
                            data_file_found = True
                            break
                        except ValueError:
                            # 如果不在同一驱动器，使用基本名称
                            files_to_pack.append((test_path, os.path.basename(test_path)))
                            self.log_message(
                                f"✓ 添加数据文件: {os.path.basename(test_path)}", "SUCCESS"
                            )
                            data_file_found = True
                            break

                if not data_file_found:
                    self.log_message(f"⚠️ 数据文件不存在: {data_path}", "WARNING")
            else:
                self.log_message("⚠️ 配置文件中未指定数据文件路径", "WARNING")

            # 创建ZIP文件
            if not files_to_pack:
                self.log_message("❌ 没有找到可打包的文件", "ERROR")
                return

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path, arcname in files_to_pack:
                    zipf.write(file_path, arcname)
                    self.log_message(f"  压缩: {arcname}", "INFO")

            # 添加元数据
            metadata = {
                "打包时间": datetime.datetime.now().isoformat(),
                "实验名称": self.config.get("exp_name"),
                "文件数量": len(files_to_pack),
                "版本": "1.0",
            }

            with zipfile.ZipFile(zip_path, "a") as zipf:
                zipf.writestr(
                    "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False)
                )

            self.log_message(
                f"✓ 打包完成! 共打包 {len(files_to_pack)} 个文件", "SUCCESS"
            )
            self.log_message(f"文件保存到: {zip_path}", "INFO")

            messagebox.showinfo(
                "打包完成",
                f"训练文件打包成功!\n\n保存位置: {zip_path}\n文件数量: {len(files_to_pack)}",
            )

        except Exception as e:
            self.log_message(f"❌ 打包失败: {e}", "ERROR")

    def unpack_training_files(self):
        """解包训练文件到本机"""
        try:
            # 选择ZIP文件
            zip_path = filedialog.askopenfilename(
                title="选择训练文件包",
                filetypes=[("ZIP压缩包", "*.zip"), ("所有文件", "*.*")],
            )

            if not zip_path:
                self.log_message("操作已取消", "INFO")
                return

            self.log_message(f"开始解包文件: {os.path.basename(zip_path)}", "INFO")

            # 读取元数据
            metadata = None
            try:
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    if "metadata.json" in zipf.namelist():
                        with zipf.open("metadata.json") as f:
                            metadata = json.load(f)
                            self.log_message(
                                f"✓ 读取元数据: {metadata.get('实验名称', '未知')}",
                                "SUCCESS",
                            )
            except:
                self.log_message("⚠️ 无法读取元数据", "WARNING")

            # 确认解包
            if metadata:
                confirm_msg = f"即将解包训练文件:\n\n实验名称: {metadata.get('实验名称', '未知')}\n打包时间: {metadata.get('打包时间', '未知')}\n文件数量: {metadata.get('文件数量', '未知')}\n\n解包将覆盖本机现有文件，是否继续？"
            else:
                confirm_msg = f"即将解包文件: {os.path.basename(zip_path)}\n\n解包将覆盖本机现有文件，是否继续？"

            if not messagebox.askyesno("确认解包", confirm_msg):
                self.log_message("解包操作已取消", "INFO")
                return

            # 解包文件
            extracted_files = []
            skipped_files = []
            with zipfile.ZipFile(zip_path, "r") as zipf:
                # 先获取文件列表
                file_list = [f for f in zipf.namelist() if f != "metadata.json"]

                for filename in file_list:
                    try:
                        # 验证文件路径
                        if not self.validate_zip_path(filename):
                            self.log_message(
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
                        self.log_message(f"  解压: {filename}", "INFO")
                    except Exception as e:
                        self.log_message(f"  ⚠️ 解压失败 {filename}: {e}", "WARNING")

            if extracted_files:
                self.log_message(
                    f"✓ 解包完成! 共解压 {len(extracted_files)} 个文件", "SUCCESS"
                )
            else:
                self.log_message("⚠️ 没有解压任何文件", "WARNING")

            if skipped_files:
                self.log_message(
                    f"⚠️ 跳过了 {len(skipped_files)} 个不安全文件", "WARNING"
                )

            # 重新加载配置
            self.load_config()

            messagebox.showinfo(
                "解包完成", f"训练文件解包成功!\n\n解压文件: {len(extracted_files)} 个"
            )

        except Exception as e:
            self.log_message(f"❌ 解包失败: {e}", "ERROR")

    def validate_zip_path(self, filename):
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

    def backup_existing_files(self, files_to_backup):
        """备份现有文件"""
        backup_dir = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)

        for file_path in files_to_backup:
            if os.path.exists(file_path):
                try:
                    shutil.copy2(
                        file_path, os.path.join(backup_dir, os.path.basename(file_path))
                    )
                    self.log_message(f"  备份: {os.path.basename(file_path)}", "INFO")
                except Exception as e:
                    self.log_message(f"  ⚠️ 备份失败 {file_path}: {e}", "WARNING")

        return backup_dir


def main():
    root = tk.Tk()
    TrainingFileManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
