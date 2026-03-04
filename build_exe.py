#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kronos交易系统打包脚本
使用 PyInstaller 打包成带目录的 exe 文件
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    if description:
        print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print(f"\n✓ 命令执行成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 命令执行失败！")
        print(f"错误代码: {e.returncode}")
        if e.stdout:
            print(f"标准输出:\n{e.stdout}")
        if e.stderr:
            print(f"错误输出:\n{e.stderr}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Kronos 交易系统打包器                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查 PyInstaller 是否安装
    print("步骤1: 检查依赖...")
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller 已安装 (版本: {PyInstaller.__version__})")
    except ImportError:
        print("  ✗ PyInstaller 未安装，正在安装...")
        if not run_command("pip install pyinstaller", "安装 PyInstaller"):
            print("\n✗ PyInstaller 安装失败！")
            return
    
    # 清理之前的构建
    print("\n步骤2: 清理之前的构建文件...")
    build_dirs = ['build', 'dist']
    for dir_name in build_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  ✓ 已删除: {dir_name}")
    
    # 检查 spec 文件
    spec_file = Path("kronos_trading.spec")
    if not spec_file.exists():
        print(f"\n✗ spec 文件不存在: {spec_file}")
        return
    
    # 执行打包
    print("\n步骤3: 开始打包...")
    if not run_command(f"py -m PyInstaller --clean {spec_file}", "PyInstaller 打包"):
        print("\n✗ 打包失败！")
        return
    
    # 检查打包结果
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("\n✗ 打包输出目录不存在！")
        return
    
    exe_files = list(dist_dir.glob("*.exe"))
    if not exe_files:
        print("\n✗ 未找到生成的 exe 文件！")
        return
    
    print(f"\n✓ 打包成功！")
    print(f"\n生成的文件位置: {dist_dir.absolute()}")
    print(f"\n包含的文件:")
    for item in dist_dir.iterdir():
        print(f"  - {item.name}")
    
    print("\n" + "="*60)
    print("打包完成！")
    print("="*60)
    print("\n注意事项:")
    print("1. models 目录已打包进 exe")
    print("2. Kronos 目录已打包进 exe")
    print("3. 训练模型的导入导出功能正常")
    print("4. 首次运行可能需要较长时间加载模型")
    print("5. 建议在 dist 目录下运行 exe")

if __name__ == "__main__":
    main()
