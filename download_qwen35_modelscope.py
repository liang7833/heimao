#!/usr/bin/env python
"""使用 ModelScope 下载 Qwen3.5-0.8B-Instruct 模型"""

import os
import sys

print("=" * 60)
print("Qwen3.5-0.8B 模型下载工具 (ModelScope)")
print("=" * 60)

# 检查 modelscope 是否安装
try:
    from modelscope import snapshot_download
    print("✓ modelscope 已安装")
except ImportError:
    print("✗ modelscope 未安装，正在安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
    from modelscope import snapshot_download
    print("✓ modelscope 安装成功")

# 确保 models 目录存在
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# 模型下载路径
repo_id = "Qwen/Qwen3.5-0.8B-Instruct"
local_dir = os.path.join(models_dir, "Qwen3.5-0.8B-Instruct")

print(f"\n正在从 ModelScope 下载模型: {repo_id}")
print(f"保存路径: {local_dir}")
print("=" * 60)

try:
    # 下载模型
    model_dir = snapshot_download(
        repo_id,
        cache_dir=models_dir,
        local_dir=local_dir
    )
    
    print("\n" + "=" * 60)
    print("✓ 模型下载成功！")
    print(f"模型位置: {model_dir}")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 模型下载失败: {e}")
    print("\n请尝试以下方法：")
    print("1. 等待一段时间，让模型同步完成")
    print("2. 访问 ModelScope 官网: https://modelscope.cn/models/Qwen/Qwen3.5-0.8B-Instruct")
    print("3. 或使用 HuggingFace 镜像源")
    sys.exit(1)
