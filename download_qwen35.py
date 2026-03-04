#!/usr/bin/env python
"""下载 Qwen3.5-0.8B-Instruct 模型"""

import os
import sys

print("=" * 60)
print("Qwen3.5-0.8B 模型下载工具")
print("=" * 60)

# 检查 huggingface-hub 是否安装
try:
    from huggingface_hub import snapshot_download
    print("✓ huggingface-hub 已安装")
except ImportError:
    print("✗ huggingface-hub 未安装，正在安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
    from huggingface_hub import snapshot_download
    print("✓ huggingface-hub 安装成功")

# 确保 models 目录存在
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# 模型下载路径
repo_id = "Qwen/Qwen3.5-0.8B-Instruct"
local_dir = os.path.join(models_dir, "Qwen3.5-0.8B-Instruct")

print(f"\n正在下载模型: {repo_id}")
print(f"保存路径: {local_dir}")
print("=" * 60)

try:
    # 下载模型
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print("\n" + "=" * 60)
    print("✓ 模型下载成功！")
    print(f"模型位置: {local_dir}")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 模型下载失败: {e}")
    print("\n请尝试手动下载:")
    print(f"  访问: https://huggingface.co/{repo_id}")
    print(f"  或运行: huggingface-cli download {repo_id} --local-dir {local_dir}")
    sys.exit(1)
