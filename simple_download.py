#!/usr/bin/env python
"""简单的 Qwen3.5-0.8B 下载脚本"""

import os
from huggingface_hub import snapshot_download

print("=" * 60)
print("正在下载 Qwen3.5-0.8B 模型...")
print("=" * 60)

repo_id = "Qwen/Qwen3.5-0.8B"
local_dir = "models/Qwen3.5-0.8B-Instruct"

try:
    print(f"\n仓库: {repo_id}")
    print(f"保存到: {os.path.abspath(local_dir)}")
    print("\n开始下载...")
    
    model_dir = snapshot_download(
        repo_id=repo_id, local_dir=local_dir)
    
    print("\n" + "=" * 60)
    print("✓ 下载成功!")
    print(f"模型位置: {model_dir}")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 下载失败: {e}")
    import traceback
    traceback.print_exc()
