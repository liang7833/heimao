#!/usr/bin/env python
"""下载Qwen2.5-0.5B模型"""

import os
from huggingface_hub import snapshot_download

def download_qwen_0_5b():
    """下载Qwen2.5-0.5B模型"""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = os.path.join("models", "Qwen2.5-0.5B-Instruct")
    
    print(f"正在下载模型: {model_name}")
    print(f"保存到: {local_dir}")
    
    try:
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("✅ 模型下载完成！")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")

if __name__ == "__main__":
    download_qwen_0_5b()