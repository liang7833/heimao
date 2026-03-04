#!/usr/bin/env python
"""下载Kronos base和mini模型到本地models目录"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("错误: 需要安装huggingface_hub库")
    print("请运行: pip install huggingface_hub")
    sys.exit(1)

def download_model(model_id, local_dir):
    """下载模型到指定目录"""
    print(f"正在下载 {model_id} 到 {local_dir}...")
    
    # 如果目录已存在，先删除
    if os.path.exists(local_dir):
        print(f"  目录已存在，跳过下载")
        return True
    
    try:
        # 下载模型文件（排除缓存文件）
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            ignore_patterns=["*.lock", "*.metadata", ".git*", "*.py", "*.pyc", "*.txt", "*.md"],
            token=None  # 不需要token，公开模型
        )
        print(f"  ✓ 下载完成: {model_id}")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败 {model_id}: {e}")
        return False

def main():
    print("=" * 60)
    print("Kronos模型下载工具")
    print("=" * 60)
    print()
    
    # 创建models目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 需要下载的模型列表
    models_to_download = [
        # (HuggingFace模型ID, 本地目录名)
        ("NeoQuasar/Kronos-base", "kronos-base"),
        ("NeoQuasar/Kronos-mini", "kronos-mini"),
        ("NeoQuasar/Kronos-Tokenizer-base", "kronos-tokenizer-base"),
        ("NeoQuasar/Kronos-small", "kronos-small"),  # 可能已经存在，但确保完整
    ]
    
    success_count = 0
    
    for model_id, local_name in models_to_download:
        local_dir = models_dir / local_name
        
        # 跳过已经完整存在的目录
        if local_dir.exists():
            # 检查是否包含必要的文件
            required_files = ["config.json", "model.safetensors", "README.md"]
            has_all_files = all((local_dir / f).exists() for f in required_files)
            
            if has_all_files:
                print(f"✓ {local_name} 已存在且完整，跳过")
                success_count += 1
                continue
        
        if download_model(model_id, local_dir):
            success_count += 1
    
    print()
    print("=" * 60)
    print("下载完成!")
    print(f"成功下载: {success_count}/{len(models_to_download)} 个模型")
    print()
    print("模型目录结构:")
    for item in models_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}/")
            for file in item.glob("*"):
                if file.is_file() and not file.name.startswith("."):
                    print(f"    - {file.name}")
    
    # 验证文件
    print()
    print("验证文件...")
    for model_name in ["kronos-base", "kronos-mini", "kronos-small", "kronos-tokenizer-base"]:
        model_dir = models_dir / model_name
        if model_dir.exists():
            config = model_dir / "config.json"
            weights = model_dir / "model.safetensors"
            if config.exists() and weights.exists():
                print(f"  ✓ {model_name}: 配置和权重文件完整")
            else:
                print(f"  ✗ {model_name}: 缺少文件")

if __name__ == "__main__":
    main()