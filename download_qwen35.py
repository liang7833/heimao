#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen 3.5:4B 模型下载脚本
下载 Qwen/Qwen2.5-3B-Instruct 或 Qwen/Qwen2.5-7B-Instruct 模型到项目目录
"""

import os
import sys
import argparse

def download_model(model_name: str, output_dir: str):
    """
    使用 huggingface_hub 下载模型
    
    Args:
        model_name: HuggingFace 模型名称
        output_dir: 输出目录
    """
    print(f"="*80)
    print(f"正在下载模型: {model_name}")
    print(f"保存到: {output_dir}")
    print(f"="*80)
    
    try:
        from huggingface_hub import snapshot_download
        
        print("\n开始下载...")
        print("这可能需要几分钟到几十分钟，请耐心等待...")
        print("提示：如果网络不好，可以多次运行此脚本，会自动续传\n")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            max_workers=4,
            resume_download=True
        )
        
        print("\n" + "="*80)
        print("✓ 模型下载完成！")
        print(f"✓ 模型目录: {output_dir}")
        print("="*80)
        
        print("\n现在可以运行测试了:")
        print(f"  python test_qwen35.py")
        print("\n或修改 qwen_analyzer.py 中的模型目录指向新下载的模型")
        
    except ImportError:
        print("\n❌ 需要安装 huggingface_hub")
        print("请运行:")
        print("  pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="下载 Qwen 模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-4B-Instruct",
        help="模型名称 (默认: Qwen/Qwen2.5-4B-Instruct，推荐) 或 Qwen/Qwen3.5-4B"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: models/qwen35)"
    )
    
    args = parser.parse_args()
    
    # 确定输出目录
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)
    
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "models", "qwen35")
    else:
        output_dir = args.output_dir
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载模型
    download_model(args.model, output_dir)

if __name__ == "__main__":
    main()
