# Qwen模型下载指南

## 重要提示

**Qwen模型文件较大（约3-15GB），请确保：**
1. 有足够的磁盘空间（至少20GB）
2. 有稳定的网络连接
3. 有足够的显存（7B模型建议8GB+显存）

---

## 方法一：使用一键下载脚本（推荐）

运行以下命令自动下载Qwen2.5-1.5B-Instruct（默认推荐，超轻量快速）：

```bash
py download_qwen_model.py
```

按照提示选择合适的模型即可。

---

## 方法二：手动从HuggingFace下载

### 步骤1：安装依赖

```bash
py -m pip install huggingface_hub transformers torch
```

### 步骤2：选择并下载模型

推荐的模型（按兼容性排序）：

| 模型 | 大小 | 显存要求 | 推荐 |
|------|------|---------|------|
| **Qwen/Qwen2.5-1.5B-Instruct** | ~3GB | 4GB+ | ⭐⭐⭐ 首选推荐 |
| Qwen/Qwen2.5-3B-Instruct | ~6GB | 6GB+ | ⭐⭐ 轻量版 |
| Qwen/Qwen2.5-7B-Instruct | ~14GB | 8GB+ | ⭐ 完整版 |
| Qwen/Qwen3.5-4B | ~8GB | 6GB+ | 最新版（需升级transformers） |

### 步骤3：下载命令示例

下载Qwen2.5-1.5B-Instruct（推荐）：

```python
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "models/qwen"

snapshot_download(
    repo_id=model_name,
    local_dir=output_dir,
    local_dir_use_symlinks=False
)
```

或者在Python中运行：

```bash
py -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/qwen', local_dir_use_symlinks=False)"
```

## 使用Qwen3.5-4B（可选）

如果确实想用Qwen3.5-4B，请按以下步骤：

```bash
# 第一步：升级依赖（必须）
py -m pip install --upgrade transformers huggingface_hub
py -m pip install --upgrade torch

# 第二步：下载Qwen3.5-4B
py download_qwen_model.py --model Qwen/Qwen3.5-4B
```

## Qwen2.5-1.5B-Instruct特点

| 参数 | 规格 |
|------|------|
| 参数量 | 15亿 |
| 模型大小 | ~3GB |
| 显存要求 | 4GB+ |
| 推荐指数 | ⭐⭐⭐ 首选 |

## 快速开始（推荐）

**不想下载大模型？直接用规则分析模式！**

1. 启动程序
2. 勾选主面板的"🔮 启用Qwen分析"
3. 开始交易！

规则分析模式完全免费、无需下载、立即可用！

---

## 方法三：使用规则分析模式（无需下载）

如果不想下载大模型，可以直接使用**规则分析模式**！

### 规则分析模式特点：
- ✅ 无需下载任何模型
- ✅ 立即可用
- ✅ 基于技术指标（MA、支撑阻力等）
- ✅ 与Kronos数据格式完全兼容

### 如何启用规则分析模式：

程序会自动检测：
- 如果没有找到模型文件，自动切换到规则分析模式
- 启用Qwen分析后即可使用，无需额外配置

---

## 验证模型是否安装成功

1. 启动程序
2. 在主面板启用"🔮 启用Qwen分析"
3. 查看终端输出：

如果看到：
```
正在加载Qwen模型...
找到模型目录: h:\kronos交易\models\qwen
正在加载tokenizer...
正在加载模型...
✓ Qwen模型加载成功
```
说明本地模型加载成功！

如果看到：
```
未找到本地Qwen模型目录
将使用基于规则的分析
使用基于规则的Qwen分析（备用方法）
```
说明正在使用规则分析模式，无需模型文件！

---

## 目录结构

模型文件应放在以下位置之一：

```
h:\kronos交易\
├── models\
│   └── qwen\              ← 推荐位置
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── model.safetensors
│       └── ...
└── _internal\
    └── models\
        └── qwen\          ← 打包后位置
            └── ...
```

---

## 常见问题

**Q: 规则分析模式效果怎么样？**
A: 规则分析模式基于标准技术指标（移动平均线、支撑阻力位等），对于普通交易已经足够。

**Q: 下载了模型还是用不了？**
A: 检查：
1. 模型文件是否在 `models/qwen/` 目录
2. 是否有 `config.json` 和 `tokenizer_config.json` 文件
3. 显存是否足够

**Q: 可以同时使用规则分析和模型吗？**
A: 程序会优先使用本地模型，如果没有则自动使用规则分析。

---

## 快速开始（推荐）

**不想下载大模型？直接用规则分析模式！**

1. 启动程序
2. 勾选主面板的"🔮 启用Qwen分析"
3. 开始交易！

规则分析模式完全免费、无需下载、立即可用！
