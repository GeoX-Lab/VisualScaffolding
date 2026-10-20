# Visual Scaffolding

[English](README.md) · **简体中文**

论文 **Visual Scaffolding: Toward Native Multimodal Reasoning** 的代码与 benchmark。

📄 **论文：**[arXiv:2610.25247](https://arxiv.org/abs/2610.25247) · [PDF](https://arxiv.org/pdf/2610.25247)

| 资源 | 内容 | Hugging Face |
|---|---|---|
| 📚 **NMR-Bench** | 780 张测试图像及问题、答案 | [数据集](https://huggingface.co/datasets/ShaoRun/nmr-bench) |
| 🧮 **Burden Counting 模型** | 完整的任务专用模型 | [模型](https://huggingface.co/ShaoRun/vscaffold-4b-burden-counting) |
| 🧵 **Path Tracing 模型** | 完整的任务专用模型 | [模型](https://huggingface.co/ShaoRun/vscaffold-4b-path-tracing) |

数据集和模型发布在 Hugging Face；本仓库提供评测与推理代码。

## 🔎 工作简介

文本思维链能把中间结论留给后续步骤使用，但简短的文字描述可能丢失图像中的物体身份或空间细节。**原生多模态推理（NMR）**关注的是：语言与原生视觉表示能否共同承载可复用的中间状态。

Visual Scaffolding 是这一思路的具体实现。模型在生成推理过程时，可以输出 `[ref]` 请求视觉证据。Visual Retriever 从输入图像的细粒度特征中选择相关内容，Evidence Adaptor 将所选特征插入当前推理序列；模型随后继续生成，后续步骤仍可利用已插入的视觉内容。

`图像与问题 → 推理 → [ref] → 检索视觉特征 → 插入证据 → 继续推理`

具体实现基于 Qwen3-VL-4B-Thinking，两个任务使用分别训练的模型。本仓库提供推理代码和 **NMR-Bench** 评测器；benchmark 检验持续保持物体身份和空间连续性的能力。测试数据与模型已分别发布在 Hugging Face 仓库。

## 🎯 Benchmark

| 任务 | 模型需要完成的工作 | 测试图像 |
|---|---|---:|
| **Burden Counting** | 在拥挤场景中计数或比较物体 | 420 |
| **Path Tracing** | 沿交叉曲线追踪并匹配两端标记 | 360 |

每条测试记录包含图像、问题和标准答案。评测器只将**图像和问题**传给模型，标准答案仅用于评分。

### Burden Counting · 持续区分物体实例

420 张测试图像覆盖**七种问题形式**。其中五种是在类别或属性约束下计数：仅按类别（T1a）、尺寸（T2a）、倾斜角度（T2b）、遮挡状态（T2c），以及尺寸与倾斜角度的组合（T2d）。另两种要求比较大、小物体的数量（T3a），或找出给定类别中出现最多的一类（T3b）。

每种形式都包含 **20 张 Easy、20 张 Medium、20 张 Hard** 图像。难度在各形式内部设定，并不对应一个对所有形式通用的物体数量阈值。图像中的物体在类别、尺寸、角度和可见程度上变化，场景中还混有干扰物。

### Path Tracing · 保持曲线的空间连续性

360 张测试图像要求找出所有曲线两端的标记配对。端点使用**字母、数字、图标或形状**，每类 90 张。曲线交叉时，需要沿原曲线继续追踪，避免跳到附近的另一条线。

| 难度 | 每图曲线数 | 实际交叉数 | 测试图像 |
|---|---:|---:|---:|
| **Easy** | 3 | 0–5 | 120 |
| **Medium** | 4–5 | 1–10 | 120 |
| **Hard** | 6–7 | 3–17 | 120 |

标准测试集中，曲线数量、布局和交叉情况会共同变化。

### 🖼️ Easy → Medium → Hard

| 难度 | Burden Counting | Path Tracing |
|:---:|:---:|:---:|
| **Easy** | <img src="assets/benchmark/counting_test_T2c_000000.png" width="240" alt="Easy 难度的水獭遮挡计数样例"> | <img src="assets/benchmark/path_test_cell000_seed00000000.png" width="240" alt="Easy 难度的三条曲线样例"> |
| **Medium** | <img src="assets/benchmark/counting_test_T2c_000020.png" width="240" alt="Medium 难度的遮挡计数样例"> | <img src="assets/benchmark/path_test_cell024_seed00024000.png" width="240" alt="Medium 难度的四条曲线样例"> |
| **Hard** | <img src="assets/benchmark/counting_test_T2c_1000040.png" width="240" alt="Hard 难度的蝴蝶遮挡计数样例"> | <img src="assets/benchmark/path_test_cell060_seed00060000.png" width="240" alt="Hard 难度的图标端点样例"> |

六张图片均来自测试集，其中三张 Burden Counting 图像使用同一种遮挡计数问题形式（T2c）。完整问题与标准答案位于 Hugging Face 数据集中，数据集卡说明了记录格式；图片来源与许可见 [DATA_LICENSE.md](DATA_LICENSE.md)。

## 🚀 快速开始

参考环境使用 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install huggingface_hub
hf download ShaoRun/nmr-bench --repo-type dataset --local-dir benchmark
```

### 评测任意模型的预测

为所选任务的**每张测试图像**准备一条 JSONL 预测。每行需包含该任务 `test.jsonl` 中的 `sample_id` 和模型的原始回答：

```json
{"sample_id":"counting_test_T1a_000000","response":"model-generated response"}
```

运行评分器：

```bash
python -m visual_scaffolding score \
  --task burden_counting \
  --benchmark benchmark \
  --predictions predictions/counting.jsonl \
  --output outputs/counting-summary.json
```

评测 Path Tracing 时，将任务改为 `path_tracing`，并提供对应的预测文件。评分器会检查样本是否齐全，缺失或重复的 ID 会报错。数据集中的 `SCHEMA.md` 说明了字段及答案格式。

### 由评测器调用模型

评测器支持实现 `predict(image, question)` 的模型适配器。[examples/hf_adapter.py](examples/hf_adapter.py) 提供了 Hugging Face 模型示例。先安装适用于当前 GPU 的 PyTorch，再安装推理依赖：

```bash
pip install -e '.[inference]'
```

创建包含模型 ID 的 `model.json`，例如 `{"model_name":"Qwen/Qwen3-VL-4B-Thinking"}`，然后运行：

```bash
python -m visual_scaffolding evaluate \
  --task burden_counting \
  --benchmark benchmark \
  --adapter examples.hf_adapter:create \
  --model-config model.json \
  --output outputs/counting.jsonl
```

命令会写出逐题预测以及相邻的 `.summary.json`；已有输出不会被覆盖。

<details>
<summary>运行任务专用的 Visual Scaffolding 模型</summary>

两个模型分别针对不同任务训练。先安装上文的推理依赖，再下载所需任务的模型并运行：

```bash
hf download ShaoRun/vscaffold-4b-burden-counting --local-dir checkpoints/burden_counting
python -m visual_scaffolding evaluate \
  --task burden_counting \
  --benchmark benchmark \
  --checkpoint checkpoints/burden_counting \
  --device cuda:0 \
  --output outputs/vscaffold-counting.jsonl
```

Path Tracing 使用 `ShaoRun/vscaffold-4b-path-tracing`、`--task path_tracing` 和另一个输出路径。模型目录包含完整权重、处理器、分词器及 Visual Scaffolding 配置。

</details>

## 📄 许可

代码使用 [Apache 2.0](LICENSE)。Benchmark 图像组合与标注使用 CC BY-SA 4.0。示例图中的 OpenMoji 和 Twemoji 素材保留各自的许可，详见 [DATA_LICENSE.md](DATA_LICENSE.md) 与 [NOTICE.md](NOTICE.md)。
