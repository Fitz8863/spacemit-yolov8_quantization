# SpaceMiT YOLOv8 VOC 量化与精度评估

本项目用于在 **PASCAL VOC 20 类目标检测数据集**上训练/导出 YOLOv8 模型，并通过 SpaceMiT `xslim` 对 ONNX 模型进行 PTQ（Post-Training Quantization，训练后量化）。仓库同时提供：

- VOC 训练集、验证集和量化校准集；
- YOLOv8n / YOLOv8s 的 FP32 与量化 ONNX 模型；
- 与训练预处理一致的 `RGB + letterbox + /255` 校准代码；
- 单图推理、画框及 mAP 精度评估脚本；
- 量化前后模型精度对比所需的完整流程。

> 当前模型输出为 `[1, 24, 8400]`，其中 `24 = 4` 个边界框参数 `+ 20` 个 VOC 类别分数。

## 目录结构

克隆并还原数据集后，主要目录如下：

```text
spacemit-yolov8/
├── README.md
├── requirements.txt
├── config.json                       # xslim 量化配置
├── preprocess.py                     # xslim 校准预处理入口
├── generate_data_txt.py              # 生成校准图片路径列表
├── test.py                           # ONNX 单图推理并画框
├── eval_map.py                       # FP32/量化 ONNX mAP 评估
├── bus.jpg                           # 单图推理示例
├── models/                           # YOLOv8n FP32/量化模型与报告
├── models_yolov8s/                   # YOLOv8s FP32/量化模型与报告
├── VOC/
│   ├── VOC.yaml                      # Ultralytics 数据集配置
│   ├── train/
│   │   ├── images/                   # 16,551 张训练图片
│   │   └── labels/                   # YOLO 格式训练标注
│   ├── val/
│   │   ├── images/                   # 4,952 张 VOC2007 test 图片
│   │   └── labels/                   # YOLO 格式验证标注
│   └── quantify_dataset/             # 400 张 PTQ 校准图片
└── datasets/                         # Git LFS 管理的数据集归档分片
```

## Python 文件说明

### `preprocess.py`

提供给 `xslim` 的量化校准预处理函数：

```python
preprocess_impl(path_list, input_parametr)
```

处理顺序为：

1. 按模型输入尺寸做居中的 letterbox，保持宽高比，填充值为 `114`；
2. OpenCV BGR 转 RGB；
3. 像素除以 `255`，转换到 `[0, 1]`；
4. HWC 转 NCHW；
5. 返回 `float32 torch.Tensor`。

这个预处理应与模型训练和精度评估保持一致。除非模型明确使用了 BGR 或拉伸 resize，否则不要更改。

### `generate_data_txt.py`

扫描校准集中的图片并生成绝对路径列表，供 `config.json` 的 `data_list_path` 使用：

```bash
python generate_data_txt.py \
  VOC/quantify_dataset \
  VOC/quantify_dataset/data.txt
```

由于 `data.txt` 中保存的是当前机器上的绝对路径，**每次换机器、移动仓库或解压数据集后都应重新生成**。

### `test.py`

使用 ONNX Runtime 对单张图片推理，并完成：

- letterbox + RGB + `/255` 预处理；
- YOLOv8 输出解码；
- 按类别执行 NMS；
- 将方框恢复到原图坐标；
- 绘制类别、置信度和边界框并保存图片。

脚本既可读取量化模型，也可读取同接口的 FP32 ONNX 模型。

### `eval_map.py`

在 YOLO 格式数据集上评估 ONNX 模型，支持 FP32 和量化模型。数据集需要满足：

```text
<dataset>/images/xxx.jpg
<dataset>/labels/xxx.txt   # class_id x_center y_center width height，均为归一化坐标
```

评估过程会将模型预测的 **类别和方框** 与 label 中的真实目标进行一对一 IoU 匹配，并将以下情况计入结果：

- 正确类别且 IoU 达标：TP；
- 类别识别错误：预测类别产生 FP，真实类别未召回；
- 方框 IoU 不达标：FP，同时真实目标未召回；
- 同一真实目标的重复预测：除一个匹配框外，其余为 FP；
- 多检：FP；
- 漏检：降低 Recall，进而降低 AP。

输出指标包括每个类别及整体的：

- `AP@0.5`；
- `AP@0.5:0.95`；
- GT 数量；
- 总耗时和 CPU 吞吐率。

当前指标采用现代 YOLO/COCO 风格的 PR 曲线与多 IoU 阈值评估。它适合比较 FP32 与量化模型的精度变化，但不等同于严格的 VOC2007 DevKit 11 点 AP；YOLO txt 标注也不保存 VOC XML 中的 `difficult` 忽略框。

## 模型文件

| 目录 | 模型 | 类型 |
|---|---|---|
| `models/` | `yolov8n_relu_110.onnx` | YOLOv8n FP32 |
| `models/` | `yolov8n_relu_110.q.onnx` | YOLOv8n 量化 |
| `models/` | `yolov8n_silu_110.onnx` | YOLOv8n FP32 |
| `models/` | `yolov8n_silu_110.q.onnx` | YOLOv8n 量化 |
| `models_yolov8s/` | `yolov8s_relu_50.onnx` | YOLOv8s FP32 |
| `models_yolov8s/` | `yolov8s_relu_50.q.onnx` | YOLOv8s 量化 |
| `models_yolov8s/` | `yolov8s_silu_50.onnx` | YOLOv8s FP32 |
| `models_yolov8s/` | `yolov8s_silu_50.q.onnx` | YOLOv8s 量化 |

`.q_report.md` 为对应模型的量化报告。

## 数据集

| 数据 | 路径 | 数量 | 用途 |
|---|---|---:|---|
| 训练集 | `VOC/train` | 16,551 张图片 + 16,551 个 label | YOLOv8 训练 |
| 验证集 | `VOC/val` | 4,952 张图片 + 4,952 个 label | mAP 评估 |
| 校准集 | `VOC/quantify_dataset` | 400 张图片 | xslim PTQ 校准 |

验证集包含 12,032 个非 `difficult` GT 实例，类别编号顺序如下：

```text
0 aeroplane    1 bicycle      2 bird          3 boat
4 bottle       5 bus          6 car           7 cat
8 chair        9 cow         10 diningtable  11 dog
12 horse      13 motorbike   14 person       15 pottedplant
16 sheep      17 sofa        18 train        19 tvmonitor
```

数据来源及使用条件请遵循 PASCAL VOC 官方说明；仓库中的标注已转换为 Ultralytics 使用的 YOLO txt 格式。

## 获取仓库和还原数据集

数据集总大小约 2.3 GB，包含 4 万多个小文件。为避免普通 Git 仓库历史膨胀，本项目将 ONNX、大文件及数据归档交给 **Git LFS** 管理，并把训练集拆成小于 1 GiB 的分片。

### 1. 安装 Git LFS 并克隆

```bash
git lfs install
git clone git@github.com:Fitz8863/spacemit-yolov8.git
cd spacemit-yolov8
git lfs pull
```

如果克隆时跳过了 LFS 下载，可在仓库中重新执行：

```bash
git lfs pull
```

### 2. 还原数据集

在仓库根目录运行：

```bash
(cd datasets && sha256sum -c SHA256SUMS)
cat datasets/voc_train.tar.part-* | tar -xf -
cat datasets/voc_val.tar.part-* | tar -xf -
cat datasets/voc_quantify_dataset.tar.part-* | tar -xf -
```

检查数量：

```bash
find VOC/train/images -type f | wc -l
find VOC/train/labels -type f | wc -l
find VOC/val/images -type f | wc -l
find VOC/val/labels -type f | wc -l
find VOC/quantify_dataset -maxdepth 1 -type f -iname '*.jpg' | wc -l
```

预期输出依次为：

```text
16551
16551
4952
4952
400
```

然后重新生成本机校准列表：

```bash
python generate_data_txt.py VOC/quantify_dataset VOC/quantify_dataset/data.txt
```

## 环境配置

本项目当前在远程服务器的 `xslim` Conda 环境中使用：

```bash
ssh heweijie@10.0.50.15
source /home/heweijie/WorkSpace2/miniconda3/etc/profile.d/conda.sh
conda activate xslim
cd /data2/home2/heweijie/WorkSpace/project/yolo_quan
```

新环境可以参考：

```bash
conda create -n xslim python=3.12 -y
conda activate xslim
pip install -r requirements.txt
pip install xslim
```

如果只做 CPU ONNX 推理和 mAP 评估，可以不安装 CUDA；`eval_map.py` 和 `test.py` 会自动选择可用的 ONNX Runtime provider。

## 使用方法

### 1. Ultralytics 训练示例

安装 Ultralytics：

```bash
pip install ultralytics
```

使用仓库内数据集配置训练：

```bash
yolo detect train \
  model=yolov8n.pt \
  data=VOC/VOC.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0
```

这里是通用示例。仓库中现有的 ReLU/SiLU 模型可能来自额外的网络结构或激活函数修改，复现时还需要使用对应的训练代码和超参数。

导出 ONNX 的通用示例：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=17
```

### 2. xslim 量化

先修改 `config.json` 中的 FP32 ONNX 路径，例如：

```json
"onnx_model": "models/yolov8n_silu_110.onnx"
```

确保校准列表已经针对当前机器重新生成：

```bash
python generate_data_txt.py VOC/quantify_dataset VOC/quantify_dataset/data.txt
```

执行量化：

```bash
python -m xslim --config ./config.json
```

量化前后必须保持预处理一致。本仓库默认统一使用：

```text
letterbox + RGB + float32 / 255 + NCHW
```

### 3. 单图推理和画框

量化模型：

```bash
python test.py \
  --model models/yolov8n_silu_110.q.onnx \
  --image bus.jpg \
  --output bus_result_silu.jpg \
  --conf 0.25 \
  --iou 0.45
```

FP32 模型：

```bash
python test.py \
  --model models/yolov8n_silu_110.onnx \
  --image bus.jpg \
  --output bus_result_fp32.jpg
```

如果模型明确以 BGR 训练，才添加 `--bgr`；Ultralytics 默认模型通常不需要。

### 4. FP32 模型完整精度评估

```bash
python eval_map.py \
  --model models/yolov8n_silu_110.onnx \
  --dataset VOC/val \
  --conf 0.001 \
  --iou 0.7 \
  --workers 4 \
  --ort-threads 1 \
  --opencv-threads 1 \
  --save-report models/yolov8n_silu_110_fp32_eval.md
```

### 5. 量化模型完整精度评估

```bash
python eval_map.py \
  --model models/yolov8n_silu_110.q.onnx \
  --dataset VOC/val \
  --conf 0.001 \
  --iou 0.7 \
  --workers 4 \
  --ort-threads 1 \
  --opencv-threads 1 \
  --save-report models/yolov8n_silu_110_int8_eval.md
```

### 6. 快速冒烟测试

只评估前 30 张图片：

```bash
python eval_map.py \
  --model models/yolov8n_silu_110.onnx \
  --dataset VOC/val \
  --limit 30
```

`--limit` 只适合检查流程是否正常，不能作为最终精度结果。

## 评估参数建议

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `--conf` | `0.001` | mAP 需要完整 PR 曲线，阈值应保持较低 |
| `--iou` | `0.7` | NMS IoU 阈值，不是 TP 匹配阈值 |
| `--max-det` | `300` | 每张图片最多保留的检测数量 |
| `--workers` | `4` | CPU 并行图片 worker，可按服务器核心数调整 |
| `--ort-threads` | `1` | 每个 worker 的 ORT 线程数，量化模型建议保持 1 以便复现 |
| `--opencv-threads` | `1` | 多 worker 时避免 OpenCV 线程过度竞争 |
| `--letterbox` | 开启 | 与默认训练和量化校准预处理一致 |
| `--bgr` | 关闭 | 默认输入 RGB |

为了公平比较 FP32 和量化模型，两次评估必须使用同一数据集、预处理、NMS 参数、置信度阈值和类别顺序。

量化精度变化可以按下面方式理解：

```text
ΔmAP@0.5      = INT8 mAP@0.5      - FP32 mAP@0.5
ΔmAP@0.5:0.95 = INT8 mAP@0.5:0.95 - FP32 mAP@0.5:0.95
```

## 常见问题

### `config.json` 找不到校准图片

`data.txt` 是绝对路径列表。切换机器或目录后重新执行：

```bash
python generate_data_txt.py VOC/quantify_dataset VOC/quantify_dataset/data.txt
```

### FP32 和量化模型能否使用同一个评估脚本？

可以。只要两个 ONNX 模型的输入输出接口一致，`eval_map.py` 不依赖 `.q.onnx` 文件名，也没有量化模型专用的统计分支。

### 为什么评估使用 `conf=0.001`，单图演示使用 `conf=0.25`？

单图展示更关心画面整洁，因此可以过滤低置信度框；mAP 需要根据所有置信度阈值形成 PR 曲线，因此应保留更多候选框。

### 为什么推荐 `workers=4, ort-threads=1`？

脚本为每个 worker 创建独立 ONNX Runtime session，通过图片级并行提高 CPU 吞吐率。每个 session 保持单线程可以减少过度订阅，并提高量化结果的复现一致性。

## Git LFS 维护

查看 LFS 文件：

```bash
git lfs ls-files
```

只克隆代码、不立即下载大文件：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Fitz8863/spacemit-yolov8.git
```

随后按需下载：

```bash
git lfs pull --include='models/**'
git lfs pull --include='datasets/voc_val.tar.part-*'
```

请注意 GitHub 账户的 LFS 存储和流量配额；数据集更新时不要反复提交仅有时间戳变化的归档文件。
