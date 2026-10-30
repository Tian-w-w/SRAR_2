# SRAR_2

独立实现 `SRAR_2.md` 的位置 × 系数联合优势预测。原 SRAR 代码和结果保持原目录。本仓库不包含模型、数据、旧结果或训练权重；**当前只完成本地工程验证，尚无真实 Qwen 实验成绩**。

实现范围：Qwen2-7B Base、BF16、冻结 0–27 层与任务 query heads、native SDPA；BoolQ/SST-2/LogiQA 可运行入口。LITM 配置已提取，但旧观察器硬编码 0.9，与冻结 0.95 不一致，LITM 输入与旧特征 parity 尚未通过，运行入口会明确拒绝。BABILong/RULER 的任务适配仍待实现，不能据通用模块已就绪宣称六任务实验完成。

## 已实现

- 共享 944 维联合 Router，输出 `[B,3,3]`，参数 1,635,169；独立原结构 Router 为 1,631,025。六投影只执行一次，系数 embedding 后广播评分。
- 稳定动作 ID、BASE 优先平局、网格/schema/checkpoint 绑定；完整九动作效用与优势、等价动作核验、缺失动作显式拒绝。
- SmoothL1 + 0.5 BCE + 0.2 RankNet；有效动作掩码、等价对屏蔽、空排序可求导零值。排序先在每对的有效样本上平均，再平均有效动作对。
- train-only scaler、正/非正组平衡采样、AdamW、dev tau=0 早停、验证集严格 `score > tau` 校准、完整阈值前沿与全触发/全保留端点。
- J、独立重训 B2、每系数独立重训 B3、B4、B5、J-BASE-mask、J-Always 的验证集诊断。
- 独立生成与初始特征捕获、原子缓存和行级恢复、失败记录、来源校验；在线 API 不接收参考答案或效用，最多初始加一次候选生成。
- GPU smoke 对照 AST 提取的旧短任务实现，检查初始/BASE token 与特征完全相同、状态重置、expanded K 选择性、cache 不受污染以及 1/2 次生成预算。**该 smoke 尚未在 GPU 执行。**

## CPU 验证

无需下载 Backbone。现有环境可直接运行：

```powershell
cd D:\Codex_Project\srar2
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q
$env:PYTHONPATH = "$PWD\src"
python -m srar2 --help
python -m srar2 audit
python -m srar2 generate-counterfactuals --dry-run
```

新环境可先 `python -m pip install -e ".[test]"`。`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免加载主机其他项目的 pytest 插件；本项目不依赖第三方 pytest 插件。合成端到端测试的数据只用于工程检查，不是实验结果。

## 单张 RTX 5090

预计可顺序完成 Qwen2-7B BF16 的短任务及轻量 Router 训练。RTX 5090 为 32GB 显存，约 7.6B 参数的 BF16 权重约 15.2GB；本项目只训练 MLP，不对 Backbone 反传。单样本 KV 理论开销为 `2 × 28 × 4 × 128 × 2 = 57,344 bytes/token`，8K 约 448MiB，另有 expanded K、全部 hidden states、临时张量、观察器和 CUDA 开销。长上下文能否放下须以实测峰值为准，不保证所有长度都适用，也不靠量化改变 BF16 协议。

使用支持 Blackwell 的 PyTorch CUDA 构建。以下是一个明确的待 GPU 验证组合，不能当成本机已验证环境：

```bash
git clone https://github.com/Tian-w-w/SRAR_2.git
cd SRAR_2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -c "import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0)); print(torch.ones(2,device='cuda')+1)"
```

固定 Transformers 4.53.2 是因为继承的 native SDPA 实现以该版本为参照。旧服务器快照为 PyTorch 2.13.0 / Transformers 4.53.2；新环境与旧环境的差异要披露，且所有新对照在同一新环境运行并通过 token parity，不直接混用旧成绩。

硬件与兼容性依据：[NVIDIA RTX 5090](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/geforce-rtx-5090-founders-edition/)、[PyTorch Blackwell/CUDA 12.8 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

## 复用原 SRAR 数据划分进行 BoolQ 小规模实验

先把原 SRAR 数据和模型搬到 GPU 服务器，或将 `--legacy-root` 指向已有旧项目。以下历史路径来自 `SHORT_DATASET_REGISTRY.json`，本机没有这些文件，不能视为已经核验服务器实际存在：

```text
<legacy-root>/runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_train_effective.jsonl
<legacy-root>/runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_validation.jsonl
<legacy-root>/runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_formal.jsonl
<legacy-root>/models/modelscope/Qwen/Qwen2-7B/
```

只预览命令，不加载模型或数据：

```bash
python tools/run_boolq_smoke.py \
  --legacy-root /root/zerotuning_adaptive \
  --model-path /root/zerotuning_adaptive/models/modelscope/Qwen/Qwen2-7B \
  --samples 8 --dry-run
```

实际启动：

```bash
python tools/run_boolq_smoke.py \
  --legacy-root /root/zerotuning_adaptive \
  --model-path /root/zerotuning_adaptive/models/modelscope/Qwen/Qwen2-7B \
  --samples 8
```

按顺序导入原 train/dev manifest（不重划分）、校验分组、取各自前 8 条作为工程 smoke 子集、运行 2 条真实 parity、生成 16×10=160 次离线答案、保存 oracle，再训练 J 和 3 个固定系数评分器并作验证比较。parity 另需 15 次生成和 1 次张量审计 forward/样本；与离线 160 次分开计费。初始特征只来自固定 `(0,gamma0)`。默认网格 `[0.95,0.76,1.14]`，不会按 Formal 改动。

该脚本不读取 Formal，不自动开展正式评测。8 条 smoke 训练也使用方案的最多 50 epochs/耐心 7，Selector 在 CPU 可运行；只能证明链路，不支持统计结论。断点恢复可加 `--resume`；代码、配置、模型、数据哈希变化时必须使用新输出目录。多进程分片必须分别写唯一统计输出，或使用各自运行目录再审计合并；首版一张卡按顺序执行。

旧 prepared manifest 常没有 `source_question_id`，导入器从原 query span 恢复问题文本（SST-2 从 context span 恢复句子），保留 `sample_id`、原顺序、原 split、源文件 hash。若发现原划分有同问题跨集合，工具会报错，不能静默过滤后把它称为原实验。LITM 多排列必须保留独立 row_uid，并按原问题分组。

## 单独调用各步骤

`python -m srar2 --help` 展示所有入口。模型路径和数据文件通过参数传入；默认配置的 null 路径是明确缺失项，不会回退到自动下载或默认数据。所有输出限制在独立 `--project-root` 内，默认本仓库。

```bash
python -m srar2 prepare-manifest --input-rows /path/boolq_validation.jsonl \
  --model-path /path/Qwen2-7B --split dev --output "$PWD/data/dev.jsonl"
python -m srar2 generate-counterfactuals --manifest "$PWD/data/dev.jsonl" \
  --model-path /path/Qwen2-7B --split dev --max-samples 8 \
  --output "$PWD/runs/pilot" --dry-run
```

`extract-features` 与 counterfactual 生成共用入口，确保特征与 y0 来自同一次初始计算；它也生成动作标签，并非额外单独跑九次特征。`--max-samples 0` 表示整个 manifest。训练强制要求记录完整覆盖传入 manifest，不能对大 manifest 只生成前 8 条然后静默当成完整数据；使用脚本产生的独立 smoke 子集。

`train --kind fixed --coefficient 0/1/2` 逐个训练 B2/B3 对照；所有候选仍相对同一个 y0。`analyze` 独立校准 J/B2/B3/B4/B5 的门控，并给出 oracle 与闭合诊断。它是离线验证集分析，**不能用其耗时作为在线延迟**。B3/B4 验证选优以及搜索成本会保留在报告中。

完成单任务全部 train/dev 和三个种子以前，不建议冻结 Formal。具备条件时：

```bash
python -m srar2 freeze --checkpoint "$PWD/runs/joint.pt" \
  --calibration "$PWD/runs/calibration.json" --output "$PWD/runs/frozen.json"
python -m srar2 infer --manifest /path/normalized_formal.jsonl --split formal \
  --checkpoint "$PWD/runs/joint.pt" --frozen "$PWD/runs/frozen.json" \
  --model-path /path/Qwen2-7B --max-samples 0 --output "$PWD/runs/formal.jsonl"
python -m srar2 evaluate --manifest /path/normalized_formal.jsonl \
  --predictions "$PWD/runs/formal.jsonl"
```

`freeze` 绑定 checkpoint、scaler（在 checkpoint 内）、gamma grid、action order、tau、协议、代码、模型/tokenizer 及运行身份；`infer` 逐样本执行新生成，评分在 `evaluate` 独立进行。切换不同 GPU/依赖版本必须建立新运行和 parity，首版不静默接受跨设备 cache 迁移。

## 当前限制和证据

审计见 [PROTOCOL_AUDIT.md](reports/PROTOCOL_AUDIT.md)，原文件 SHA256 见 [source_manifest.json](reports/source_manifest.json)。新代码沿用冻结 heads/gamma0，没有重新 profiling。GPU smoke 未执行，因此初始特征与原实现的真实模型 parity、实际显存、速度和新增 oracle 空间仍未知。

首版暂不实现：长任务运行适配、LITM 聚类 bootstrap/RULER 分层 bootstrap、所有对照的 Formal 统一执行、正式三种子研究、远程任务调度、跨版本数值容差放宽。RULER 宏/微平均已有独立测试，但这不代表 RULER 全链路可运行。

原生成器与官方评分 helper 的选定函数经 AST 抽取，原 operator 字节复制，来源和 hash 均保留；它们是项目已有第三方派生代码。本仓库没有擅自为这些继承文件增加新的开源许可证。若对外作开源发布，需核实原项目授权与第三方许可。
