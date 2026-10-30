# SRAR_2 实验与实现方案

> 文档日期：2026-10-29。供没有前文上下文的新会话独立阅读并实施。
> 本文定义一个尚未实现、尚无实验结果的新方案。所有建议的网格、种子和实验规模均为计划，不能写成已完成事实。
> SRAR_2 是内部项目名称，暂不代表最终论文标题或已确认的新颖性。

## 0. 给接手会话的任务说明

用户已有 SRAR 工作，目前在一个会议审稿中，希望最大限度复用现有实现，开展一个实质性的后续研究。用户已经选择的方向是：**除预测候选锚点位置之外，还预测干预系数，通过联合操作的优势估计选择位置与系数组合。**

请依据本文件实现并验证该方案。不要把任务换成自适应候选数量、跨任务迁移、自适应层选择或单纯改网络结构。

第一版的明确边界：

- 优先支持 Qwen2-7B Base，保持原有 native SDPA 干预算子。
- 保留原有 RE、RM、RL 三个候选位置及任务对应的坐标构建逻辑。
- 保留按任务冻结的干预层和 query head 集合，不重新搜索 heads。
- 初始输出仍由固定的 `(initial_anchor, gamma0)` 产生。
- 每个区域位置增加少量离散系数选项，第一版每任务三个系数，并包含原系数 `gamma0`。
- 使用一个共享评分器预测所有 `(anchor, gamma)` 操作的优势。
- 保留原有“无收益则保留初始输出”的门控，重新在验证集校准阈值。
- 在线仅生成初始答案和至多一个选中的区域候选答案，不在线枚举九个答案。
- 所有旧结果与旧运行目录保留；新实现、新配置、新缓存和新结果独立存储。

先执行源码定位、协议审计、CPU 单元测试和小规模预实验，再扩展正式实验。遇到一个任务的历史材料不齐，可继续实现通用模块、测试和其他材料完整的任务；不要虚构缺失数据，也不要因为旧论文溯源未全部完成就放弃可独立推进的工作。

本文件是实现与研究计划，不授权对外提交论文、发送消息或覆盖旧实验。新会话收到用户实施指令后，在该指令范围内执行；远程服务器连接、计算资源与实际路径须现场确认。

## 1. 原 SRAR 是什么

原 SRAR（Sample-Adaptive Attention Rerouting）的 Backbone 冻结，只训练一个轻量 MLP Selector。

对输入 `x`：

1. 在准备后的输入 token 0，即初始锚点 `c0`，执行冻结配置下的干预并生成 `y0`。
2. 从这次初始计算提取隐藏状态、注意力统计和输出不确定性等特征。
3. 上下文的早、中、晚三个区域分别给出候选锚点 RE、RM、RL，近似位于上下文内部的 `1/6、1/2、5/6`。
4. Selector 预测三个候选输出相对初始输出的优势。
5. 最大预测优势严格大于验证集冻结的阈值 `tau`，才在选定候选上再生成一次；否则返回 `y0`。

原工作的主要实验：两种 Backbone、六个任务的主比较；锚点敏感性；Rescue/Harm/Unchanged；Initial、Always Reroute、Random Anchor、Full SRAR 的消融。门控与 Rescue/Harm 已经是原有内容，不能当作 SRAR_2 新贡献。

SRAR_2 的新问题是：**在相同候选位置下，不同样本是否需要不同干预系数；位置与系数是否需要联合选择，而不是分开决定？**

## 2. 工作区、参考代码与证据等级

### 2.1 已知路径

本地 Windows 工作区：`D:/Codex_Project`。

历史脚本中记录的 Linux 项目根目录：`/root/zerotuning_adaptive`。

历史 Qwen 模型目录：`/root/zerotuning_adaptive/models/modelscope/Qwen/Qwen2-7B`。

这些服务器路径是历史线索，接手会话必须检查实际存在性与模型身份，不得假设已经连接服务器。

### 2.2 建议先读的文档

以下文件在制定本方案时已确认本地存在：

- `D:/Codex_Project/srar_share_package_20261020/README.md`
- `D:/Codex_Project/srar_share_package_20261020/RELEASE_SCOPE.md`
- `D:/Codex_Project/srar_share_package_20261020/training_calibration/TRAINING_CALIBRATION_CHAIN.md`
- `D:/Codex_Project/srar_share_package_20261020/main_experiments/MAIN_RESULTS_CHAIN.md`
- `D:/Codex_Project/srar_share_package_20261020/analyses/ANALYSES_CHAIN.md`

分享包是裁剪版：保留训练、推理、门控阈值校准等代码，但移除了历史干预参数搜索等内容，一些预测参考文件与数据缓存需要另行恢复。修改后的分享包哈希不等于历史执行文件哈希。包内 README 的“存在链路”也不等于正式结果已通过完整复现。

`D:/Codex_Project` 下还存在 `srar_code_package_20261020`、`srar_inference_calibration_package_20261020`、`srar_paper_export` 等副本。不要按目录时间或文件名直接认定某副本权威。建立所选源码与运行配置的哈希清单。

### 2.3 核心源码线索

| 用途 | 本地参考路径 |
|---|---|
| Qwen 短任务全链路、Router、Observer | `D:/Codex_Project/stage26q2_short_base_srar.py` |
| 短任务准备、prompt 和坐标 | `D:/Codex_Project/stage26q2_short_prepare.py` |
| Qwen 干预算子 | `D:/Codex_Project/srar_paper_export/scripts/qwen2_zerotuning_sdpa_native_semantics_v2.py` |
| LITM Selector 与阈值校准 | `D:/Codex_Project/srar_paper_export/scripts/stage26q2_litm_srar_selector.py` |
| LITM 初始特征 | `D:/Codex_Project/srar_paper_export/scripts/stage26q2_litm_srar_features.py` |
| LITM 离线链路与正式推理 | 同目录 `stage26q2_litm_srar_p1.py`、`stage26q2_litm_srar_formal.py` |
| BABILong 链路 | 同目录 `stage26q2_babilong_srar_counterfactual.py`、`stage26q2_babilong_srar_features.py`、`stage26q2_babilong_srar_selector.py`、`stage26q2_babilong_srar_formal.py` |
| RULER 链路 | 同目录 `stage26q2_ruler_srar_counterfactual.py`、`stage26q2_ruler_srar_features.py`、`stage26q2_ruler_srar_selector.py`、`stage26q2_ruler_srar_formal.py` |
| 救援与损害统计参考 | `D:/Codex_Project/audit_qwen2_srar_rescue_harm.py` |
| 原消融协议与汇总参考 | `D:/Codex_Project/preflight_qwen2_srar_core_ablation.py`、`D:/Codex_Project/finalize_qwen2_srar_core_ablation.py` |

Qwen 原短任务源码中重点检查：`Router`、`attention_summaries`、`Observer`、`make_features`、`generate_one`、`counterfactual_worker`、`rank_loss`、`train_selector`、`calibrate`、`formal_worker`。

参数配置参考目录：

```text
D:/Codex_Project/srar_share_package_20261020/resources/zerotuning_adaptive/runs/
  stage26q2_short_base_srar/{sst2,boolq,logiqa}/selector/frozen_config.json
  stage26q2_litm_srar/frozen_config_parent.json
  stage26q2_babilong_srar/frozen_config_parent.json
  stage26q2_ruler_srar/frozen_config_parent.json
```

优先提取运行所需的最终值，不依赖旧文件内部的绝对路径或旧 seal 状态来跳过核验。

## 3. 方法定义与固定范围

### 3.1 初始参照

令 `F` 为冻结模型，`omega` 包含层、头、算子、tokenization、解码等固定设置；公式中省略 `omega`。

\[
y_0=F(x;c_0,\gamma_0),\qquad U_0=S_t(y_0,r_x).
\]

- `c0` 是准备后输入的 token 0，不能无条件称为 BOS。
- `gamma0` 是该任务经过溯源确认的原冻结系数。
- 同一新实验中的所有组合与对照共享这一初始参照；不因候选系数改变而重新定义 `U0`。
- 首版不自适应初始系数，不把 `(c0, alternative_gamma)` 纳入主方法动作集。
- 无干预 Vanilla 与初始干预结果不同，报告时分开。

### 3.2 候选动作与监督

\[
\mathcal C(x)=\{c_{\mathrm{RE}},c_{\mathrm{RM}},c_{\mathrm{RL}}\},\quad
\Gamma_t=\{\gamma_0,\gamma_-,\gamma_+\}.
\]

\[
\mathcal A_t(x)=\mathcal C(x)\times\Gamma_t.
\]

对每个动作离线生成并评分：

\[
y_{i,j}=F(x;c_i,\gamma_j),\qquad
A_{i,j}=S_t(y_{i,j},r_x)-U_0.
\]

保留完整 `3 × M` 效用／优势矩阵及平局。不要只保留一个最优动作标签，不从初始答案或其他系数的输出推算新动作得分。

预测与选择：

\[
\widehat A_{i,j}=f_\theta([z_i(x);e_j]),\qquad
(i^*,j^*)=\arg\max_{i,j}\widehat A_{i,j}.
\]

\[
y_{\mathrm{final}}=
\begin{cases}
F(x;c_{i^*},\gamma_{j^*}),&\max_{i,j}\widehat A_{i,j}>\tau,\\
y_0,&\text{otherwise}.
\end{cases}
\]

保留初始输出是显式动作，通过门控实现，不是第十个需要生成的区域候选。

### 3.3 Qwen 算子必须保持的语义

Qwen2-7B Base、BF16、native SDPA。选定层中，将 KV heads 重复至 query-head 数目之后，克隆 expanded K，仅将选中 query heads 对应的锚点 K 向量乘以 `gamma`；之后交给原 SDPA。不要原地污染 pre-repeat KV cache。

候选动作间只改变锚点与系数，层与头保持冻结。每个动作的生成都应从该输入独立执行，不能把初始计算的 KV cache 原样拿来当新位置／新系数下的有效 cache。

初始特征观察器只能观察固定 `(c0, gamma0)` 的计算，不能按每个候选系数额外运行 Backbone 再取特征。

固定局部 query/key 时，K 缩放的指数权重因子为：

\[
\rho=\exp((\gamma-1)z_c),\quad z_c=q^\top k_c/\sqrt{d_h}.
\]

因此系数增大不保证锚点概率增大，也不保证任务收益增加。使用“较小／原始／较大系数”，不要无条件称“弱／中／强关注”。这里没有单调性、最优性或准确率保证。

### 3.4 首版不做的扩展

不更换 Backbone 权重、不微调 Backbone、不改变三个区域定义、不新增多锚点联合干预、不搜索层与头、不引入在线答案枚举、不改成两个独立分类头、不直接回归未评估过的连续系数。

## 4. 原协议参考与必须现场核实的内容

### 4.1 Qwen 历史参考值

以下是历史记录线索，不是新实验的已验证结果；实际 manifest、有效特征数和冻结配置优先。

| 任务 | Selector 训练数 | 验证数 | Formal 数 | 原 gamma0 | 原 SRAR 最终回答上限 |
|---|---:|---:|---:|---:|---|
| SST-2 | 4,999 | 1,000 | 500 | 0.50 | 1 token |
| BoolQ | 5,000 | 1,000 | 500 | 0.95 | 8 tokens |
| LogiQA | 4,996 | 1,000 | 500 | 0.95 | 8 tokens |
| LITM | 2,000 | 1,194 | 1,197 | 0.95 | 100 tokens |
| BABILong | 2,000 | 1,000 | 4,371 | 1.05 | 25 tokens |
| RULER | 2,000 | 1,040 | 1,040 | 0.95 | 子任务独立上限 |

RULER 历史上限：NIAH 128；VT 30；CWE 120；FWE 50；QA1/QA2 各 32。正式指标是 13 子任务等权宏平均，不是全部样本简单平均。

所有任务最终回答使用任务 raw prompt，无 chat template；贪心生成 `do_sample=False`、`num_beams=1`，SRAR 使用 KV cache。不要擅自补写脚本未显式设置的 sampling 参数。

历史 Qwen 干预层为 0–27。query head 列表：

| 任务 | heads |
|---|---|
| SST-2 | 23,26,24,7,1,27,5,16,0,17,13,25 |
| BoolQ | 23,26,24,27,25,1,3,10,0,5,20,11 |
| LogiQA | 23,26,3,1,24,5,25,27,22,0,10,9 |
| LITM | 23,26,24,3,27,25,9,1,13,0,5,15 |
| BABILong | 5,2,27,10,24,17,23,26,7,16,11,22 |
| RULER | 3,23,26,24,0,5,9,1,11,25,22,13 |

禁止因为加入系数预测而重新 profiling heads，然后将全部收益归因于联合预测。

### 4.2 数据完整性

- LITM 的 1,197 条记录来自 399 个问题的三个文档排列。按 manifest row 唯一标识配对，不能按重复 question/sample ID 去重。
- 检查原始问题是否跨训练、验证、测试集；仅 row ID 不同不代表无泄漏。若需要修订划分，保存新版 manifest，所有比较方法重跑，不能与旧表混用。
- BABILong 优先检查是否确为同一 valid4371 队列。
- 样本缺失、解析失败、非有限特征必须计数并按预先规则处理，不能静默过滤提高得分。
- 所有数值结果从逐样本记录自动计算，不能以论文展示值作为测试目标或训练信号。

### 4.3 历史问题清单：不能自动认定已解决，也不能认定最新包一定仍有问题

1. 早期导出 LITM 特征观察器曾出现硬编码 `0.9`，而冻结主算子为 `0.95`。需检查实际选定源码和执行记录，保证观察与初始推理使用相同 gamma0。
2. Qwen 短任务通常 `add_special_tokens=False`，部分长任务导出脚本为 True。实际模型输入 token IDs、prompt hash 与 anchor indices 是对齐依据，不能用一句“都不加 BOS”概括。
3. 旧稿 LITM 初始 61.4、SRAR 67.7，与触发子集 Rescue 13.3、Harm 7.8 在相同二值评分前提下不闭合。新实验独立生成闭合报告，不把该组展示值当正确目标。
4. 旧 RULER 展示舍入和正式宏平均／样本二值统计可能混用。分开记录主指标与诊断指标。
5. 新发现的 20261020 包含更多配置与训练链文档，早期“缺失”判断需现场复核。分析包描述也可能与旧稿图 3 的模型不同，不能按旧名称推断来源。
6. 分享版删除了部分历史参数搜索代码及资源。参考原运行文件或新建透明的新实验协议，不伪造历史 seal。

以上问题不要求复制旧稿的错误。必要时建立 `SRAR_ref_rerun` 作为同协议重跑的基线，并明确与历史稿件数字的区别。

## 5. 系数网格：给出可执行默认值，同时控制探索范围

### 5.1 首版默认网格

每任务先采用围绕 gamma0 的三个正系数：

\[
\Gamma_t=[\gamma_0,\;0.8\gamma_0,\;1.2\gamma_0].
\]

这是方案默认值，不是已调优设置或理论结论。对应历史参考值：

| gamma0 | 系数数组（顺序固定） |
|---:|---|
| 0.50 | [0.50, 0.40, 0.60] |
| 0.95 | [0.95, 0.76, 1.14] |
| 1.05 | [1.05, 0.84, 1.26] |

此前讨论中的 `[0.80,0.95,1.10]` 只是示例，不要求同时跑两套网格。首版先固定上述一套，避免隐性大量搜索。

如果小规模 train/validation 诊断表明范围没有可利用空间或导致严重退化，可在查看 Formal 之前修改，并记录所有尝试和计算成本。不得根据 Formal 的收益反复移动网格。

### 5.2 编号与去重

- `anchor_id`: RE=0、RM=1、RL=2。
- `coefficient_id`: BASE=0、LOW=1、HIGH=2；数值以配置为准。
- 输出张量采用 `[B, 3, M]`，存储和接口均明确维度。
- 持久化动作 ID 使用稳定字符串，例如 `RE:BASE`，不要依赖 float 字符串或隐式 flatten 顺序。
- 设置 `action_order` 显式列表；分数完全相同时，先 BASE 系数，再 RE/RM/RL，再 LOW/HIGH。
- 短文本可能有重复 token anchor。对等价动作可共享生成缓存，但必须保留标签映射并处理重复比较，不能生成冲突结果。
- 若未来网格含 gamma=1，不同锚点数学上可能都是恒等干预。验证实际算子分支与 Vanilla parity，规范化重复动作，不能把重复恒等项当作多个独立证据。

## 6. 网络接口与特征定义

### 6.1 现有特征

Qwen 观察层为零起始编号 14、27，hidden size 为 3,584。现有特征张量：

- `vectors`: `[B,10,3584]`。
- `scalars`: `[B,3,16]`。

已检查的 Router 向量索引顺序：

```text
0  query_mean, layer 14
1  query_mean, layer 27
2  initial_anchor, layer 14
3  initial_anchor, layer 27
4:7  RE/RM/RL, layer 14
7:10 RE/RM/RL, layer 27
```

不要仅因为文件都称“10 个向量”就认为序列一致，导入每种旧缓存时核对索引 schema。

16 个标量由每观察层五项候选注意力统计、四项首个生成 token 分布不确定性、prompt length/4096、候选 token index/max(1,L-1) 组成。五项统计的计算范围必须复用并验证现有实现；记录 causal mask 与 query span。

所有特征来自 `(c0,gamma0)` 的同一次初始计算。候选输出、候选 correctness、候选输出概率、参考答案、未来 token、Formal 标签不得进入输入特征。

### 6.2 共享联合评分器

保留六个从 3,584 维到 64 维的投影，得到两层拼接后的 q、b、ci，各 128 维。

保留原七个向量块的顺序：

```text
q, ci, b, q*ci, q*b, ci-b, q-ci
```

拼接 16 维标量与 16 维区域 embedding，形成原 928 维候选表示 `zi`。新增 `Embedding(M,16)`，拼接得到 944 维联合操作表示。

评分 MLP：输入 944 维，两个隐藏层分别为 256、64 维，输出一个实数；隐藏层使用 GELU 与 dropout 0.2。位置与系数组合全部共享该网络。

建议接口：

```python
scores = model(vectors, scalars_normalized)  # [B, 3, M]
```

先计算一次三候选表示，再广播至 M 个系数，避免重复投影或重复提取 Backbone 特征。

现有 Qwen 模型参数量记录为 1,631,025。按上述结构，M=3 时新增输入权重 16*256 与 3*16 个 embedding 参数，理论增量为 4,144；以实际模型 `numel()` 检验，不把此推导当作运行报告。

第一版使用离散 embedding，不声称可直接泛化到未训练过的系数。checkpoint 必须绑定系数值、顺序与 action schema；加载时发现不一致要报错。

### 6.3 初始化与回归测试

正式实验默认从头训练，避免仅新方法使用有利的预训练条件。可实现旧权重兼容加载用于 smoke：复制旧投影、区域 embedding、原 928 列第一层权重和后续层，新增 16 列置零。关闭 dropout 后，所有系数输出起初应等于旧位置分数。

该兼容加载只证明接口正确，不等于新方法已经学会系数效应。如果用于正式 warm start，须作为明确实验设置报告并给基线匹配条件。

## 7. 离线数据生成、缓存与 schema

### 7.1 生成流程

对 train/validation 每条 manifest row：

1. 固定初始配置生成 y0、计算 U0，同时提取并保存初始特征。
2. 对 RE/RM/RL 与 M 个系数的所有有效组合独立生成，保存原始文本、解析结果、任务效用。
3. 计算完整优势矩阵，验证 `A == U_candidates - U0`。
4. 对已验证兼容的旧 BASE 系数输出复用缓存；其他组合真实生成。

M=3 时，每样本为 1+3M=10 次完整生成；原方法为 4 次。若四个旧输出可以严格复用，则新增六次。只允许在模型、tokenizer、token IDs、解码、算子、层、头、锚点、gamma、评分版本全部一致时复用。

特征不必为九个动作复制九份；保存一份 vectors 和三份候选 scalars。缺失特征时重新提取，不能假设旧文本输出包含足够信息恢复隐藏状态。

### 7.2 建议记录结构

```json
{
  "schema_version": "srar2-v1",
  "dataset": "litm",
  "split": "train",
  "row_uid": "stable_manifest_row_identifier",
  "source_question_id": "original_group_identifier",
  "manifest_sha256": "...",
  "input_ids_sha256": "...",
  "protocol_sha256": "...",
  "feature_schema_sha256": "...",
  "gamma_values": [0.95, 0.76, 1.14],
  "anchor_token_indices": {"RE": 0, "RM": 0, "RL": 0},
  "initial": {
    "anchor_token_index": 0,
    "gamma": 0.95,
    "prediction_raw": "...",
    "prediction_parsed": "...",
    "utility": null
  },
  "actions": [
    {
      "action_id": "RE:BASE",
      "anchor_id": 0,
      "coefficient_id": 0,
      "gamma": 0.95,
      "prediction_raw": "...",
      "prediction_parsed": "...",
      "utility": null,
      "advantage": null,
      "generation_status": "pending"
    }
  ],
  "feature_file": "features/train/row_uid.npz"
}
```

上面是格式示意，锚点 0 与 null 均是占位符，不得当成真实记录运行或默默填零。实际动作数组应完整包含有效组合。

缓存 key 至少包括 row UID、input IDs hash、model/tokenizer revision、operator hash、配置 hash、anchor token index、gamma、generation config 与 scorer version。新系数必须进入 key，禁止仅按 anchor 查旧缓存。

使用原子写入、可恢复分片、失败状态和完整性汇总。禁止旧 `gamma` 单值字段遮盖联合动作实际执行系数。

### 7.3 数据权限与测试隔离

训练、调参、网格选择、早停、阈值校准均不得使用 Formal 效用。由于旧 Formal 在之前工作中已被观察，新实验必须透明说明沿用基准，不应声称这些数据对研究者完全未知；可在资源允许时补充新任务／新测试队列，但不把它作为首版工程阻塞项。

正式推理冻结之前不生成 Formal 九动作 oracle 表。如果冻结后为分析生成 Formal 全动作缓存，放在单独的 `analysis_only` 路径；在线策略接口不得读取它。

## 8. 训练与验证协议

### 8.1 首版训练设置

- 每个 Backbone–任务单独训练，先 Qwen。
- AdamW，学习率 1e-4，weight decay 0.01。
- 最多 50 epochs，早停耐心 7。
- GELU，dropout 0.2。
- 默认 batch size：短任务／LITM 为 42，BABILong／RULER 为 32；显存不足可调整并如实记录。
- 第一个 smoke/正式单种子为 101；稳健性阶段建议训练种子 101、102、103，各自验证集校准。
- Backbone 全部冻结；模型依赖、随机性设置、checkpoint hash 写入记录。

这些是复用旧方案的起点，不是已验证的 SRAR_2 最优超参数。

### 8.2 标准化与样本采样

标量均值和标准差仅用训练集三候选标量计算，std floor=1e-6。不要因九动作广播而把数据大小错误计成九倍，也不要把候选输出统计加入标准化输入。保存并冻结 scaler。

按样本是否存在任一正优势联合动作划分正／非正组；两组都非空时，样本权重分别为 `1/(2*N_positive)` 和 `1/(2*N_nonpositive)`，有放回采样，每 epoch N 次。只有一组时使用定义明确的均匀采样，禁止除零。

原 SRAR 重跑默认按原三动作定义采样组。由于动作空间扩大可能改变采样分布，最终增益若较小，应补充统一采样策略的控制实验或至少披露差异，不能忽略这个混杂因素。

### 8.3 损失

\[
\mathcal L=\operatorname{SmoothL1}(\widehat{\mathbf A},\mathbf A)
+0.5\operatorname{BCEWithLogits}(\widehat{\mathbf A},\mathbb 1[\mathbf A>0])
+0.2\mathcal L_{\mathrm{RankNet}}.
\]

首版保留原权重。回归与 BCE 对有效动作取均值。排序比较有效且真实优势不相等的动作对；可沿用旧代码“每对在有效样本上平均，再对有效动作对平均”的定义，必须写清、固定，并给空集合返回可求导零值。

M=3 时九动作最多 36 个无序对，不能硬编码旧三个候选的三对。对重复等价动作屏蔽无意义比较。损失不随候选对数量直接求和放大。

同一个实数分数兼作回归输出和 BCE logit，不能声称它天然是校准的真实效用或概率；推理按原方案使用 raw score 与验证阈值。

### 8.4 checkpoint 与 tau

为最小改动，首版采用原 Qwen 顺序：

1. 每 epoch 用验证集、临时阈值 tau=0 计算完整联合策略的任务效用相对初始输出的增益，选择最高增益 checkpoint；平局规则固定，例如保留更早 epoch。
2. checkpoint 固定后，在验证集最大动作分数的所有不同值及全触发／全保留端点上校准严格 `max_score > tau`。
3. 选择验证集主指标最高者；平局优先较低触发率，再较大 tau。
4. RULER 使用 13 子任务宏平均主指标，不用全样本微平均替代。
5. 冻结 checkpoint、scaler、gamma grid、action order、tau、协议哈希，再执行 Formal。

实现阈值端点时处理 strict inequality：tau=max(score) 可全保留；全触发用有限的 nextafter(min_score,-inf) 或显式策略标志，不往严格 JSON 写非标准 Infinity。

不沿用旧 tau；所有需要门控的对照均按相同验证原则独立校准。记录 grid 与 checkpoint 搜索成本。

## 9. 在线推理实现要求

```python
def infer_srar2(x, model, selector, frozen):
    # 这一入口不接收 gold/reference/candidate utility。
    y0, features = initial_generate_and_capture(
        x, anchor=frozen.initial_anchor, gamma=frozen.gamma0
    )
    joint_scores = selector(features)  # [3, M]，均来自同一次 initial capture
    action, best_score = stable_argmax(joint_scores, frozen.action_order)
    if best_score <= frozen.tau:
        return y0, decision_record(triggered=False, passes=1)
    final = fresh_generate(
        x, anchor=action.token_index, gamma=action.gamma,
        layers=frozen.layers, heads=frozen.heads
    )
    return final, decision_record(triggered=True, passes=2)
```

这是建议接口，不是现成可运行脚本。实现中还要确保：

- 每次样本／动作前重置 operator 配置与 Observer 状态；样本 A 选过的 gamma 不得泄漏至样本 B 的初始计算。
- `fresh_generate` 不能复用不同干预下的失效 KV cache。
- 在线日志记录原始九个分数、选择的 anchor/gamma、tau、triggered、passes、输出及运行 hash；评分在生成后独立进行。
- 开启 `eval()` 与正确的无梯度模式。若按样本设置 operator 属性，首版使用明确的单样本推理或按同一动作分组，不能让同一 batch 的不同 gamma 错用一个全局属性。
- 不要把先生成九个候选、再查表选答案的离线评估冒充在线延迟结果。

## 10. 对照实验设计

### 10.1 核心比较

| ID | 方法 | 明确定义 | 主要问题 |
|---|---|---|---|
| B0 | Vanilla | 无干预，其他输入／解码一致 | 原模型表现 |
| B1 | Initial | 固定 `(c0,gamma0)` | 所有优势共同参照 |
| B2 | SRAR-ref | 三位置、单 gamma0、原 928 维结构，同协议重跑 | 相比原方法是否改善 |
| B3 | Best-fixed-gamma SRAR | 对每个固定 gamma_j 训练三位置评分器，均以同一 y0 为参照，验证集选最优配置与 tau | 是否只是原系数没调好 |
| B4 | Gamma-only | 固定一个区域位置，仅选择系数；固定区域由验证集确定，初始 y0 不变 | 系数自适应的单独作用 |
| B5 | Sequential | 用原位置评分器在 gamma0 下选位置，再用联合评分器评价该位置的系数，最后独立校准 tau | 是否需要联合位置选择 |
| J | SRAR_2 | 三位置 × M 系数联合评分，独立校准 tau | 完整方法 |

B3 不得对每个样本用参考答案选择 gamma，也不得在 Formal 上挑最好的固定配置。B4 固定区域的选择与系数网格冻结都只用验证集；如果另做初始位置系数自适应，需要新增相应标签，不能拿区域标签替代。

B5 是明确的分阶段策略，不是两个独立分类头直接拼接。记录其额外评分器开销。

### 10.2 控制与消融

- `J-BASE-mask`：同一个联合模型仅允许 BASE 系数动作，重新校准 tau。帮助区分新增动作与联合训练表示的影响，但不能代替独立重训的 B2。
- `J-Always`：联合模型始终执行最高分区域动作，取消最终门控。
- `J-ShuffledGamma`：若要检验系数信息，可在严格定义下做系数标签／输入对应关系扰动的诊断；不作为首版必需项，不用破坏标签一致性的方法训练主模型。
- 可选相同 944 维容量的三动作对照，判断少量新增参数是否解释收益。
- 候选池 oracle 是离线上界，不是可部署基线，不与真实方法混为同一成绩类别。

### 10.3 数据与计算公平性

所有比较固定相同模型、样本 manifest、token IDs、生成上限、评分器及初始参照。B2 与 J 的标签成本不同，要单独报告；不能说完全等训练成本。

B3 与 J 尽可能使用同一批九动作离线结果，让比较回答联合决策问题，而不是偷偷给一方更多标签。考虑统一采样、训练预算和模型容量的必要控制，不通过隐藏基线调参成本或新方法额外搜索制造优势。

旧 ACT/AutoPASTA 的历史表只能作为已披露协议下的背景。若正式纳入新主表，应在相同输入与评分协议下核验或重跑；不同辅助生成预算要披露。

## 11. 评价指标与诊断

### 11.1 主指标

- 每任务官方／已冻结评分，显示 0–100，内部保留未舍入值。
- RULER 13 子任务等权宏平均，同时提供每子任务结果。
- 六任务均值仅作为次级汇总，不掩盖退化任务。
- 三个训练种子的均值、标准差；Random 基线的随机种子不能替代 Selector 训练重复。
- 基于同一测试样本的配对差异；LITM bootstrap 按原始问题聚类，保留其排列；RULER 按子任务分层并保留宏平均权重。

### 11.2 Rescue/Harm 与一致性

二值任务报告：总样本数、触发数、Rescue、Harm、both-correct、both-wrong 的计数；分别给全测试集分母与触发子集分母的比例，不混用。

相同二值评分与配对样本下必须验证：

\[
\Delta\mathrm{Accuracy}
=\frac{N_{\mathrm{Rescue}}-N_{\mathrm{Harm}}}{N}
=\frac{N_{\mathrm{trigger}}}{N}
\left(\frac{N_{\mathrm{Rescue}}}{N_{\mathrm{trigger}}}
-\frac{N_{\mathrm{Harm}}}{N_{\mathrm{trigger}}}\right).
\]

零触发时比例记为不适用，整体变化为零。RULER 等分数型效用主分析用逐样本得分差与子任务宏平均闭合；若另报二值转移，独立定义其 correctness 判据，不能用它替代宏平均提升。

### 11.3 位置—系数交互与 oracle

- 统计每个位置在各系数下的得分／优势分布，而不只画平均注意力。
- 比较原动作集 oracle：`max(U0, U_RE_BASE, U_RM_BASE, U_RL_BASE)` 与扩展动作集 oracle：`max(U0, all U_i_j)`。
- 因扩展集包含原集，其逐样本 oracle 不应降低；若降低，先查缓存或数据对齐错误。
- oracle 上升不保证可学习选择器上升，必须分开报告“新增可改善空间”和“实际恢复收益”。
- 报告选择不同 gamma 的样本比例、各 anchor–gamma 操作的触发／收益／损害，识别是否退化成一个固定系数。
- 检查最佳系数是否随位置变化。只看到系数使用频率不同不足以证明耦合有价值，B3/B5 的性能比较更关键。
- 保留平局，不能随机指定一个所谓最佳 gamma 后夸大偏好差异。

### 11.4 效率

分别测量初始生成、特征提取附加开销、Selector 评分、触发后的生成、总端到端时间与峰值显存；记录 GPU、batch、token 长度、输出长度、warmup、同步计时方式。

在本方案中平均完整生成次数为 `1 + trigger_rate`，但不等于延迟倍率。九动作评分不是九次 Backbone 生成。离线标签生成 GPU 时间、token 数、缓存命中率单列。

禁止仅报告 MLP 参数量就声称整个方法“几乎零开销”。

## 12. 实施阶段与停止／继续依据

### 阶段 A：定位与协议审计

读取本文件及所在目录指令，建立源码、模型、配置、数据、checkpoint 的路径和哈希清单；检查本地／服务器资源。先不修改旧代码目录。

产物：`PROTOCOL_AUDIT.md`、`source_manifest.json`、`data_manifest_audit.json`、待解决事项列表。

### 阶段 B：CPU 实现与单元测试

实现联合 Router、动作注册表、标签 schema、损失、阈值选择、离线策略评估和日志。用合成数据测试，不需要先下载 Backbone 或启动大规模生成。

产物：可导入模块、CLI help、通过的单元测试、示例配置。

### 阶段 C：小规模真实 smoke 与潜力检查

- 工程 smoke 默认选 BoolQ：短生成、Qwen 原观察器较完整，先 8–16 条训练／验证样本。
- 研究预实验优先 LITM：建议先 100 条训练与 100 条验证记录，仅用非 Formal；要记录并处理同问题排列与数据分组。该规模仅作筛查，不作为正式显著性证据。
- 如 LITM 特征／输入 parity 未解决，继续 BoolQ 的完整链路，修复 LITM 适配，不把有问题的历史缓存直接投入实验。
- 先看新增系数是否产生不同输出及额外 oracle 收益，再做小规模 Selector 训练。
- 若没有潜在收益，检查实现是否正确、系数是否真正应用、评分是否有效；确认后如实报告该网格无明显空间，可在非 Formal 范围内开展有记录的有限探索。不要无上限搜索直至找到漂亮结果。

产物：smoke parity、动作覆盖、oracle 对比、实际预算估算。阶段 C 不应产生“新方法已优于原 SRAR”的最终结论。

### 阶段 D：单任务完整训练／验证

在已通过审计的一个任务上完成九动作 train/validation 数据、J 与 B2/B3/B5、tau 冻结。先一个训练种子打通；有合理潜力后扩展三个种子与 B4。

若 J 不优于 B3，应优先分析它是否只是全局系数重调带来的收益，而不是继续堆叠无关模块。

### 阶段 E：冻结 Formal 推理与扩展任务

冻结全部设置后评估同一 Formal 队列。先完成一个任务，再扩展其他 Qwen 任务。所有正负结果保留，不只选择提升任务汇报。

Llama 属于后续跨 Backbone 验证，不是首版实现前置条件。它的算子在 softmax 后缩放锚点概率并归一化，系数 `r` 与 Qwen `gamma` 不同；需要独立候选网格、特征 schema 与运行协议，不能复制 Qwen 的数值或样本规模。

## 13. 建议工程组织与 CLI

新代码优先独立放在 `D:/Codex_Project/srar2/`；服务器可放在现有项目中的 `srar2/`。使用 `--project-root` 和配置路径传入实际位置，避免写死 Windows/Linux 绝对路径。

```text
srar2/
  README.md
  configs/
    qwen_boolq.yaml
    qwen_litm.yaml
  src/srar2/
    actions.py
    operator_adapter.py
    features.py
    model.py
    losses.py
    records.py
    metrics.py
    calibration.py
    inference.py
    cli.py
  tests/
  reports/
```

建议统一 CLI 子命令：`audit`、`smoke`、`generate-counterfactuals`、`extract-features`、`build-labels`、`train`、`calibrate`、`freeze`、`infer`、`evaluate`、`analyze`。

所有昂贵命令支持 `--dry-run`、`--max-samples`、`--shard-index`、`--num-shards`、`--resume`。输出应显示计划生成次数、已验证缓存命中数及尚缺动作，避免一次误操作启动所有数据集。

配置至少包含：

```yaml
experiment: srar2_qwen_litm_v1
backbone_id: Qwen/Qwen2-7B
dtype: bfloat16
attention_backend: sdpa
operator: K_AFTER_REPEAT_KV_NATIVE_SDPA_SEMANTICS
initial_anchor: 0
gamma0: 0.95
gamma_values: [0.95, 0.76, 1.14]
coefficient_ids: [BASE, LOW, HIGH]
anchors: [RE, RM, RL]
observed_layers: [14, 27]
hidden_size: 3584
coefficient_embedding_dim: 16
seed: 101
max_epochs: 50
patience: 7
learning_rate: 0.0001
weight_decay: 0.01
batch_size: 42
dropout: 0.2
loss_weights: {regression: 1.0, positive: 0.5, ranking: 0.2}
strict_gate: true
```

这不是完整可运行配置。模型路径、版本、heads、干预层、tokenization、manifest、scorer、输出上限与哈希必须通过 audit 补齐，缺失就明确报错，不能自行使用不匹配默认值。

## 14. 验收测试

### 14.1 必需 CPU 测试

1. 网络输入、输出与梯度：`[B,10,3584]`、`[B,3,16]` 输入得到 `[B,3,M]`；各投影、gamma embedding 有正确梯度。
2. 每种 tensor flatten/unflatten 与 action ID 映射往返一致，tie order 与配置一致。
3. 所有 gamma 值变更、顺序变更、schema 变更能被 checkpoint/cache hash 校验发现。
4. M=1 与兼容权重下，在 eval 模式恢复旧模型输出和原三个候选的 tie 行为。
5. 标签优势差、掩码、重复动作、全平局、全无正样本、缺失动作处理正确。
6. 排序损失在无有效对时返回零且不产生 NaN；在正确／错误排序的合成例子上方向符合预期。
7. tau 等号不触发，含相同最大分数时阈值前沿正确，全触发／全保留端点正确。
8. 验证集校准不得读取 Formal；在线推理 API 不接收 reference 或候选真实效用。
9. 二值 Rescue/Harm 闭合，RULER 宏平均与微平均在构造样例上能区分。
10. LITM 同问题多个 row 保留，禁止以 question ID 覆盖行；训练／验证／测试的分组检查正确。
11. 新缓存中 gamma、operator、input hash 改变时不命中旧输出。

### 14.2 必需真实推理 smoke

1. gamma0 的原三个候选与选定原实现／重跑参照在同输入协议下逐样本对齐。
2. 初始输出在单 gamma 与多 gamma 配置中一致；初始特征不依赖候选 gamma grid。
3. 实际只修改指定 token、层、query heads 的 expanded K，pre-repeat cache 不被污染。
4. 连续两个不同动作／样本之间无 gamma 或 Observer 状态泄漏。
5. forced retain 只执行初始生成；forced reroute 仅增加一次生成；新候选生成独立构建有效 cache。
6. 在线执行得到的选定动作输出与同协议离线缓存一致；若因非确定性不一致，记录原因并重新制定可验证的 parity 标准，不能放宽到只看平均分。
7. 注意力 Observer 使用真实 gamma0，而不是历史常量；需要修复时建立新版本并重新评估对应基线。

## 15. 交付物与结论边界

实现会话应交付：

- 独立新代码、配置、依赖版本、可复现入口与必要测试。
- 协议审计报告，说明选用哪份旧源码、哪些结果可复用、哪些必须重新生成。
- 完整的 train/validation 动作标签与特征索引、运行预算和失败记录。
- checkpoint、scaler、gamma grid、action schema、tau 及其冻结哈希。
- 同协议对照结果、配对分析、oracle／交互分析、效率记录；未运行的项目明确标为未完成。
- 简明结论：新增操作是否提供潜力、Selector 是否学会利用、是否超过最佳固定系数、代价是什么。

如果正式运行所需模型、数据、服务器权限或资源缺失，交付已完成代码、CPU 测试、dry-run 清单及具体缺失项，不宣称真实实验通过。

不预设以下结论：SRAR_2 一定提升；gamma 越大越好；多动作一定更准确；局部注意力恒等式保证任务收益；九动作评分意味着九次在线生成；修改网络后足以独立投稿。

## 16. 相关研究与投稿衔接

这是一项后续研究设计，原 SRAR 尚在审。新稿须按最终目标会议／期刊要求说明与相关在审工作的关系和实质差异，不能把同一贡献轻微改写后默认可同时投稿。

供接手会话继续检索的资料：

- ZeroTuning：<https://arxiv.org/abs/2505.11739>。不同论文版本／官方算子可能变化，应区分本项目冻结实现与最新官方实现，不静默替换实验协议。
- When is Your LLM Steerable?：<https://arxiv.org/abs/2606.11599>。涉及利用早期模型状态预测干预效果并辅助系数搜索；需要进一步阅读全文比较，而不是声称“首次预测系数”。
- IEEE 对相关已发表／在审工作披露与投稿的说明：<https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/>。具体规则以目标渠道为准。

拟验证的贡献应聚焦于：**固定初始计算下的位置—系数联合效用预测、对两者交互的建模，以及有限生成次数下对固定系数和分阶段选择的实证优势。** 是否具有足够新颖性与价值，要根据文献比较和真实结果判断。

---

## 接手会话的第一轮行动清单

1. 读本文件、工作区指令和代码包 RELEASE_SCOPE，确认是实现 SRAR_2，不是修改旧稿。
2. 确认模型／数据／服务器实际路径，列出选用源码和配置；不要扫描或输出无关凭据。
3. 建立独立 `srar2/` 目录，先实现动作 schema、944 维联合评分器、可变动作损失、校准与测试。
4. 做 M=1／旧权重兼容和合成策略测试，再用少量 BoolQ 样本真实 smoke。
5. 在非 Formal 数据上检查新系数的潜在收益与成本；通过后完成一个任务的受控比较。
6. 根据结果推进其他任务与种子，并持续报告事实、限制和剩余工作。不要为了符合预期而修改结果、评分或测试样本。
