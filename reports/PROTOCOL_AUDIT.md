# SRAR_2 协议审计

日期：2026-10-30。新工程根目录 `D:/Codex_Project/srar2`；原目录只读引用。

## 选定来源

以工作区 `stage26q2_short_base_srar.py` 的 Router、短任务 Observer、特征、生成及训练定义为参照；算子选 `srar_paper_export/scripts/qwen2_zerotuning_sdpa_native_semantics_v2.py`，内容原样复制。参数从分享包最终 frozen_config 提取，不使用旧绝对路径或历史 seal 跳过核验。完整来源清单、hash、文件字节数见 `source_manifest.json`。README/RELEASE_SCOPE/训练、主实验及分析链文档已阅读。

原 928 维结构来自 7×128 向量块、16 标量、16 区域 embedding。新网络增加16维系数 embedding；原参数 1,631,025，新参数 1,635,169，新增4,144。默认从头训练；兼容旧权重仅做测试，不参与默认新方法训练。

## 已发现问题

1. 选定 LITM feature 源码 `capture_probs` 确实硬编码 `*=.9`；冻结任务 gamma 为0.95。现有 LITM 缓存无法仅凭文件名确认符合新协议。通用新 Observer 使用实际 gamma0，LITM 任务适配与重算 parity 未完成，配置标为 audit-only。
2. 短任务使用 `add_special_tokens=False`，LITM 历史使用 True。不能把不同输入假设合并；新 manifest 保存实际 token ID hash，短任务导入器核验长度、原坐标与分组。
3. 分享包是裁剪派生版；不能据文件齐全或 README 的 RECOVERED 标签宣称正式成绩复现。旧包没有找到完整短任务原 train/dev manifest 与模型权重。未复用任何旧预测、特征或 tau。
4. 历史短任务冻结配置包含 Instruct 参照痕迹，但实际 Stage26 主链是 Base；新入口禁止路径或模型来源名称带 Instruct，并校验 Qwen2-7B 结构。架构和目录名本身不能完全证明权重是 Base：必须使用用户已知的 Base 权重，记录全部分片内容 hash，不能靠改目录名冒充。
5. 本机：Windows、Python3.9.7、torch2.8.0+cpu、numpy1.20.0；CUDA不可用；Transformers/datasets未安装。历史服务器环境为 torch2.13.0/transformers4.53.2。GPU迁移固定 Transformers 并重新检查 token parity，不能把旧 GPU 环境的数值直接当新结果。

## 数据与实验状态

原数据的已知服务器相对位置：

- `runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_train_effective.jsonl`
- `runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_validation.jsonl`
- `runs/stage26q2_short_prepare/manifests_qwen2_base/boolq_formal.jsonl`
- LITM train：`data/qwen2_derived_long_trains/LITM_Qwen2-7b/train.jsonl`
- LITM validation：`results/stage24_long_rawprompt/manifests/litm_validation.jsonl`
- LITM prepared：`runs/stage26q2_litm_srar/input_view/` 下旧 train/dev 文件。

上述来自旧脚本和registry，是路径线索；未连接服务器核验。本轮不重新下载或重划数据。新导入器保留原 sample_id 与顺序，并新增 source row index/文件hash/tokenhash；prepared短任务缺少原 question 字段时从既有 token span 恢复分组。发现重叠即停止，不静默丢行。

阶段A：源码/配置定位完成，模型/数据现场核验受资产缺失限制。阶段B：独立模块和CPU验证。阶段C：已提供2条 parity与8+8条BoolQ全链路脚本，但当前**未执行真实Qwen实验**。阶段D/E均未开展。不存在真实 oracle、模型收益、Selector学习或显著性结论。

## 可复用边界

允许复用：原数据成员/划分、raw prompt和坐标定义、冻结层和heads、gamma0、解码上限、官方评分函数逻辑。

本轮实际复用的旧答案/特征数量为零。新 cache key 含row UID、input IDs、权重/tokenizer内容hash、operatorhash、协议、特征schema、anchor、gamma、生成设置和scorer版本。新增LOW/HIGH必须真实生成。新特征只采固定initial计算，在线评分后至多追加一次独立候选生成，不复用不同干预的失效KV cache。

正式评估须先完成真实 parity、动作覆盖、非Formal oracle、受控训练与校准。校准接口拒绝 Formal；pre-experiment counterfactual入口拒绝Formal。所有正负输出保留，不能把合成测试或dry-run写成真实结果。
