# 交付状态

2026-10-30。新目录：`D:/Codex_Project/srar2`。

- 已实现：独立核心模块、短任务运行入口、旧数据导入、标签缓存、训练/校准、J/B2/B3/B4/B5验证分析、冻结与在线推理入口。
- CPU验证：35项测试通过；包含真实CPU SDPA张量运算与合成端到端训练。具体输出见 `pytest.txt`。
- 真实模型实验：未完成。当前CPU环境缺少CUDA、Transformers、Qwen2-7B权重和原train/dev数据文件；具体项见 `environment_audit.json`。没有真实准确率或oracle结果。
- 后续第一步：GPU机器clone仓库，按README安装环境，将 `--legacy-root` 指向保留原数据划分的旧SRAR项目，运行 `tools/run_boolq_smoke.py`。
- 首版运行任务：BoolQ/SST-2/LogiQA。LITM仍需任务适配与修复后parity；BABILong/RULER未实现运行适配。Formal/三种子研究与长任务统计分析未开展。
- 原代码和结果：未写入旧SRAR目录；来源清单内原文件hash在交付时再次核验。新权重、cache和结果只写新工程及测试临时目录。

`--dry-run`与合成测试不构成真实实验；不声称SRAR_2优于SRAR或最佳固定系数。
