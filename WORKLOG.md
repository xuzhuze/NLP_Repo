# 长期记忆 Agent 接手记录

## 2026-06-01 Codex 接手

### 已确认

- Claude 已完成初版目录骨架、三组 baseline、S1/S2 主系统和占位评测入口。
- 当前机器没有配置 `memory_agent/.env`，也没有 `LLM_*`、`EMBED_*`、`JUDGE_*`
  环境变量，因此不能调用真实云端 API。
- 当前 Python 环境没有安装 `faiss-cpu`。
- 作业说明中的助教工具包链接
  `https://cdn.jsdelivr.net/gh/zifengcheng/NJU_NLP_Project_2026@main/agent_momery/eval_kit.zip`
  在 2026-06-01 返回 `404`。
- 用户补充了正确仓库链接。真实目录是 `长期记忆对话 Agent/`，旧链接中的
  `agent_momery/` 是错误路径。

### 已完成

- 从官方仓库下载 `memory_agent/data/locomo_official/locomo10.json`，用于真实格式适配。
- 数据加载器同时兼容助教简化版与官方 LoCoMo 单文件格式。
- 官方数据已验证可解析：10 个对话、1986 个问题。
- Store 在安装 `faiss-cpu` 时使用 FAISS；未安装时自动使用 NumPy 后备索引。
- Writer 保存 `source_turn_ids`；Controller 额外落盘规范化后的 `raw_dialogues.json`。
- `conflict_aware` Updater 会检查阈值以上的多个邻居，不再只看 top-1。
- 评测入口支持离线 `token_f1`、`exact_match` 和可选 `judge` 三种模式。
- 新增离线测试，覆盖数据加载、向量检索、来源追踪、冲突更新和汇总评测。
- CLI 级测试发现并修复 Writer prompt 中 JSON 示例花括号未转义导致的
  `KeyError: '"text"'`。
- 依赖拆分为基础、可选 FAISS 和开发测试三组，Windows 无 FAISS 时仍可运行。
- 已从正确仓库路径下载并解压助教正式 `eval_kit`。
- 已生成正式 `memory_agent/data/eval_set.json`：10 段对话、160 题，四类各 40 题。
- 新增 `agent/ta_adapter.py`，B0/B1/B2/S1/S2 均实现正式 `ingest()` / `answer()` 接口。
- 新增正式数据准备、生成和 Judge 包装入口。Judge 包装入口会将 `.env` 中的
  `JUDGE_*` 映射为助教脚本读取的 `LLM_*`。
- 助教原始 `prepare_eval_set.py` 在 Windows GBK 默认编码下会因 emoji 写入失败；
  本项目包装入口显式使用 UTF-8 写出，并在 vendored 助教脚本副本中统一补充 UTF-8
  读写与输出目录创建，规避该问题，不改变评分逻辑。

### 验证结果

- `python -B -m pytest memory_agent/tests -q -p no:cacheprovider`：`15 passed`
- `python -B -m ruff check --no-cache memory_agent`：通过
- 官方数据规范化统计：10 个对话、272 个 session、5882 个 turn、1986 个问题
- CLI 离线流程：B0/B1/B2/S2 均成功生成 trace；S2 成功执行 `ADD -> UPDATE`
- 助教正式 `run_generation.py` 已通过固定假 Agent 完成端到端 CLI 测试：动态导入、
  `ingest()`、`answer()`、嵌套输出目录创建和 `predictions.json` 写出均正常。
- vendored 助教 `prepare_eval_set.py` 已在 Windows 下直接复测：UTF-8 写出正常。

### 待外部输入

- 在 `memory_agent/.env` 填入真实 API 配置后，按 README 用
  `--limit_conversations 2` 跑 B0/B1/B2/S1/S2。
- 小样本跑通后再执行完整 160 题实验和助教 Judge，生成结果表。
