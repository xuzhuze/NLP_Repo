# Long-Term Memory Agent

NJU NLP 大作业：支持长期记忆的对话 Agent。生成、Embedding 和 Judge 均通过
OpenAI 兼容 API 调用，无需本地 GPU。

## 当前能力

- B0 `NoMemoryAgent`、B1 `FullContextAgent`、B2 `VanillaRAGAgent`
- S1 `AppendOnlyMemoryAgent` 与 S2 `ConflictAwareMemoryAgent`
- 已接入助教正式 `eval_kit` 的 `ingest()` / `answer()` 接口
- 原始对话与派生记忆分开落盘，逐题保存完整 trace
- 安装 `faiss-cpu` 时使用 FAISS；未安装时自动使用 NumPy 后备索引
- 离线 token-F1 评分与助教权威 LLM-as-Judge 评分

## 安装依赖

```powershell
python -m pip install -r memory_agent/requirements.txt
```

兼容平台建议安装 FAISS：

```powershell
python -m pip install -r memory_agent/requirements-faiss.txt
```

如果 Windows 无法安装 `faiss-cpu`，程序会自动使用 NumPy 后备索引。跑测试还需要：

```powershell
python -m pip install -r memory_agent/requirements-dev.txt
```

## 离线验证

在大作业根目录执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m pytest memory_agent/tests -q -p no:cacheprovider
python -B -m ruff check --no-cache memory_agent
```

离线验证不需要 API key。

## 配置 API

```powershell
Copy-Item memory_agent/.env.example memory_agent/.env
```

编辑 `memory_agent/.env`，填入 `LLM`、`EMBED`、`JUDGE` 三组 OpenAI 兼容配置。
`.env` 已加入忽略列表，不要提交。

## 准备正式评测集

助教工具包已解压到 `memory_agent/eval/ta/eval_kit/`。官方 LoCoMo 数据位于：

```text
memory_agent/data/locomo_official/locomo10.json
```

生成助教格式的分层抽样集：

```powershell
python -m memory_agent.eval.prepare_ta_eval_set
```

默认输出 `memory_agent/data/eval_set.json`：10 段对话、160 题，`single_hop`、
`temporal`、`multi_hop`、`open_domain` 四类各 40 题。

## 正式生成

先用前 2 段对话验证 API：

```powershell
$EVAL = 'memory_agent/data/eval_set.json'
$PRED = 'memory_agent/experiments/predictions'

python -m memory_agent.eval.run_ta_generation --eval_set $EVAL --agent memory_agent.agent.ta_adapter:NoMemoryAgent            --output "$PRED/B0.json" --limit_conversations 2
python -m memory_agent.eval.run_ta_generation --eval_set $EVAL --agent memory_agent.agent.ta_adapter:FullContextAgent         --output "$PRED/B1.json" --limit_conversations 2
python -m memory_agent.eval.run_ta_generation --eval_set $EVAL --agent memory_agent.agent.ta_adapter:VanillaRAGAgent           --output "$PRED/B2.json" --limit_conversations 2
python -m memory_agent.eval.run_ta_generation --eval_set $EVAL --agent memory_agent.agent.ta_adapter:AppendOnlyMemoryAgent     --output "$PRED/S1.json" --limit_conversations 2
python -m memory_agent.eval.run_ta_generation --eval_set $EVAL --agent memory_agent.agent.ta_adapter:ConflictAwareMemoryAgent  --output "$PRED/S2.json" --limit_conversations 2
```

小样本跑通后删除 `--limit_conversations 2` 跑完整抽样集。中断后可加 `--resume`。

## 正式评测

助教 Judge 是最终权威指标。包装入口会自动从 `memory_agent/.env` 读取 `JUDGE_*`：

```powershell
python -m memory_agent.eval.run_ta_judge --predictions memory_agent/experiments/predictions/S2.json --output memory_agent/experiments/results/S2.json --num_workers 4
```

项目自带的离线汇总只用于快速迭代：

```powershell
python -m memory_agent.eval.run_eval --runs memory_agent/experiments/runs --scorer token_f1
```

## 旧版入口

`memory_agent.baselines.*` 与 `memory_agent.agent.controller` 仍保留，用于离线 smoke test
和兼容旧数据格式。正式实验统一使用助教 `eval_kit` 入口。
