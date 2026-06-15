#!/usr/bin/env bash
# 快速实验编排：3 段对话 × 5 组系统，生成 + Judge。
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
cd "D:/Desktop/3-2/NLP/Project"

PY=./.venv/Scripts/python.exe
EVAL=memory_agent/data/eval_set.json
PRED=memory_agent/experiments/predictions
RES=memory_agent/experiments/results
LIM=3
mkdir -p "$PRED" "$RES"

declare -A AGENTS=(
  [B0]=memory_agent.agent.ta_adapter:NoMemoryAgent
  [B1]=memory_agent.agent.ta_adapter:FullContextAgent
  [B2]=memory_agent.agent.ta_adapter:VanillaRAGAgent
  [S1]=memory_agent.agent.ta_adapter:AppendOnlyMemoryAgent
  [S2]=memory_agent.agent.ta_adapter:ConflictAwareMemoryAgent
)

ORDER="B0 B1 B2 S1 S2"

echo "########## 生成阶段 ##########"
for k in $ORDER; do
  echo ">>>>> [$k] 生成开始 $(date '+%H:%M:%S')"
  $PY -B -m memory_agent.eval.run_ta_generation \
      --eval_set "$EVAL" --agent "${AGENTS[$k]}" \
      --output "$PRED/$k.json" --limit_conversations $LIM
  echo "<<<<< [$k] 生成结束 $(date '+%H:%M:%S') rc=$?"
done

echo "########## Judge 阶段 ##########"
for k in $ORDER; do
  echo ">>>>> [$k] Judge 开始 $(date '+%H:%M:%S')"
  $PY -B -m memory_agent.eval.run_ta_judge \
      --predictions "$PRED/$k.json" \
      --output "$RES/$k.json" --num_workers 4
  echo "<<<<< [$k] Judge 结束 $(date '+%H:%M:%S') rc=$?"
done

echo "########## 全部完成 $(date '+%H:%M:%S') ##########"
