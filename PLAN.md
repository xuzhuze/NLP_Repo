# 长期记忆对话 Agent —— 项目方案

> 2026-06-01 起草。基于原 proposal 微调。全程使用云端 API，不本地部署模型。
> 距 6/3 Milestone 2 天，6/17 Final 16 天。

---

## 一、对 Proposal 的微调

| 项目 | Proposal 原方案 | 调整后 | 理由 |
|------|----------------|--------|------|
| 探索方向 | 仅方向 C（冲突更新） | 仍以 C 为主；时间允许再加方向 A 的三因子打分 | 作业明确"2 个有加分"；A 仅在 Retriever 加打分函数，复用性高 |
| 生成模型 | 云端 Qwen2.5-7B | **全程云端 API**（SiliconFlow/DashScope 的 Qwen 系列） | 用户决定：不折腾本地 vLLM |
| Embedding | bge-small-zh-v1.5 | **bge-m3（云端 API 调用）** | LoCoMo 是英文数据，zh 模型会失败；bge-m3 多语言 |
| Judge | DeepSeek | DeepSeek（保持异源） | 规避 self-evaluation bias |

---

## 二、系统架构

```
                      ┌────────────────────────────────────┐
                      │       Agent Controller              │
                      │  (主流程编排 + 全链路日志)            │
                      └──────────────┬─────────────────────┘
              session 结束 │         │ 新 query
                    ▼               ▼
        ┌──────────────────┐   ┌──────────────────┐
        │  Memory Writer   │   │ Memory Retriever │
        │ LLM 提取记忆单元  │   │ dense top-k      │
        │ {fact, ts, sid}  │   │ (可选三因子打分)  │
        └────────┬─────────┘   └─────────┬────────┘
                 │                       │
                 ▼                       │
        ┌──────────────────┐             │
        │ Memory Updater   │             │
        │ append_only /    │             │
        │ conflict_aware   │             │
        └────────┬─────────┘             │
                 ▼                       ▼
              ┌────────────────────────────┐
              │  Memory Store (FAISS+JSON) │
              │  {id, text, ts, sid, ver,  │
              │   status, supersedes}      │
              └────────────────────────────┘
```

**关键原则**：
- 严格区分 raw_dialogue（JSONL 落盘，不入库）与 memory_unit（派生记忆，入库）
- 每条记忆：`{id, text, source_session_id, source_turn_ids, ts, version, status, supersedes}`
- 每次 QA 全链路落盘 trace：`{query, retrieved, prompt, response, judge}`

---

## 三、目录结构

```
memory_agent/
├── configs/
│   ├── prompts.yaml          # writer / conflict / qa 三类 prompt
│   └── model.yaml            # LLM / embedding / top_k 等
├── memory/
│   ├── store.py              # FAISS + metadata JSON 双写
│   ├── writer.py             # 会话结束批量提取
│   ├── retriever.py          # dense（默认）/ three_factor（可选）
│   └── updater.py            # append_only / conflict_aware
├── agent/
│   ├── controller.py         # 编排
│   ├── llm_client.py         # OpenAI 兼容客户端，chat+embed
│   ├── ta_adapter.py         # 助教 ingest()/answer() 正式接口
│   └── tracer.py             # 全链路日志
├── baselines/
│   ├── no_memory.py
│   ├── full_context.py
│   └── vanilla_rag.py        # 原始切片 RAG
├── eval/
│   ├── prepare_ta_eval_set.py # 生成正式分层抽样集
│   ├── run_ta_generation.py  # 调用助教生成脚本
│   ├── run_ta_judge.py       # 调用助教 Judge
│   └── run_eval.py           # 离线快速汇总
├── experiments/              # predictions / results / runs
├── data/                     # LoCoMo / eval_set / traces
├── scripts/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 四、实验矩阵

| Exp | 系统 | Retriever | Updater | 用途 |
|-----|------|-----------|---------|------|
| B0 | no_memory | — | — | baseline |
| B1 | full_context | — | — | baseline |
| B2 | vanilla_rag | dense on raw turns | — | baseline |
| **S1** | 本系统-A | dense | append_only | 我方基线 |
| **S2** | 本系统-B | dense | **conflict_aware** | **主探索（方向 C）** |
| S3（加分） | 本系统-C | three_factor | conflict_aware | 方向 A 消融 |

**汇报**：按助教正式 `eval_kit` 的 4 类题型（`single_hop` / `temporal` /
`multi_hop` / `open_domain`）分别给 Judge Score、F1、EM 和总分；附 LLM 调用次数
+ 端到端延迟。

**关键观察点**：正式抽样集没有单独的“信息更新追踪”类别。S2 相比 S1 的收益应结合
分类指标与含冲突事实的 bad case 单独分析，不能只靠类别均值推断。

---

## 五、时间节点（贴合今日 2026-06-01）

| 时间 | 任务 |
|------|------|
| 6/1（今天） | 目录骨架 + LLM client + Store + Vanilla RAG baseline |
| 6/2 | 跑通 B0/B1/B2 + S1（append_only）小样本；准备 milestone 提交 |
| 6/3 Milestone | 提交 baseline 全集初步结果 |
| 6/4–6/9 | 实现 Updater conflict_aware；跑 S2 全集；按题型拆解 |
| 6/10–6/13 | 加分项 S3；bad case 分析；成本/延迟指标 |
| 6/14–6/17 | ACL 论文 + PPT + 代码整理 + 答辩 |

---

## 六、最小工作量原则

- 默认只做 B0/B1/B2/S1/S2 五组实验，S3 时间够再做
- 不做本地 vLLM、不做本地 GPU 部署
- 不做 Reflection、不做遗忘曲线、不做多种检索器并存
- prompt 集中在 `configs/prompts.yaml`，方便换不改代码
