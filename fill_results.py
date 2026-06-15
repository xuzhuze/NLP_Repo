"""读取 results/{B0..S2}.json，把真实数字写进 presentation.html。
所有数字来自异源 LLM-as-Judge 评测结果，不做任何编造。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "memory_agent/experiments/results"
HTML = ROOT / "presentation.html"
SYS = ["B0", "B1", "B2", "S1", "S2"]
CATS = ["single_hop", "temporal", "multi_hop", "open_domain"]
CAT_SHORT = {"single_hop": "single", "temporal": "temporal", "multi_hop": "multi", "open_domain": "open"}


def pct(x):
    return f"{x * 100:.1f}"


def load(sys):
    p = RES / f"{sys}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build():
    R = {}
    for s in SYS:
        d = load(s)
        if not d:
            print(f"[warn] missing {s}.json")
            continue
        ov = d["overall"]
        bc = d.get("by_category", {})
        R[s] = {
            "judge": pct(ov["score"]),
            "f1": pct(ov["f1"]),
            "em": pct(ov["em"]),
            "lat": f"{ov.get('avg_latency_sec', 0):.2f}",
            "calls": "—",
        }
        for c in CATS:
            R[s][CAT_SHORT[c]] = pct(bc[c]["score"]) if c in bc else "—"
    return R


def replace_table(html, R):
    # 用 data-r="SYS.field" 定位单元格，替换其文本
    def repl(m):
        attr = m.group(1)
        sys, field = attr.split(".")
        val = R.get(sys, {}).get(field)
        if val is None:
            return m.group(0)
        return f'data-r="{attr}"{m.group(2)}>{val}<'
    # 匹配  data-r="X.y" ...>OLD<
    return re.sub(r'data-r="([^"]+)"([^>]*)>[^<]*<', repl, html)


def main():
    R = build()
    if not R:
        print("没有结果文件，先跑实验。")
        return
    html = HTML.read_text(encoding="utf-8")
    html = replace_table(html, R)
    # 状态提示
    n_done = len(R)
    status = "（真实评测数据 · 异源 deepseek-v3 Judge）" if n_done == 5 else f"（已填入 {n_done}/5 组）"
    html = re.sub(r'(<span class="red" data-fill="RESULT_STATUS">)[^<]*(</span>)', rf'\g<1>{status}\g<2>', html)
    HTML.write_text(html, encoding="utf-8")
    # 控制台打印一份对照表
    print(f"已填入 {n_done} 组系统。总体 Judge 得分：")
    for s in SYS:
        if s in R:
            print(f"  {s}: judge={R[s]['judge']}  f1={R[s]['f1']}  em={R[s]['em']}  lat={R[s]['lat']}s")
    if "S1" in R and "S2" in R:
        delta = float(R["S2"]["judge"]) - float(R["S1"]["judge"])
        print(f"  >>> S2 - S1 消融净收益: {delta:+.1f} pp")


if __name__ == "__main__":
    main()
