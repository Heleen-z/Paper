"""Rebuild thesis pilot figures from per-run evidence. Run from any directory."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pypdfium2
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/pilot"
WIDTH, HEIGHT = 760, 416
INK, MUTED, GRID = "#18313D", "#4F6470", "#DCE5E7"
BLUE, TEAL, GOLD = "#3267A8", "#007A70", "#A14F25"
SERIES = {"random": ("随机", BLUE), "entropy": ("熵选样", TEAL)}
FAMILIES = {"create": "任务创建", "budget": "预算调整", "feedback": "标注反馈",
            "pause": "暂停与恢复", "retry": "失败重试", "rollback": "版本回滚"}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Figure:
    """Use top-left coordinates for layout; the vector PDF also supplies the PNG."""

    def __init__(self, name, title, subtitle):
        self.path = OUT / (name + ".pdf")
        self.pdf = Canvas(str(self.path), pagesize=(WIDTH, HEIGHT), invariant=1)
        self.pdf.setTitle(title)
        self.box(0, 0, WIDTH, HEIGHT, "#FFFFFF")
        self.box(30, 24, 5, 25, TEAL)
        self.text(47, 43, title, 23, bold=True)
        self.text(47, 66, subtitle, 11, MUTED)
        self.line(30, 82, 730, 82, GRID)

    def text(self, x, y, value, size=11, color=INK, bold=False, align="left"):
        self.pdf.setFillColor(HexColor(color))
        self.pdf.setFont("cn-bold" if bold else "cn", size)
        draw = {"left": self.pdf.drawString, "center": self.pdf.drawCentredString,
                "right": self.pdf.drawRightString}[align]
        draw(x, HEIGHT-y, str(value))

    def line(self, x1, y1, x2, y2, color=GRID, weight=1, dash=None):
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setLineWidth(weight)
        self.pdf.setDash(dash or [])
        self.pdf.line(x1, HEIGHT-y1, x2, HEIGHT-y2)
        self.pdf.setDash([])

    def box(self, x, y, w, h, color):
        self.pdf.setFillColor(HexColor(color))
        self.pdf.rect(x, HEIGHT-y-h, w, h, fill=1, stroke=0)

    def point(self, x, y, color, square=False, radius=3):
        self.pdf.setFillColor(HexColor(color))
        if square:
            self.pdf.rect(x-radius, HEIGHT-y-radius, radius*2, radius*2, fill=1, stroke=0)
        else:
            self.pdf.circle(x, HEIGHT-y, radius, fill=1, stroke=0)

    def finish(self, footer):
        self.line(30, 379, 730, 379)
        self.text(30, 399, footer, 10.5, MUTED)
        self.pdf.save()
        with pypdfium2.PdfDocument(self.path) as doc:
            page = doc[0]
            bitmap = page.render(scale=3.5)
            bitmap.to_pil().save(self.path.with_suffix(".png"), dpi=(252, 252))
            bitmap.close()
            page.close()


def nice_axis(low, high):
    ideal = max(high-low, 0.8)/4
    unit = 10**math.floor(math.log10(ideal))
    step = min([n*unit for n in (1, 2, 2.5, 5, 10) if n*unit >= ideal-1e-9])
    minimum = math.floor(low/step)*step
    maximum = math.ceil(high/step)*step
    if maximum == minimum:
        maximum += step
    return minimum, maximum, step


def model():
    rows = read_rows(ROOT / "results/model_pilot/metrics.csv")
    series = defaultdict(dict)
    budgets = {}
    labels = {}
    for row in rows:
        method = row["method"].lower()
        if method not in SERIES:
            continue
        step, seed = int(row["round"]), int(row["seed"])
        key = (method, seed)
        if step in series[key]:
            raise ValueError(f"Duplicate run {key}, round {step}")
        value, budget = float(row["top1"]), float(row["budget_fraction"])
        assert 0 <= value <= 1 and 0 <= budget <= 1
        series[key][step] = 100*value
        if step in budgets:
            assert math.isclose(budgets[step], 100*budget)
        budgets[step] = 100*budget
        labels[step] = int(row["unique_labels"])
    seeds = sorted({seed for _, seed in series})
    steps = sorted(budgets)
    assert seeds and steps[0] == 0 and steps[-1] > 0
    assert all(sorted(series[(method, seed)]) == steps for method in SERIES for seed in seeds)
    final = steps[-1]
    fig = Figure("model-budget", "标注预算与目标域识别准确率",
                 f"Office-Home  ·  Product → Real-World      {len(seeds)} 个配对种子      {len(steps)-1} 轮查询")
    fig.text(32, 108, "目标域 Top-1 准确率（%）", 14, bold=True)
    fig.text(497, 108, "末轮配对差（百分点）", 14, bold=True)
    fig.text(497, 130, "熵选样减去随机选样", 10.5, MUTED)
    left, right, top, bottom = 76, 443, 165, 305
    all_values = [val for run in series.values() for val in run.values()]
    ylo, yhi, delta = nice_axis(min(all_values)-0.2, max(all_values)+0.2)
    def y(v):
        return bottom - (v-ylo)/(yhi-ylo)*(bottom-top)
    def x(v):
        return left + v/budgets[final]*(right-left)
    for idx in range(round((yhi-ylo)/delta)+1):
        val = ylo+delta*idx
        fig.line(left, y(val), right, y(val))
        fig.text(left-9, y(val)+3.5, f"{val:g}", 10, MUTED, align="right")
    means = {}
    for idx, (method, (title, color)) in enumerate(SERIES.items()):
        keyx = 66+110*idx
        fig.line(keyx, 132, keyx+22, 132, color, 2, [4, 3] if method=="random" else None)
        fig.point(keyx+11, 132, color, square=method=="random", radius=2.7)
        fig.text(keyx+29, 135, title, 10.5)
        means[method] = {step: statistics.mean(series[(method, s)][step] for s in seeds) for step in steps}
        for a,b in zip(steps[:-1], steps[1:]):
            fig.line(x(budgets[a]), y(means[method][a]), x(budgets[b]), y(means[method][b]),
                     color, 2.1, [5, 3] if method=="random" else None)
        for step in steps:
            fig.point(x(budgets[step]), y(means[method][step]), color, square=method=="random")
    for step in steps:
        fig.text(x(budgets[step]), bottom+18, f"{budgets[step]:.2f}".rstrip("0").rstrip("."), 9.5, MUTED, align="center")
    fig.text((left+right)/2, 350, "累计揭示标签占目标查询池的比例（%）", 11, align="center")
    fig.line(468, 104, 468, 353)
    diffs = [series[("entropy", seed)][final]-series[("random", seed)][final] for seed in seeds]
    fig.text(500, 161, "种子", 10, MUTED)
    fig.text(730, 161, "差值", 10, MUTED, align="right")
    span = max(0.5, max(map(abs, diffs)))
    center = 613
    fig.line(center, 172, center, 281, GRID, 1.2, [2, 3])
    for i, (seed, diff) in enumerate(zip(seeds, diffs)):
        yy = 184+i*86/max(1,len(seeds)-1)
        xx = center+diff/span*54
        fig.text(501, yy+4, seed, 12)
        fig.line(center, yy, xx, yy, TEAL if diff>=0 else GOLD, 2)
        fig.point(xx, yy, TEAL if diff>=0 else GOLD, radius=3.6)
        fig.text(730, yy+4, f"{diff:+.2f}", 12, align="right")
    fig.text(center, 295, "0", 10, MUTED, align="center")
    fig.line(497, 308, 730, 308)
    fig.text(499, 330, "配对差均值", 12, bold=True)
    fig.text(730, 330, f"{statistics.mean(diffs):+.2f}", 14, bold=True, align="right")
    fig.finish(f"折线取种子均值。每条轨迹末轮累计揭示 {labels[final]} 张图像的参考标签，测试集独立封存。")
    return (f"同一迁移方向下，随机与熵选样各运行{len(seeds)}个配对种子；"
            f"折线显示测试Top-1均值，末轮预算为{labels[final]}个唯一目标标签。"
            f"末轮随机与熵选样均值分别为{means['random'][final]:.2f}%和{means['entropy'][final]:.2f}%，"
            f"平均配对差为{statistics.mean(diffs):+.2f}个百分点。右图逐一种子呈现差值。")


def workflow():
    rows = [r for r in read_rows(ROOT / "results/workflow/case_results.csv") if r["entry"]=="structured"]
    assert rows and len({r["case_id"] for r in rows}) == len(rows)
    def passed(row):
        if row["passed"].lower() not in ("0", "1", "true", "false"):
            raise ValueError(row["passed"])
        return row["passed"].lower() in ("1","true")
    observed = {(r["family"], r["split"]): [] for r in rows}
    for row in rows:
        observed[(row["family"], row["split"])].append(row)
    fig = Figure("workflow-checks", "结构化入口的任务执行验收",
                 "预设任务  ·  确定性工具后端      每个圆点代表一个已运行案例")
    fig.text(32, 110, "按任务族判定", 15, bold=True)
    fig.text(328, 110, "预算账本的三种变化", 15, bold=True)
    fig.line(299, 101, 299, 352)
    fig.text(148, 143, "开发", 11, MUTED, align="center")
    fig.text(220, 143, "固定评估", 11, MUTED, align="center")
    for index, (family, title) in enumerate(FAMILIES.items()):
        yy=170+29*index
        fig.text(32, yy+3, title, 12)
        for split, center in (("dev",148),("eval",220)):
            group=observed.get((family,split),[])
            for j,r in enumerate(group):
                xx=center+(j-(len(group)-1)/2)*16
                ok=passed(r)
                fig.point(xx,yy, TEAL if ok else GOLD, square=not ok, radius=4.7)
    fig.point(40, 344, TEAL, radius=4)
    fig.text(52, 348, "通过", 10.5, MUTED)
    fig.point(122, 344, GOLD, square=True, radius=4)
    fig.text(134, 348, "未通过", 10.5, MUTED)
    selected = {"budget-2":"调整剩余量", "budget-3":"下调低于已用量", "pause-3":"追加额度并暂停"}
    by_id={r["case_id"]:r for r in rows}
    fig.text(328, 143, "固定评估案例", 10, MUTED)
    fig.text(558, 143, "总预算（张）", 10, MUTED, align="center")
    fig.text(680, 143, "已用（张）", 10, MUTED, align="center")
    for i, (cid, desc) in enumerate(selected.items()):
        r=by_id[cid]
        yy=175+i*54
        fig.text(328, yy, desc, 13, bold=True)
        fig.text(328, yy+16, cid, 10, MUTED)
        fig.text(558, yy+3, f"{r['budget_before']} → {r['budget_after']}", 15, align="center")
        fig.text(680, yy+3, f"{r['spent_before']} → {r['spent_after']}", 15, align="center")
        fig.line(328, yy+26, 729, yy+26)
    evaluated = [r for r in rows if r["split"] == "eval"]
    fig.text(328, 344, "固定评估通过", 12, bold=True)
    fig.text(727, 344, f"{sum(map(passed,evaluated))} / {len(evaluated)}", 16, TEAL, bold=True, align="right")
    fig.finish("评分依据是终态、操作事件和业务约束。训练工具在流程验收中使用测试夹具。")
    return (f"结构化输入回放共{len(rows)}例，其中固定评估{len(evaluated)}例，"
            f"通过{sum(map(passed,evaluated))}例。左图列出逐族结果，右图从运行记录读取前后预算与"
            "已用标签数。该实测覆盖结构化工具执行，不估计语言指令理解准确率或真人任务耗时。")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", choices=("model","workflow","all"), nargs="?", default="all")
    parser.add_argument("--font", default="C:/Windows/Fonts/Deng.ttf")
    parser.add_argument("--font-bold", default="C:/Windows/Fonts/Dengb.ttf")
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for alias, name in (("cn",args.font),("cn-bold",args.font_bold)):
        pdfmetrics.registerFont(TTFont(alias, name))
    for name,slug,func in (("model","model-budget",model),("workflow","workflow-checks",workflow)):
        if args.figures in ("all",name):
            caption=func()
            (OUT/(slug+".txt")).write_text(caption+"\n",encoding="utf-8")
            print(slug)


if __name__ == "__main__":
    main()
