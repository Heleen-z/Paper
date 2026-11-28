"""Render the Chinese Markdown working draft, including its evidence tables and figures."""
import html
import os
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/初步研究与撰写/论文前三章初稿.md"
DEST = ROOT / "output/论文初步验证版.pdf"
WIDTH, HEIGHT = A4
MARGIN = 49
CONTENT = WIDTH - 2 * MARGIN - 13
INK = "#233340"
TEAL = "#12685E"
MUTED = "#546674"
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def rich(text):
    def link(match):
        label, target = match.groups()
        if target.startswith("https://"):
            return f'<a href="{target}" color="{TEAL}">{label}</a>'
        return label
    return LINK.sub(link, html.escape(text)).replace("**", "").replace(chr(96), "")


def font_setup():
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name, file in [("CJK", "msyh.ttc"), ("CJKb", "msyhbd.ttc")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file), subfontIndex=0))


def style(name, size, leading, color=INK, **kw):
    return ParagraphStyle(name, fontSize=size, leading=leading, textColor=color,
                          wordWrap="CJK", **{"fontName": "CJK", **kw})


def render():
    font_setup()
    body = style("body", 10.2, 17.1, spaceAfter=6, firstLineIndent=20.4)
    title = style("title", 24.5, 33, fontName="CJKb", spaceBefore=7, spaceAfter=10)
    chapter = style("chapter", 16.4, 23, TEAL, fontName="CJKb", spaceBefore=14,
                    spaceAfter=8, keepWithNext=True)
    section = style("section", 12.2, 18, fontName="CJKb", spaceBefore=8,
                    spaceAfter=5, keepWithNext=True)
    small = style("note", 9, 14, MUTED, spaceAfter=6)
    caption = style("caption", 9, 14, MUTED, spaceBefore=5, spaceAfter=8)
    cell = style("cell", 8.7, 13.2)
    cell_head = style("cell head", 8.7, 13.2, TEAL, fontName="CJKb")
    story = []
    lines = SOURCE.read_text(encoding="utf-8-sig").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("|"):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            items = []
            for n, row in enumerate(rows):
                values = [x.strip() for x in row.strip("|").split("|")]
                if all(re.fullmatch(r"[-: ]+", x) for x in values):
                    continue
                items.append([Paragraph(rich(x), cell_head if n == 0 else cell) for x in values])
            table = Table(items, colWidths=[CONTENT / len(items[0])] * len(items[0]), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5F1EE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(TEAL)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend([table, Spacer(1, 9)])
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(line)
            label, target = match.groups()
            path = (SOURCE.parent / target).resolve()
            with PILImage.open(path) as pic:
                w, h = pic.size
            scale = min(CONTENT / w, 355 / h)
            story.append(KeepTogether([Image(str(path), width=w*scale, height=h*scale),
                                       Paragraph(rich(label), caption)]))
        elif line.startswith("### "):
            story.append(Paragraph(rich(line[4:]), section))
        elif line.startswith("## "):
            story.extend([CondPageBreak(125), Paragraph(rich(line[3:]), chapter)])
        elif line.startswith("# "):
            story.append(Paragraph(rich(line[2:]), title))
        else:
            is_note = line.startswith(("本科毕业论文", "关键词", "[R", "数据来源", "图注", "表5"))
            story.append(Paragraph(rich(line), small if is_note else body))

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setTitle("融合主动域适应的交互式多模态智能体研究")
        canvas.setFont("CJK", 8.3)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(MARGIN, HEIGHT - 33, "人工智能　 /　 本科毕业论文工作稿")
        canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 33, "初步验证版　·　2026.11")
        canvas.setStrokeColor(colors.HexColor("#A9C8C3"))
        canvas.line(MARGIN, HEIGHT - 42, WIDTH - MARGIN, HEIGHT - 42)
        canvas.line(MARGIN, 37, WIDTH - MARGIN, 37)
        canvas.drawString(MARGIN, 23, "Office-Home　 /　 交互式主动域适应")
        canvas.drawRightString(WIDTH - MARGIN, 23, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(DEST), pagesize=A4, title="融合主动域适应的交互式多模态智能体研究",
                            leftMargin=MARGIN, rightMargin=MARGIN, topMargin=51, bottomMargin=44)
    doc.build(story, onFirstPage=frame, onLaterPages=frame)


if __name__ == "__main__":
    render()
