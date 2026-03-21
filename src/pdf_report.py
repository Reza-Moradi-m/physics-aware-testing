# src/pdf_report.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
from reportlab.pdfgen import canvas


def strip_markdown(md: str) -> str:
    """
    Convert simple markdown to cleaner plain text for PDF output.
    """
    s = md if md is not None else ""

    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    s = s.replace("`", "")
    s = re.sub(r"(?m)^\s*-\s+", "• ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    line_height: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
) -> float:
    c.setFont(font_name, font_size)

    def split_paragraph(par: str) -> list[str]:
        words = par.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            trial = w if not cur else (cur + " " + w)
            if c.stringWidth(trial, font_name, font_size) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = text.split("\n")

    for par in paragraphs:
        if par.strip() == "":
            y -= line_height
            continue

        for line in split_paragraph(par):
            if y < 72:
                c.showPage()
                y = letter[1] - 72
                c.setFont(font_name, font_size)

            c.drawString(x, y, line)
            y -= line_height

    return y


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return f"[Missing file: {path}]"
    return path.read_text(encoding="utf-8", errors="replace")


def _draw_image_fit_width(
    c: canvas.Canvas,
    img_path: Path,
    x: float,
    y_top: float,
    max_width: float,
    max_height: float,
) -> float:
    if not img_path.exists():
        c.setFont("Helvetica", 11)
        c.drawString(x, y_top, f"[Missing image: {img_path}]")
        return y_top - 20

    img = utils.ImageReader(str(img_path))
    iw, ih = img.getSize()

    scale_w = max_width / float(iw)
    scale_h = max_height / float(ih)
    scale = min(scale_w, scale_h)

    w = iw * scale
    h = ih * scale

    y_bottom = y_top - h
    c.drawImage(str(img_path), x, y_bottom, width=w, height=h, preserveAspectRatio=True, mask="auto")
    return y_bottom - 18


def build_executive_pdf(
    out_path: str = "results/executive_report.pdf",
    dashboard_png: str = "results/master_dashboard.png",
    week_report_md: str = "results/week06_report.md",
    regression_md: str = "results/regression_summary.md",
    mitigations_md: str = "results/mitigations.md",
    title: str = "RobustAI Engine — Executive Reliability Report",
    subtitle: Optional[str] = "Operational stress audit with model comparison, safety gate, and mitigations",
) -> None:
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)

    dash = Path(dashboard_png)
    week_md = Path(week_report_md)
    reg_md = Path(regression_md)
    mit_md = Path(mitigations_md)

    c = canvas.Canvas(str(out), pagesize=letter)
    page_w, page_h = letter

    # Cover
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, page_h - 72, title)

    c.setFont("Helvetica", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(72, page_h - 92, f"Generated: {ts}")

    if subtitle:
        c.drawString(72, page_h - 110, subtitle)

    y = page_h - 140
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "1) Reliability Dashboard")
    y -= 16

    y = _draw_image_fit_width(
        c,
        dash,
        x=72,
        y_top=y,
        max_width=page_w - 144,
        max_height=page_h - 220,
    )

    c.showPage()

    def add_section(heading: str, body: str) -> None:
        x = 72
        y = page_h - 72
        max_w = page_w - 144

        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, heading)
        y -= 18

        _draw_wrapped_text(
            c,
            body,
            x=x,
            y=y,
            max_width=max_w,
            line_height=12,
            font_name="Helvetica",
            font_size=10,
        )
        c.showPage()

    add_section("2) Audit Summary", strip_markdown(_safe_read_text(week_md)))
    add_section("3) Regression Summary", strip_markdown(_safe_read_text(reg_md)))
    add_section("4) Mitigation Recommendations", strip_markdown(_safe_read_text(mit_md)))

    c.save()
    print(f"Executive PDF saved: {out}")


if __name__ == "__main__":
    build_executive_pdf()