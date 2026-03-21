# src/pdf_report.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
from reportlab.pdfgen import canvas


# ----------------------------
# Markdown -> Plain text cleanup
# ----------------------------
def strip_markdown(md: str) -> str:
    """
    Turn simple Markdown into plain text suitable for executive PDF.
    Removes headers (#), bold/italics markers, backticks, and normalizes bullets.
    """
    s = md if md is not None else ""

    # Remove headings like "# Title" / "## Title"
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)

    # Remove bold/italic markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    # Remove inline code backticks
    s = s.replace("`", "")

    # Convert "- " bullets to "• "
    s = re.sub(r"(?m)^\s*-\s+", "• ", s)

    # Collapse 3+ newlines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


# ----------------------------
# PDF helpers
# ----------------------------
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
    """
    Draw text with simple word-wrap. Returns the new y position after drawing.
    """
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

    # Normalize line endings
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = text.split("\n")

    for par in paragraphs:
        if par.strip() == "":
            y -= line_height  # blank line
            continue
        for line in split_paragraph(par):
            # Page break if needed
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
    """
    Draw image, fitting to width/height bounds while preserving aspect ratio.
    Returns the new y position below the image.
    """
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
    subtitle: Optional[str] = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)

    dash = Path(dashboard_png)
    week_md = Path(week_report_md)
    reg_md = Path(regression_md)
    mit_md = Path(mitigations_md)

    c = canvas.Canvas(str(out), pagesize=letter)
    page_w, page_h = letter

    # ---------- Cover / Header ----------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, page_h - 72, title)

    c.setFont("Helvetica", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(72, page_h - 92, f"Generated: {ts}")

    if subtitle:
        c.drawString(72, page_h - 110, subtitle)

    # ---------- Dashboard (page 1) ----------
    y = page_h - 140
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "1) Reliability Certificate Dashboard")
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

    # Helper for text sections
    def add_section(heading: str, body: str) -> None:
        x = 72
        y = page_h - 72
        max_w = page_w - 144

        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, heading)
        y -= 18

        y = _draw_wrapped_text(
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

    # ---------- Section 2: Week report ----------
    add_section("2) Week Report", strip_markdown(_safe_read_text(week_md)))

    # ---------- Section 3: Regression ----------
    add_section("3) Regression Check (V1 vs V2)", strip_markdown(_safe_read_text(reg_md)))

    # ---------- Section 4: Required Mitigations ----------
    add_section("4) Required Mitigations (Fix-it Engine)", strip_markdown(_safe_read_text(mit_md)))

    c.save()
    print(f"✅ Executive PDF saved: {out}")


if __name__ == "__main__":
    build_executive_pdf()