# app/export_report.py
"""
Auto-screenshot + PDF report generator for AI²-SED.
Captures the current dashboard, plots DEI trend, and embeds all visuals in one research-grade PDF.
"""

from __future__ import annotations
import io
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

# Optional screenshot library
try:
    import pyautogui
except ImportError:
    pyautogui = None

ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = ROOT / "data" / "synthetic"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_dei(df: pd.DataFrame) -> float:
    missed = (df["blue_label"] == 0).sum()
    miss_rate = missed / max(1, len(df))
    return round(100 - miss_rate * 100, 2)

def make_chart(df: pd.DataFrame, path: Path):
    """Save a simple DEI trend chart."""
    plt.figure(figsize=(6, 3))
    plt.plot(df["Round"], df["DEI"], marker="o", color="#00fff2", linewidth=2)
    plt.title("Defensive Effectiveness Index (DEI) Over Rounds", fontsize=10)
    plt.xlabel("Round")
    plt.ylabel("DEI (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def try_screenshot(path: Path):
    """Capture a screenshot of the Streamlit window if possible."""
    try:
        if pyautogui:
            time.sleep(2)
            img = pyautogui.screenshot()
            img.save(path)
            return path
    except Exception as e:
        print("Screenshot failed:", e)
    return None

def generate_pdf(author="Sunanda Vasanthi Tat"):
    if not SYN_DIR.exists():
        raise FileNotFoundError("No synthetic data folder found.")
    rounds = sorted(SYN_DIR.glob("round_*.csv"))
    if not rounds:
        raise RuntimeError("No round CSVs available.")

    # Prepare metrics
    meta = []
    for r in rounds:
        df = pd.read_csv(r)
        meta.append({
            "Round": int(r.stem.split("_")[-1]),
            "Samples": len(df),
            "DEI": compute_dei(df)
        })
    dfm = pd.DataFrame(meta).sort_values("Round")
    latest_dei = dfm["DEI"].iloc[-1]
    chart_path = REPORTS_DIR / "dei_trend.png"
    make_chart(dfm, chart_path)

    # Screenshot dashboard (optional)
    screenshot_path = REPORTS_DIR / "dashboard_shot.png"
    try_screenshot(screenshot_path)

    # === Create PDF ===
    pdf_path = REPORTS_DIR / f"AI2SED_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, h - 80, "AI²-SED: AI-vs-AI Social Engineering Defense")
    c.setFont("Helvetica", 12)
    c.drawString(50, h - 110, f"Author: {author}")
    c.drawString(50, h - 125, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, h - 130, w - 50, h - 130)

    # Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 160, "Summary")
    c.setFont("Helvetica", 12)
    c.drawString(70, h - 180, f"Total Rounds: {len(dfm)}")
    c.drawString(70, h - 195, f"Latest DEI: {latest_dei}%")
    c.drawString(70, h - 210, f"Average DEI: {dfm['DEI'].mean():.2f}%")
    c.drawString(70, h - 225, "Models: Baseline (TF-IDF + LogReg), RoBERTa, T5 Generator")

    # Chart
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 260, "Performance Trend")
    c.drawImage(ImageReader(str(chart_path)), 50, h - 500, width=500, height=200)

    # Dashboard snapshot
    if screenshot_path.exists():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, h - 520, "Live Dashboard Snapshot")
        c.drawImage(ImageReader(str(screenshot_path)), 50, h - 770, width=500, height=200)

    # Footer
    c.showPage()
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Generated automatically by the AI²-SED Unified Dashboard.")
    c.save()
    print(f"[OK] Saved report → {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    generate_pdf()
