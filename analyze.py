import csv
import re
import os
from collections import Counter
from itertools import islice
from typing import Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt

# comment downloader + sentiment
from youtube_comment_downloader import YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # make langdetect deterministic

# translation: try googletrans first, fallback to deep-translator
TRANSLATOR_MODE = None
try:
    from googletrans import Translator as GT_Translator  # type: ignore
    _gt = GT_Translator()
    # test quick translate (no network test) - we'll just prefer it
    TRANSLATOR_MODE = "googletrans"
except Exception:
    TRANSLATOR_MODE = "deep-translator"

if TRANSLATOR_MODE == "deep-translator":
    from deep_translator import GoogleTranslator as DT_Translator  # type: ignore

# reportlab for PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.pdfgen import canvas


# Configuration

COMMENT_LIMIT = 1000
CSV_TEMPLATE = "{video_id}_comments.csv"
PDF_TEMPLATE = "{video_id}_report.pdf"
PIE_TEMPLATE = "{video_id}_sentiment_pie.png"
HIST_TEMPLATE = "{video_id}_compound_hist.png"
TOP_WORDS_COUNT = 20
SAMPLE_PER_SENT = 6

# Utility helpers

# robust emoji & symbol stripper (keeps letters, numbers, common punctuation)
_EMOJI_PATTERN = re.compile(
    "["                                      # all emoji ranges
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def remove_emojis_and_symbols(text: str) -> str:
    if not text:
        return ""
    # remove emoji ranges
    text = _EMOJI_PATTERN.sub("", text)
    # remove other weird symbols but keep basic punctuation .,!?'"-:
    # this will remove control chars and unusual symbols
    text = re.sub(r"[^\w\s\.,!\?'\-\":;()\/&%#@]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_video_id(url: str) -> str:
    """Extract YouTube video id from common URL formats."""
    if "youtu.be/" in url:
        return url.split("/")[-1].split("?")[0]
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    # fallback pattern
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def translate_text(text: str) -> Tuple[str, str]:
    """
    Translate text to English.
    Returns tuple (detected_lang, translated_text).
    Uses googletrans if available, otherwise deep-translator.
    """
    if not text:
        return ("unknown", "")

    # first detect language
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"

    # if already English (or detection failed but text looks english), skip translate
    if lang == "en" or lang == "unknown":
        return (lang, text)

    # try googletrans first (may not be installed or may break on some Python versions)
    if TRANSLATOR_MODE == "googletrans":
        try:
            translated = _gt.translate(text, dest="en")
            return (lang, translated.text if hasattr(translated, "text") else str(translated))
        except Exception:
            # fallback to deep-translator below
            pass

    # deep-translator fallback
    try:
        translated = DT_Translator(source="auto", target="en").translate(text)
        return (lang, translated)
    except Exception:
        # if translation fails, return original text as fallback
        return (lang, text)

def classify_sentiment(analyzer: SentimentIntensityAnalyzer, text: str) -> Tuple[str, float]:
    """Run VADER on text (assumes English) and return (label, compound)."""
    if not text:
        return ("Neutral", 0.0)
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        return ("Positive", compound)
    elif compound <= -0.05:
        return ("Negative", compound)
    else:
        return ("Neutral", compound)

def extract_words(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens

# Main analysis function

def analyze_comments(video_url: str, limit: int = COMMENT_LIMIT) -> dict:
    downloader = YoutubeCommentDownloader()
    vader = SentimentIntensityAnalyzer()

    video_id = extract_video_id(video_url) or "report"
    csv_file = CSV_TEMPLATE.format(video_id=video_id)

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    all_words = []
    rows = []
    sample_comments = {"Positive": [], "Negative": [], "Neutral": []}
    compound_scores = []
    processed = 0

    print(f"\nAnalyzing up to {limit} comments for: {video_url}\n")

    comments_iter = downloader.get_comments_from_url(video_url, sort_by=1)

    for comment in tqdm(islice(comments_iter, limit), total=limit, unit="c"):
        orig_raw = comment.get("text", "") or ""
        # remove emojis and noisy symbols (this will improve translation and sentiment)
        cleaned = remove_emojis_and_symbols(orig_raw)
        if not cleaned:
            continue

        # detect & translate
        lang, translated = translate_text(cleaned)

        # sentiment on translated text (English)
        sentiment, compound = classify_sentiment(vader, translated)

        counts[sentiment] += 1
        compound_scores.append(compound)
        all_words.extend(extract_words(translated))

        # keep sample comments (short list)
        if len(sample_comments[sentiment]) < SAMPLE_PER_SENT:
            sample_comments[sentiment].append({
                "original": orig_raw,
                "cleaned": cleaned,
                "lang": lang,
                "translated": translated,
                "compound": compound
            })

        rows.append([orig_raw, cleaned, lang, translated, sentiment, f"{compound:.4f}"])
        processed += 1

    # save CSV
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Original Comment", "Cleaned (no emoji)", "Detected Lang", "Translated (EN)", "Sentiment", "Compound Score"])
        w.writerows(rows)

    top_words = Counter(all_words).most_common(TOP_WORDS_COUNT)

    return {
        "video_id": video_id,
        "csv_file": csv_file,
        "counts": counts,
        "top_words": top_words,
        "samples": sample_comments,
        "compound_scores": compound_scores,
        "processed": processed
    }

# Charts 

def generate_charts(video_id: str, counts: dict, compound_scores: list) -> Tuple[str, str]:
    labels = list(counts.keys())
    sizes = [counts[l] for l in labels]

    pie_path = PIE_TEMPLATE.format(video_id=video_id)
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=[f"{l} ({s})" for l,s in zip(labels,sizes)], autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="w"))
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=150)
    plt.close()

    hist_path = HIST_TEMPLATE.format(video_id=video_id)
    plt.figure(figsize=(8,4.5))
    plt.hist(compound_scores, bins=20)
    plt.title("Compound Sentiment Score Distribution")
    plt.xlabel("Compound Score")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    return pie_path, hist_path

# PDF 

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        super().showPage()

    def save(self):
        page_count = len(self.pages)
        for i, page in enumerate(self.pages):
            self.__dict__.update(page)
            self.draw_page_number(i + 1, page_count)
            super().showPage()
        super().save()

    def draw_page_number(self, page_num, total):
        page_str = f"Page {page_num}/{total}"
        self.setFont("Helvetica", 9)
        width, height = A4
        self.drawRightString(width - 2 * cm, 1 * cm, page_str)


def build_pdf(report_data: dict, pie_path: str, hist_path: str) -> str:
    vid = report_data["video_id"]
    pdf_file = PDF_TEMPLATE.format(video_id=vid)

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    small = ParagraphStyle("small", parent=normal, fontSize=9)

    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    story = []

    # -------- TITLE PAGE --------
    story.append(Spacer(1, 2 * cm))
    story.append(
        Paragraph("YouTube Comment Sentiment Analysis",
                  ParagraphStyle("title", parent=h1, fontSize=26))
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Video ID: <b>{vid}</b>", normal))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Total Comments Analyzed: <b>{report_data['processed']}</b>", normal)
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Light Theme Report — Emoji/noisy symbols removed before translation & sentiment",
            small,
        )
    )
    story.append(PageBreak())

    # -------- SUMMARY --------
    story.append(Paragraph("Summary", h1))
    total = report_data["processed"] or 1
    pos = report_data["counts"]["Positive"]
    neg = report_data["counts"]["Negative"]
    neu = report_data["counts"]["Neutral"]

    summary_html = f"""
    <b>Total comments</b>: {total}<br/>
    <b>Positive</b>: {pos} ({pos/total:.1%})<br/>
    <b>Negative</b>: {neg} ({neg/total:.1%})<br/>
    <b>Neutral</b>: {neu} ({neu/total:.1%})
    """

    story.append(Paragraph(summary_html, normal))
    story.append(Spacer(1, 0.5 * cm))
    story.append(PageBreak())

    # -------- CHARTS --------
    story.append(Paragraph("Charts", h1))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Sentiment Distribution (Pie Chart)", h2))
    story.append(Image(pie_path, width=14 * cm, height=11 * cm))
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("Compound Score Distribution (Histogram)", h2))
    story.append(Image(hist_path, width=14 * cm, height=8 * cm))
    story.append(PageBreak())

    # -------- TOP WORDS --------
    story.append(Paragraph("Top Words", h1))
    story.append(Spacer(1, 0.2 * cm))

    tbl = [["Word", "Count"]] + [
        [w, str(c)] for w, c in report_data["top_words"]
    ]
    t = Table(tbl, colWidths=[10 * cm, 3 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )

    story.append(t)
    story.append(PageBreak())

    # -------- SAMPLE COMMENTS --------
    story.append(Paragraph("Sample Comments", h1))
    story.append(Spacer(1, 0.2 * cm))

    table_rows = [
        ["Sentiment", "Original Comment", "Lang", "Translated", "Score"]
    ]

    for sentiment in ("Positive", "Negative", "Neutral"):
        for sample in report_data["samples"].get(sentiment, []):
            orig = sample["original"]
            trans = sample["translated"]

            orig_p = Paragraph(
                (orig[:600] + "...") if len(orig) > 600 else orig, normal
            )
            trans_p = Paragraph(
                (trans[:600] + "...") if len(trans) > 600 else trans, normal
            )

            table_rows.append(
                [
                    Paragraph(sentiment, normal),
                    orig_p,
                    Paragraph(sample["lang"], normal),
                    trans_p,
                    Paragraph(f"{sample['compound']:.3f}", normal),
                ]
            )

    col_widths = [3 * cm, 7 * cm, 2 * cm, 7 * cm, 2 * cm]

    sample_table = Table(table_rows, colWidths=col_widths, repeatRows=1)

    ts = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    # alternating background for rows
    for i in range(1, len(table_rows)):
        bg = colors.whitesmoke if i % 2 else colors.white
        ts.add("BACKGROUND", (0, i), (-1, i), bg)

    sample_table.setStyle(ts)
    story.append(sample_table)

    # -------- BUILD PDF --------
    doc.build(story, canvasmaker=NumberedCanvas)
    return pdf_file


# CLI

def main():
    print("YouTube Comment Sentiment Analyzer — Light Theme")
    print("Note: translation will try googletrans first (if installed) and fall back to deep-translator.\n")
    url = input("Please paste the YouTube video URL and press Enter: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return

    data = analyze_comments(url, limit=COMMENT_LIMIT)
    if data["processed"] == 0:
        print("No comments were processed. Exiting.")
        return

    pie, hist = generate_charts(data["video_id"], data["counts"], data["compound_scores"])
    pdf_file = build_pdf(data, pie, hist)

    print("\n--- Done ---")
    print(f"CSV: {data['csv_file']}")
    print(f"Pie chart: {pie}")
    print(f"Histogram: {hist}")
    print(f"PDF report: {pdf_file}")
    print("------------------\n")

if __name__ == "__main__":
    main()
