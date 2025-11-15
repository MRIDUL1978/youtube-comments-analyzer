# YouTube Comment Sentiment Analyzer 🎥📊  
*A Clean, Automated Analysis Pipeline for YouTube Comments*

This project analyzes YouTube video comments using:
- **Emoji & symbol cleaning**
- **Language detection**
- **Automatic translation to English**
- **VADER sentiment analysis**
- **Top-word frequency extraction**
- **CSV export**
- **Professional PDF report generation (light theme)**
- **Pie chart + histogram visualization**

---

## 🚀 Features

### ✔ Extract up to 1000 YouTube comments  
Uses `youtube-comment-downloader` for reliable scraping.

### ✔ Clean text (remove emojis & noisy symbols)  
Improves translation quality and sentiment accuracy.

### ✔ Detect and translate any language  
- **Primary translator:** `googletrans`  
- **Fallback:** `deep-translator`  
- Ensures best-effort English translation even for mixed-language comments.

### ✔ Sentiment analysis  
Uses VADER for:
- Positive  
- Negative  
- Neutral  
- Compound score histogram

### ✔ Generate outputs  
- **CSV file** with all processed comments  
- **PNG pie chart**  
- **PNG histogram**  
- **Full professional PDF report**  
  - Summary  
  - Charts  
  - Top words  
  - Sample comments table  
  - Pagination  
  - Clean, light theme

---

## 📂 Project Structure

my_youtube_analyzer/
│
├── analyze.py
├── requirements.txt
├── README.md
├── .gitignore
├── venv/ # (ignored by git)
├── <videoid>_comments.csv
├── <videoid>_sentiment_pie.png
├── <videoid>_compound_hist.png
└── <videoid>_report.pdf



---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>


Create and activate virtual environment

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

Install dependencies

pip install -r requirements.txt


▶ Usage

Run the script:
python analyze.py

You will be prompted to enter a YouTube video URL:

Please paste the YouTube video URL and press Enter:

When finished, the tool will output:

CSV file

Pie chart

Histogram

PDF report


🧪 Requirements

See requirements.txt for full dependency list.
