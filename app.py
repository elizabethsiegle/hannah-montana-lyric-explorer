import csv
import re
import ast
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from gradient import Gradient
from exa_py import Exa

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hannah Montana — 20th Anniversary",
    page_icon="★",
    layout="wide",
)

# ── Palette ───────────────────────────────────────────────────────────────────
# Warm editorial: cream page, magenta accent, charcoal text
PAGE_BG   = "#FBF7F2"
CARD_BG   = "#FFFFFF"
BORDER    = "#E5DDD4"
TEXT      = "#1A1612"
MUTED     = "#7A6E66"
MAGENTA   = "#C8005A"   # primary — close to actual HM branding
AMBER     = "#B86E00"
BLUE      = "#1A56A0"
CHART_BG  = "#FFFFFF"
GRID_COL  = "#F0EAE2"

ERA_COLORS = {
    "Season 1":           "#C8005A",   # magenta
    "Season 2":           "#1A56A0",   # blue
    "Season 3":           "#6B3FA0",   # violet
    "The Movie":          "#B86E00",   # amber
    "Season 4 – Forever": "#1A7F5A",   # green
    "Compilations":       "#7A6E66",   # warm gray
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Inter:wght@400;500;600&display=swap');

  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] {
    background-color: #FBF7F2 !important;
    font-family: 'Inter', sans-serif;
    color: #1A1612;
  }
  [data-testid="stHeader"]          { background: transparent !important; }
  [data-testid="stSidebarContent"]  { background-color: #FFFFFF !important; }
  section[data-testid="stSidebar"]  {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5DDD4;
  }

  /* kill Streamlit's default h1/h2/h3 gold override */
  h1, h2, h3 { color: #1A1612 !important; }

  .display-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.1;
    color: #1A1612;
    margin: 0;
  }
  .display-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #7A6E66;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
  }
  .section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #C8005A;
    margin-bottom: 0.3rem;
  }
  .section-heading {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1A1612;
    margin: 0 0 0.2rem;
  }
  .section-caption {
    font-size: 0.82rem;
    color: #7A6E66;
    margin-bottom: 0.8rem;
  }

  .kpi-row { display: flex; gap: 0; border-top: 2px solid #1A1612; border-bottom: 1px solid #E5DDD4; padding: 1rem 0; }
  .kpi-item { flex: 1; padding: 0 1.5rem 0 0; }
  .kpi-item + .kpi-item { border-left: 1px solid #E5DDD4; padding-left: 1.5rem; }
  .kpi-num  { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #1A1612; line-height: 1; }
  .kpi-lbl  { font-size: 0.75rem; color: #7A6E66; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.06em; }

  .rule { border: none; border-top: 1px solid #E5DDD4; margin: 2.5rem 0; }

  /* Sidebar typography */
  [data-testid="stSidebarContent"] label,
  [data-testid="stSidebarContent"] p,
  [data-testid="stSidebarContent"] span { color: #1A1612 !important; }

  /* Radio buttons horizontal */
  div[role="radiogroup"] { gap: 0.4rem; }

  /* Lyrics box */
  .lyrics-box {
    background: #FFFFFF;
    border: 1px solid #E5DDD4;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    max-height: 300px;
    overflow-y: auto;
    font-size: 0.83rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #1A1612;
  }
  .song-meta { font-size: 0.82rem; color: #7A6E66; margin-bottom: 0.8rem; }
  .song-meta a { color: #C8005A; text-decoration: underline; }
  .song-meta strong { color: #1A1612; }

  /* Era legend dot */
  .era-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

  /* Sticky footer */
  .sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #FFFFFF;
    border-top: 1px solid #E5DDD4;
    padding: 0.55rem 1.5rem;
    font-size: 0.75rem;
    color: #7A6E66;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    box-sizing: border-box;
  }
  .sticky-footer a {
    color: #C8005A;
    text-decoration: none;
    font-weight: 500;
  }
  .sticky-footer a:hover { text-decoration: underline; }

  /* Era tabs — make them visible against the cream background */
  [data-testid="stTabs"] [role="tablist"] {
    gap: 0.4rem;
    border-bottom: 2px solid #E5DDD4;
  }
  [data-testid="stTabs"] button[role="tab"] {
    background: #FFFFFF !important;
    border: 1px solid #E5DDD4 !important;
    border-bottom: none !important;
    border-radius: 4px 4px 0 0 !important;
    color: #7A6E66 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.9rem !important;
  }
  [data-testid="stTabs"] button[role="tab"]:hover {
    background: #F0EAE2 !important;
    color: #1A1612 !important;
  }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    border-color: #1A1612 !important;
    color: #1A1612 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #FFFFFF !important;
  }

</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(label, heading, caption=""):
    st.markdown(
        f"<p class='section-label'>{label}</p>"
        f"<p class='section-heading'>{heading}</p>"
        + (f"<p class='section-caption'>{caption}</p>" if caption else ""),
        unsafe_allow_html=True,
    )

def rule():
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

def chart_layout(**kwargs):
    """Shared Plotly layout for a light editorial chart."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=11),
        margin=dict(l=10, r=60, t=10, b=10),
    )
    base.update(kwargs)
    return base

def axis_style(**kwargs):
    base = dict(gridcolor=GRID_COL, gridwidth=1, zeroline=False, linecolor=BORDER)
    base.update(kwargs)
    return base

# ── Load & parse data ─────────────────────────────────────────────────────────
def _genius_song_url(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[''']", "", slug)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return f"https://genius.com/Hannah-montana-{slug}-lyrics"

@st.cache_data
def load_songs():
    songs = []
    with open("lyrics/hannah_montana_lyrics.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            album_raw = row.get("album", "")
            try:
                album_dict    = ast.literal_eval(album_raw)
                album_name    = album_dict.get("name", "")
                release_date  = album_dict.get("release_date_for_display", "")
                album_url     = album_dict.get("url", "")
            except Exception:
                album_name   = album_raw
                release_date = ""
                album_url    = ""

            if "Forever" in album_name:
                era = "Season 4 – Forever"
            elif "Hannah Montana 3" in album_name:
                era = "Season 3"
            elif "Hannah Montana 2" in album_name or "Non-Stop" in album_name or "Rock Star" in album_name:
                era = "Season 2"
            elif "Movie" in album_name:
                era = "The Movie"
            elif "Remixed" in album_name or "Collection" in album_name or "Best of" in album_name or "Holiday" in album_name or "Anniversary" in album_name:
                era = "Compilations"
            else:
                era = "Season 1"

            songs.append({
                "title":        row["title"],
                "album":        album_name,
                "release_date": release_date,
                "album_url":    album_url,
                "genius_url":   _genius_song_url(row["title"]),
                "era":          era,
                "lyrics":       row["lyrics"],
                "word_count":   len(row["lyrics"].split()),
            })
    return pd.DataFrame(songs)

STOP = set(stopwords.words("english")) | {
    "oh", "yeah", "ooh", "ah", "hey", "la", "na", "whoa", "gonna", "gotta",
    "wanna", "cause", "'cause", "em", "let", "like", "know", "got", "get",
    "go", "come", "make", "one", "two", "back", "want", "still", "even",
    "said", "say", "see", "feel", "tell", "never", "always", "every",
    "right", "little", "around", "something", "way", "day", "time",
    "can", "would", "could", "think", "take", "show",
}

def clean_words(text):
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    return [w for w in text.lower().split() if w not in STOP and len(w) > 2]

@st.cache_data
def compute_word_data(df):
    all_words = []
    for _, row in df.iterrows():
        all_words.extend(clean_words(row["lyrics"]))
    return Counter(all_words)

@st.cache_data
def compute_yx_spotlight(df):
    """Compare 'Younger You' to the rest of the catalog."""
    sia = SentimentIntensityAnalyzer()
    yx = df[df["title"] == "Younger You"]
    if yx.empty:
        return None
    yx_row = yx.iloc[0]
    catalog = df[df["title"] != "Younger You"]

    # Word count
    yx_wc = yx_row["word_count"]
    avg_wc = catalog["word_count"].mean()
    wc_delta = round((yx_wc - avg_wc) / avg_wc * 100)

    # Average VADER compound per song
    def avg_sent(lyrics):
        lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
        if not lines:
            return 0.0
        return sum(sia.polarity_scores(l)["compound"] for l in lines) / len(lines)

    yx_sent = round(avg_sent(yx_row["lyrics"]), 2)
    cat_sent = round(catalog["lyrics"].apply(avg_sent).mean(), 2)

    # Signature words exclusive to "Younger You"
    all_counts_full = Counter(
        w for _, r in df.iterrows() for w in clean_words(r["lyrics"])
    )
    yx_words = clean_words(yx_row["lyrics"])
    yx_freq = Counter(yx_words)
    scores = {w: yx_freq[w] / (all_counts_full[w] + 1) for w in yx_freq}
    top_words = sorted(scores, key=scores.get, reverse=True)[:6]

    return {
        "yx_wc": yx_wc,
        "avg_wc": int(avg_wc),
        "wc_delta": wc_delta,
        "yx_sent": yx_sent,
        "cat_sent": cat_sent,
        "top_words": top_words,
        "release": yx_row["release_date"],
    }

@st.cache_data
def wordcloud_img(text, era="All Songs"):
    # Check if text is empty or too short
    if not text or not text.strip():
        # Return a simple figure with a message
        fig, ax = plt.subplots(figsize=(14, 3.7), facecolor="#FFFFFF")
        ax.text(0.5, 0.5, "No text available for wordcloud", 
                ha='center', va='center', fontsize=20, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    
    # Per-era accent colors fed into a single-hue custom colormap
    accent = {
        "All Songs":          "#C8005A",
        "Season 1":           "#C8005A",
        "Season 2":           "#1A56A0",
        "Season 3":           "#6B3FA0",
        "The Movie":          "#B86E00",
        "Season 4 – Forever": "#1A7F5A",
        "Compilations":       "#7A6E66",
    }.get(era, "#C8005A")

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    cmap = LinearSegmentedColormap.from_list("hm", ["#E8D8E0", accent])

    def color_func(*a, font_size=None, **kw):
        if font_size and font_size > 60:
            # Convert hex accent color to RGB tuple
            return tuple(int(accent[i:i+2], 16) for i in (1, 3, 5))
        else:
            # Convert matplotlib color to RGB integers
            color = plt.get_cmap("Greys")(0.45 + font_size / 200 if font_size else 0.5)[:3]
            return tuple(int(c * 255) for c in color)

    try:
        # Debug: Add minimum word count check
        if len(text.split()) < 5:
            fig, ax = plt.subplots(figsize=(14, 3.7), facecolor="#FFFFFF")
            ax.text(0.5, 0.5, f"Not enough words for wordcloud (only {len(text.split())} words)", 
                    ha='center', va='center', fontsize=16, color='orange')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.tight_layout(pad=0)
            return fig
            
        wc = WordCloud(
            width=1600, height=420,
            background_color="#FFFFFF",
            color_func=color_func,
            max_words=100,
            prefer_horizontal=0.65,
            min_font_size=10,
            max_font_size=120,
            random_state=7,
            collocations=False,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(14, 3.7), facecolor="#FFFFFF")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except Exception as e:
        # If wordcloud generation fails, return a simple figure with error message
        fig, ax = plt.subplots(figsize=(14, 3.7), facecolor="#FFFFFF")
        ax.text(0.5, 0.5, f"Error generating wordcloud: {str(e)[:100]}", 
                ha='center', va='center', fontsize=14, color='red', wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig


_BOILERPLATE_MARKERS = [
    "sign in", "register", "skip to content", "skip to main", "advertisement",
    "don't have an account", "table of contents", "edit source", "navigation menu",
    "retrieved from", "categories :", "this page was last",
]

def _clean_content(txt: str) -> str:
    """Strip nav/boilerplate lines from crawled page text."""
    lines = []
    for line in txt.splitlines():
        stripped = line.strip()
        if len(stripped) < 40:
            continue
        lower = stripped.lower()
        if any(marker in lower for marker in _BOILERPLATE_MARKERS):
            continue
        # Skip lines that are mostly markdown links or HTML entities
        if stripped.count("](http") > 1 or stripped.count("&amp;") > 2:
            continue
        lines.append(stripped)
    return " ".join(lines)


@st.cache_data(show_spinner=False)
def fetch_song_facts(song_title: str, release_year: str, album: str = "", lyrics: str = "") -> dict:
    """Use Exa to fetch content from music sites, then Claude to synthesize facts."""
    exa_key = os.environ.get("EXA_API_KEY", "")
    gradient_key = os.environ.get("MODEL_ACCESS_KEY", "")
    if not exa_key:
        return {"facts": None, "sources": [], "pop_context": [], "error": "No EXA_API_KEY"}
    try:
        exa_client = Exa(api_key=exa_key)

        # Build a disambiguating query using album when available
        album_hint = f' "{album}"' if album else ""
        query = f'"{song_title}" Hannah Montana{album_hint} song'

        song_res = exa_client.search_and_contents(
            query,
            num_results=6,
            type="neural",
            include_domains=[
                "en.wikipedia.org", "hannahmontana.fandom.com",
                "allmusic.com", "discogs.com",
            ],
            text={"max_characters": 1500},
        )

        content_pieces, sources = [], []
        seen_urls = set()
        for r in song_res.results:
            cleaned = _clean_content(r.text or "")
            if cleaned and len(cleaned) > 120 and r.url not in seen_urls:
                content_pieces.append(cleaned)
                sources.append({"title": r.title or r.url, "url": r.url})
                seen_urls.add(r.url)

        # Fall back to broader search if we don't have enough substance
        if len(content_pieces) < 2:
            broad_res = exa_client.search_and_contents(
                f'Hannah Montana "{song_title}"{album_hint}',
                num_results=5,
                type="neural",
                text={"max_characters": 1500},
            )
            for r in broad_res.results:
                cleaned = _clean_content(r.text or "")
                if cleaned and len(cleaned) > 120 and r.url not in seen_urls:
                    content_pieces.append(cleaned)
                    sources.append({"title": r.title or r.url, "url": r.url})
                    seen_urls.add(r.url)

        # Determine whether web content actually mentions this song by name
        title_lower = song_title.lower()
        relevant_pieces = [p for p in content_pieces if title_lower in p.lower()]

        # Claude synthesizes facts — from web content if available, lyrics if not
        facts_text = None
        claude_prompt = None
        if relevant_pieces:
            combined = "\n\n---\n\n".join(relevant_pieces[:3])
            claude_prompt = (
                f"You are a Hannah Montana music expert. Using ONLY the web content below about "
                f'the song "{song_title}", output exactly 3 bullet points.\n'
                f"Rules:\n"
                f"- No preamble, no intro, no numbering\n"
                f"- Start immediately with the first bullet using • as the symbol\n"
                f"- Each bullet is ONE specific sentence with concrete detail: "
                f"a real date, chart position, episode name/number, album name, "
                f"songwriter credit, or notable chart/cultural fact\n"
                f"- Do NOT write vague sentences like 'this song was popular' or "
                f"'it was featured on the show' — every bullet must contain a specific fact\n"
                f"- Do not invent anything not explicitly stated in the content below\n\n"
                f"Content:\n{combined}"
            )
        elif lyrics:
            # New or obscure song with no web coverage — analyse the lyrics directly
            claude_prompt = (
                f"You are a Hannah Montana music expert analyzing \"{song_title}\" "
                f"({album}, {release_year}). "
                f"The full lyrics are below. Output exactly 3 bullet points.\n"
                f"Rules:\n"
                f"- No preamble, no intro, no numbering\n"
                f"- Start immediately with the first bullet using • as the symbol\n"
                f"- Each bullet is ONE specific sentence highlighting something interesting "
                f"about the lyrics: a recurring image or motif, an emotional arc, "
                f"a specific line that captures the song's theme, or a lyrical technique\n"
                f"- Quote specific lines from the lyrics where possible\n\n"
                f"Lyrics:\n{lyrics[:2000]}"
            )
            sources = [{"title": "Lyrics analysis", "url": ""}]

        if claude_prompt and gradient_key:
            try:
                gc = Gradient(model_access_key=gradient_key)
                resp = gc.chat.completions.create(
                    messages=[{"role": "user", "content": claude_prompt}],
                    model="anthropic-claude-4.6-sonnet",
                    max_tokens=350,
                )
                msg = resp.choices[0].message
                facts_text = (msg.content or msg.reasoning_content or "").strip()
            except Exception:
                facts_text = relevant_pieces[0][:500] if relevant_pieces else None

        # Fetch pop-culture context via Wikipedia's "[year] in music" article
        pop_items = []
        if release_year and release_year.isdigit():
            pop_res = exa_client.search_and_contents(
                f"{release_year} in music",
                num_results=3,
                type="neural",
                include_domains=["en.wikipedia.org"],
                text={"max_characters": 1200},
            )
            for r in pop_res.results:
                txt = (r.text or "").strip()
                if txt and len(txt) > 120 and release_year in (r.title or ""):
                    # Extract a clean paragraph — skip nav/table-of-contents noise at the top
                    lines = [l.strip() for l in txt.splitlines() if len(l.strip()) > 60]
                    excerpt = " ".join(lines[:4])
                    # Trim to last complete sentence
                    last_stop = max(excerpt.rfind(". "), excerpt.rfind(".\n"))
                    excerpt = excerpt[: last_stop + 1].strip() if last_stop > 60 else excerpt[:400]
                    if excerpt:
                        pop_items.append({
                            "title": r.title or r.url,
                            "url": r.url,
                            "snippet": excerpt,
                        })

        # Fetch press & fan coverage — open search so we catch whoever actually has the story
        news_items = []
        news_res = exa_client.search_and_contents(
            f'"{song_title}" Hannah Montana Miley Cyrus',
            num_results=6,
            type="neural",
            text={"max_characters": 1500},
        )
        for r in news_res.results:
            txt = _clean_content(r.text or "")
            r_title_lower = (r.title or "").lower()
            # Must mention the song title explicitly
            if not (title_lower in txt.lower() or title_lower in r_title_lower):
                continue
            if len(txt) < 80:
                continue
            last_stop = max(txt.rfind(". "), txt.rfind(".\n"))
            snippet = txt[: last_stop + 1].strip() if last_stop > 60 else txt[:500]
            news_items.append({
                "title": r.title or r.url,
                "url": r.url,
                "snippet": snippet,
                "published": (r.published_date or "")[:10],
            })

        return {
            "facts": facts_text,
            "sources": sources[:2],
            "pop_context": pop_items,
            "news": news_items,
            "year": release_year,
        }
    except Exception as exc:
        return {"facts": None, "sources": [], "pop_context": [], "news": [], "error": str(exc)}


df = load_songs()
all_counts = compute_word_data(df)
ERA_ORDER = ["Season 1", "Season 2", "Season 3", "The Movie", "Season 4 – Forever", "Compilations"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:Playfair Display,serif;font-size:1.15rem;"
        "font-weight:700;color:#1A1612;margin-bottom:0'>Hannah Montana</p>"
        "<p style='font-size:0.72rem;color:#7A6E66;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-top:0'>Lyric Explorer</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border:none;border-top:1px solid #E5DDD4;margin:0.8rem 0">', unsafe_allow_html=True)
    selected_eras = st.multiselect("Filter by era", ERA_ORDER, default=ERA_ORDER)
    st.markdown('<hr style="border:none;border-top:1px solid #E5DDD4;margin:0.8rem 0">', unsafe_allow_html=True)
    top_n = st.slider("Words shown in frequency chart", 10, 50, 30, step=5)
    st.markdown(
        '<p style="font-size:0.75rem;color:#7A6E66;margin-top:1.5rem">'
        '80 songs · 2006–2026<br>Source: Genius Lyrics API</p>',
        unsafe_allow_html=True,
    )

filtered = df[df["era"].isin(selected_eras)] if selected_eras else df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='padding:2.5rem 0 1.5rem'>"
    "<p style='font-size:0.72rem;font-weight:600;letter-spacing:0.14em;"
    "text-transform:uppercase;color:#C8005A;margin-bottom:0.5rem'>20th Anniversary · 2006–2026</p>"
    "<p class='display-title'>Hannah Montana</p>"
    "<p class='display-title' style='font-style:italic;font-weight:400'>in her own words</p>"
    "<p class='display-sub'>A lyrical deep-dive into 80 songs from the Best of Both Worlds</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── KPI strip ─────────────────────────────────────────────────────────────────
kpis = [
    (f"{len(filtered)}", "songs"),
    (f"{filtered['word_count'].sum():,}", "total words"),
    (f"{int(filtered['word_count'].mean()):,}", "avg words / song"),
    (f"{filtered['word_count'].max():,}", "most words in one song"),
    (f"{len(all_counts):,}", "unique words"),
]
kpi_html = '<div class="kpi-row">' + "".join(
    f'<div class="kpi-item"><div class="kpi-num">{v}</div><div class="kpi-lbl">{l}</div></div>'
    for v, l in kpis
) + "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ── "Younger You" spotlight banner ────────────────────────────────────────────
_sp = compute_yx_spotlight(df)
if _sp:
    _wc_sign = f"+{_sp['wc_delta']}%" if _sp['wc_delta'] >= 0 else f"{_sp['wc_delta']}%"
    _sent_delta = round(_sp['yx_sent'] - _sp['cat_sent'], 2)
    _sent_sign = f"+{_sent_delta:.2f}" if _sent_delta >= 0 else f"{_sent_delta:.2f}"
    _words_html = "  ".join(
        f"<span style='background:#FDE8F0;color:{MAGENTA};border-radius:3px;"
        f"padding:0.1rem 0.45rem;font-size:0.75rem;font-weight:500'>{w}</span>"
        for w in _sp["top_words"]
    )
    st.markdown(
        f"""
        <div style='margin:1.6rem 0 0.4rem;border-left:4px solid {MAGENTA};
                    background:#FFF7FA;border-radius:0 6px 6px 0;
                    padding:1rem 1.4rem 1.1rem;'>
          <div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.55rem'>
            <span style='background:{MAGENTA};color:#fff;font-size:0.65rem;font-weight:700;
                         letter-spacing:0.1em;text-transform:uppercase;
                         border-radius:3px;padding:0.15rem 0.5rem'>New</span>
            <span style='font-family:"Playfair Display",Georgia,serif;font-size:1.1rem;
                         font-weight:700;color:{TEXT}'>"Younger You"</span>
            <span style='font-size:0.8rem;color:{MUTED}'>— Hannah Montana 20th Anniversary Special · {_sp["release"]}</span>
          </div>
          <div style='display:flex;gap:2.5rem;flex-wrap:wrap;margin-bottom:0.75rem'>
            <div>
              <div style='font-size:1.3rem;font-weight:700;font-family:"Playfair Display",serif;
                           color:{TEXT}'>{_sp["yx_wc"]}</div>
              <div style='font-size:0.7rem;color:{MUTED};text-transform:uppercase;
                           letter-spacing:0.06em'>words in this song
                <span style='color:{MAGENTA};margin-left:0.3rem;font-weight:600'>{_wc_sign} vs avg across all 79 other songs ({_sp["avg_wc"]})</span>
              </div>
            </div>
            <div>
              <div style='font-size:1.3rem;font-weight:700;font-family:"Playfair Display",serif;
                           color:{TEXT}'>{_sp["yx_sent"]:+.2f}</div>
              <div style='font-size:0.7rem;color:{MUTED};text-transform:uppercase;
                           letter-spacing:0.06em'>avg sentiment per line
                <span style='color:{MAGENTA};margin-left:0.3rem;font-weight:600'>{_sent_sign} vs avg across all 79 other songs ({_sp["cat_sent"]:+.2f})</span>
              </div>
            </div>
          </div>
          <div style='font-size:0.72rem;color:{MUTED};margin-bottom:0.4rem;
                       text-transform:uppercase;letter-spacing:0.08em'>
            Signature words
          </div>
          {_words_html}
          <p style='font-size:0.72rem;color:{MUTED};line-height:1.6;margin:0.85rem 0 0'>
            Sentiment is scored with <a href='https://github.com/cjhutto/vaderSentiment'
            style='color:{MAGENTA}'>VADER</a> (Valence Aware Dictionary and sEntiment Reasoner),
            a lexicon-based tool built specifically for expressive, short-form text — the same kind
            of language you find in song lyrics. Each word in its dictionary carries a pre-rated
            sentiment score; VADER combines those scores and adjusts for capitalization, punctuation,
            and intensifiers to produce a compound value between −1 (most negative) and +1 (most
            positive). Lyrics were fetched from <a href='https://genius.com' style='color:{MAGENTA}'>Genius</a>
            via their public API, so scores reflect the raw, unedited song text.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

rule()

# ── Word cloud ────────────────────────────────────────────────────────────────
section("Vocabulary", "Every word, all at once")

era_options = ["All Songs"] + [e for e in ERA_ORDER if e in selected_eras]
wc_choice = st.selectbox("Filter word cloud by era", era_options, index=0)

if wc_choice == "All Songs":
    wc_text = " ".join(w for _, row in filtered.iterrows() for w in clean_words(row["lyrics"]))
else:
    wc_text = " ".join(
        w for _, row in filtered[filtered["era"] == wc_choice].iterrows()
        for w in clean_words(row["lyrics"])
    )

# Debug: Check if text is empty
if not wc_text.strip():
    matching_songs = len(filtered[filtered["era"] == wc_choice]) if wc_choice != "All Songs" else len(filtered)
    st.warning(f"No text found for {wc_choice}. Songs matched: {matching_songs}")

wc_fig = wordcloud_img(wc_text, era=wc_choice)
st.pyplot(wc_fig, width='stretch')
plt.close("all")
st.markdown(
    f"<p style='font-size:0.72rem;color:{MUTED}'>"
    f"Common words (I, you, the, and, etc.) are filtered out as stopwords so more "
    f"meaningful lyrical words surface. To see a word like <em>you</em> here, it would "
    f"need to appear far more than average across all songs.</p>",
    unsafe_allow_html=True,
)

rule()

# ── Top words + era breakdown ─────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    section("Frequency", f"The {top_n} most-used words")

    filtered_counts = compute_word_data(filtered)
    top_words = filtered_counts.most_common(top_n)
    words_df = pd.DataFrame(top_words, columns=["word", "count"])

    # Single-hue bars — darker = more frequent
    max_c = words_df["count"].max()
    bar_colors = [
        f"rgba(200,0,90,{0.25 + 0.75*(c/max_c):.2f})" for c in words_df["count"]
    ]

    fig_bar = go.Figure(go.Bar(
        x=words_df["count"],
        y=words_df["word"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=words_df["count"],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="<b>%{y}</b> — %{x} times<extra></extra>",
    ))
    fig_bar.update_layout(
        **chart_layout(height=500),
        yaxis=dict(autorange="reversed", showgrid=False, linecolor=BORDER),
        xaxis=axis_style(title=None),
    )
    st.plotly_chart(fig_bar, width='stretch')

with col_right:
    section("Breakdown", "Songs by era")

    era_counts = (
        filtered.groupby("era").size()
        .reindex(ERA_ORDER).dropna().astype(int).reset_index()
    )
    era_counts.columns = ["era", "count"]

    fig_donut = go.Figure(go.Pie(
        labels=era_counts["era"],
        values=era_counts["count"],
        hole=0.62,
        marker=dict(
            colors=[ERA_COLORS[e] for e in era_counts["era"]],
            line=dict(color=CARD_BG, width=3),
        ),
        textinfo="percent",
        textfont=dict(size=11, color="#FFFFFF"),
        hovertemplate="<b>%{label}</b><br>%{value} songs (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        **chart_layout(height=300, margin=dict(l=10, r=10, t=10, b=10)),
        showlegend=True,
        legend=dict(
            font=dict(size=10, color=TEXT),
            bgcolor="rgba(0,0,0,0)",
            x=0.5, xanchor="center", y=-0.15, yanchor="top",
            orientation="v",
        ),
        annotations=[dict(
            text=f"<b>{era_counts['count'].sum()}</b><br><span style='font-size:10px'>songs</span>",
            x=0.5, y=0.5, font=dict(size=16, color=TEXT),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig_donut, width='stretch')

    st.markdown(
        f"<p style='font-size:0.72rem;color:{MUTED};margin:0.2rem 0 0.8rem'>"
        f"<strong style='color:{TEXT}'>Compilations</strong> includes remixed albums, "
        f"holiday releases, Best of collections, and the "
        f"<strong style='color:{TEXT}'>20th Anniversary Special</strong> "
        f"(2026) — songs that don't belong to a specific season.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:0.75rem;font-weight:600;letter-spacing:0.1em;"
        "text-transform:uppercase;color:#7A6E66;margin:1rem 0 0.3rem'>Avg length by era</p>",
        unsafe_allow_html=True,
    )
    era_len = (
        filtered.groupby("era")["word_count"]
        .mean().reindex(ERA_ORDER).dropna().round(0).astype(int).reset_index()
    )
    era_len.columns = ["era", "avg"]
    fig_len = go.Figure(go.Bar(
        x=era_len["era"],
        y=era_len["avg"],
        marker=dict(color=[ERA_COLORS[e] for e in era_len["era"]], line=dict(width=0)),
        text=era_len["avg"],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="<b>%{x}</b><br>avg %{y} words<extra></extra>",
    ))
    fig_len.update_layout(
        **chart_layout(height=180, margin=dict(l=5, r=5, t=5, b=5)),
        xaxis=dict(tickangle=-25, tickfont=dict(size=8), showgrid=False, linecolor=BORDER),
        yaxis=axis_style(title=None),
    )
    st.plotly_chart(fig_len, width='stretch')

rule()

# ── Signature words per era ───────────────────────────────────────────────────
section(
    "Identity",
    "Signature words by era",
    "Words that belong most exclusively to each era — scored by how much their local frequency exceeds their overall frequency.",
)

era_tabs = st.tabs([e for e in ERA_ORDER if e in selected_eras])

for tab, era in zip(era_tabs, [e for e in ERA_ORDER if e in selected_eras]):
    with tab:
        era_df = filtered[filtered["era"] == era]
        local_words = []
        for _, row in era_df.iterrows():
            local_words.extend(clean_words(row["lyrics"]))
        local_freq = Counter(local_words)
        scores = {
            w: local_freq[w] / (all_counts[w] + 1)
            for w in local_freq if len(w) > 2 and w not in STOP
        }
        top15 = sorted(scores, key=scores.get, reverse=True)[:15]
        top15_counts = [local_freq[w] for w in top15]
        accent = ERA_COLORS[era]
        bar_cols = [
            f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},{0.2 + 0.8*(c/max(top15_counts)):.2f})"
            for c in top15_counts[::-1]
        ]
        fig_sig = go.Figure(go.Bar(
            y=top15[::-1],
            x=top15_counts[::-1],
            orientation="h",
            marker=dict(color=bar_cols, line=dict(width=0)),
            text=top15_counts[::-1],
            textposition="outside",
            textfont=dict(color=MUTED, size=9),
            hovertemplate="<b>%{y}</b> — %{x} times<extra></extra>",
        ))
        fig_sig.update_layout(
            **chart_layout(height=380),
            xaxis=axis_style(title=None),
            yaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sig, width='stretch')

rule()

# ── Longest songs + theme breakdown ──────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    section("Length", "The 15 longest songs")

    top15_songs = (
        filtered.nlargest(15, "word_count")
        [["title", "era", "word_count", "album", "release_date", "genius_url"]]
        .reset_index(drop=True)
    )
    fig_songs = go.Figure(go.Bar(
        x=top15_songs["word_count"],
        y=top15_songs["title"],
        orientation="h",
        marker=dict(color=[ERA_COLORS[e] for e in top15_songs["era"]], line=dict(width=0)),
        text=top15_songs["word_count"],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        customdata=top15_songs[["era", "album", "release_date", "genius_url"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x} words · %{customdata[0]}<br>"
            "Album: %{customdata[1]}<br>"
            "Released: %{customdata[2]}<br>"
            "genius.com ↗ %{customdata[3]}"
            "<extra></extra>"
        ),
    ))
    fig_songs.update_layout(
        **chart_layout(height=420),
        yaxis=dict(autorange="reversed", showgrid=False, linecolor=BORDER),
        xaxis=axis_style(title=None),
    )
    st.plotly_chart(fig_songs, width='stretch')

with col_b:
    section("Themes", "What Hannah sings about")

    THEMES = {
        "Love & Heart":   ["love", "heart", "crush", "kiss", "baby", "forever"],
        "Fame & Star":    ["star", "fame", "world", "dream", "lights", "shine"],
        "Friends & True": ["friend", "true", "together", "real", "trust", "care"],
        "Free & Wild":    ["free", "fly", "wild", "run", "dance", "rock"],
    }
    theme_colors = [MAGENTA, AMBER, BLUE, "#1A7F5A"]

    counts_by_era = {}
    for era in [e for e in ERA_ORDER if e in selected_eras]:
        era_words = []
        for _, row in filtered[filtered["era"] == era].iterrows():
            era_words.extend(clean_words(row["lyrics"]))
        era_freq = Counter(era_words)
        counts_by_era[era] = {
            t: sum(era_freq.get(k, 0) for k in kws)
            for t, kws in THEMES.items()
        }

    theme_df = pd.DataFrame(counts_by_era).T.reset_index().rename(columns={"index": "era"})

    fig_theme = go.Figure()
    for theme, color in zip(THEMES.keys(), theme_colors):
        fig_theme.add_trace(go.Bar(
            name=theme,
            x=theme_df["era"],
            y=theme_df[theme],
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate=f"<b>{theme}</b> · %{{x}}<br>%{{y}} occurrences<extra></extra>",
        ))
    fig_theme.update_layout(
        barmode="group",
        **chart_layout(height=420, margin=dict(l=10, r=10, t=10, b=10)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT),
            orientation="h", yanchor="bottom", y=1.02,
        ),
        xaxis=dict(tickangle=-20, tickfont=dict(size=9), showgrid=False, linecolor=BORDER),
        yaxis=axis_style(title=None),
    )
    st.plotly_chart(fig_theme, width='stretch')

rule()

# ── Song explorer ─────────────────────────────────────────────────────────────
section("Deep dive", "Song explorer", "Pick a song — see its top words and full lyrics.")

sel_song = st.selectbox(
    "Song", options=filtered["title"].tolist(), label_visibility="collapsed"
)
song_row = filtered[filtered["title"] == sel_song].iloc[0]
col1, col2 = st.columns([2, 3])

with col1:
    song_words = clean_words(song_row["lyrics"])
    song_freq = Counter(song_words).most_common(20)
    sw_df = pd.DataFrame(song_freq, columns=["word", "count"])
    era_accent = ERA_COLORS.get(song_row["era"], MAGENTA)
    sw_max = sw_df["count"].max()
    sw_colors = [
        f"rgba({int(era_accent[1:3],16)},{int(era_accent[3:5],16)},{int(era_accent[5:7],16)},{0.2+0.8*(c/sw_max):.2f})"
        for c in sw_df["count"]
    ]
    fig_sw = go.Figure(go.Bar(
        x=sw_df["count"],
        y=sw_df["word"],
        orientation="h",
        marker=dict(color=sw_colors, line=dict(width=0)),
        text=sw_df["count"],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        customdata=[[song_row["album"], song_row["release_date"], song_row["genius_url"]]] * len(sw_df),
        hovertemplate=(
            "<b>%{y}</b> — %{x} times<br>"
            "Album: %{customdata[0]}<br>"
            "Released: %{customdata[1]}<br>"
            "genius.com ↗ %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    fig_sw.update_layout(
        **chart_layout(height=380),
        yaxis=dict(autorange="reversed", showgrid=False, linecolor=BORDER),
        xaxis=axis_style(title=None),
    )
    st.plotly_chart(fig_sw, width='stretch')

with col2:
    release = f" · {song_row['release_date']}" if song_row["release_date"] else ""
    st.markdown(
        f"<div class='song-meta'>"
        f"<strong>{song_row['era']}</strong>{release}<br>"
        f"{song_row['album'] or '—'}<br>"
        f"<a href='{song_row['genius_url']}' target='_blank'>View on Genius ↗</a>"
        f"</div>"
        f"<div class='lyrics-box'>{song_row['lyrics'][:3000]}"
        f"{'…' if len(song_row['lyrics']) > 3000 else ''}</div>",
        unsafe_allow_html=True,
    )

rule()

# ── Song Facts (Exa + Claude) ─────────────────────────────────────────────────
section(
    "Song facts",
    "Look up any song",
    "Pick a song — Exa pulls from Songfacts, Wikipedia & Genius, Claude distills the highlights.",
)

_sf_pick_col, _sf_btn_col, _ = st.columns([3, 1, 2])
with _sf_pick_col:
    _facts_song = st.selectbox(
        "facts_song",
        options=df["title"].tolist(),
        label_visibility="collapsed",
        key="facts_song_select",
    )
with _sf_btn_col:
    _run_facts = st.button("Research ↗", type="primary", key="run_facts")

if _run_facts:
    _facts_row = df[df["title"] == _facts_song].iloc[0]
    _year_match = re.search(r"\b(19|20)\d{2}\b", str(_facts_row["release_date"] or ""))
    _facts_year = _year_match.group(0) if _year_match else ""

    with st.spinner(f"Researching \"{_facts_song}\"…"):
        _facts_album = str(_facts_row.get("album", "") or "")
        _facts_lyrics = str(_facts_row.get("lyrics", "") or "")
        _facts_data = fetch_song_facts(_facts_song, _facts_year, _facts_album, _facts_lyrics)

    if _facts_data.get("error"):
        st.warning(f"Research unavailable: {_facts_data['error']}")
    else:
        _sf_col1, _sf_col2 = st.columns([3, 2])

        with _sf_col1:
            st.markdown(
                f"<p style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.08em;color:{MUTED};margin-bottom:0.8rem'>Did you know?</p>",
                unsafe_allow_html=True,
            )
            if _facts_data.get("facts"):
                _bullet_lines = [
                    l.strip() for l in _facts_data["facts"].splitlines()
                    if l.strip() and len(l.strip()) > 10
                ]
                for _bl in _bullet_lines:
                    _bl_clean = _bl.lstrip("•·-– ").strip()
                    st.markdown(
                        f"<div style='display:flex;gap:0.7rem;align-items:flex-start;"
                        f"margin-bottom:0.65rem'>"
                        f"<span style='color:{MAGENTA};font-size:1.1rem;line-height:1.4'>★</span>"
                        f"<p style='font-size:0.88rem;color:{TEXT};margin:0;line-height:1.6'>{_bl_clean}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if _facts_data.get("sources"):
                    _src_parts = []
                    for _s in _facts_data["sources"]:
                        _s_url = _s.get("url", "")
                        _s_parts = _s_url.split("/")
                        _s_domain = _s_parts[2] if len(_s_parts) > 2 else _s.get("title", "source")
                        if _s_url:
                            _src_parts.append(
                                f"<a href='{_s_url}' target='_blank' "
                                f"style='color:{MUTED};font-size:0.7rem'>{_s_domain}</a>"
                            )
                        else:
                            _src_parts.append(
                                f"<span style='color:{MUTED};font-size:0.7rem'>{_s_domain}</span>"
                            )
                    st.markdown(
                        f"<p style='margin-top:0.5rem;color:{MUTED};font-size:0.7rem'>"
                        f"Sources: {' · '.join(_src_parts)}</p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<p style='font-size:0.85rem;color:{MUTED}'>No facts found for this song.</p>",
                    unsafe_allow_html=True,
                )

        with _sf_col2:
            # News panel — shown when Vice/Billboard/Yahoo/Reddit have coverage
            _news = _facts_data.get("news", [])
            if _news:
                st.markdown(
                    f"<p style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:{MUTED};margin-bottom:0.8rem'>In the press &amp; online</p>",
                    unsafe_allow_html=True,
                )
                for _ni in _news:
                    _ni_domain = _ni["url"].split("/")[2] if _ni.get("url") else ""
                    _ni_date = f" · {_ni['published']}" if _ni.get("published") else ""
                    st.markdown(
                        f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
                        f"border-radius:4px;padding:0.75rem 1rem;margin-bottom:0.5rem'>"
                        f"<a href='{_ni['url']}' target='_blank' style='font-size:0.85rem;"
                        f"font-weight:600;color:{MAGENTA};text-decoration:none;line-height:1.4;"
                        f"display:block;margin-bottom:0.2rem'>{_ni['title']}</a>"
                        f"<p style='font-size:0.7rem;color:{MUTED};margin:0 0 0.35rem'>"
                        f"{_ni_domain}{_ni_date}</p>"
                        f"<p style='font-size:0.82rem;color:{TEXT};margin:0;line-height:1.55'>"
                        f"{_ni['snippet']}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Pop context — shown when no news, or as fallback below news
            if not _news and _facts_year:
                st.markdown(
                    f"<p style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:{MUTED};margin-bottom:0.8rem'>Pop in {_facts_year}</p>",
                    unsafe_allow_html=True,
                )
                if _facts_data.get("pop_context"):
                    for _pc in _facts_data["pop_context"]:
                        _pc_domain = _pc["url"].split("/")[2] if _pc.get("url") else ""
                        _raw = _pc["snippet"]
                        _last_stop = max(_raw.rfind(". "), _raw.rfind(".\n"))
                        _pc_text = _raw[: _last_stop + 1].strip() if _last_stop > 60 else _raw
                        st.markdown(
                            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
                            f"border-radius:4px;padding:0.75rem 1rem;margin-bottom:0.5rem'>"
                            f"<a href='{_pc['url']}' target='_blank' style='font-size:0.85rem;"
                            f"font-weight:600;color:{MAGENTA};text-decoration:none;line-height:1.4;"
                            f"display:block;margin-bottom:0.2rem'>{_pc['title']}</a>"
                            f"<p style='font-size:0.7rem;color:{MUTED};margin:0 0 0.35rem'>{_pc_domain}</p>"
                            f"<p style='font-size:0.82rem;color:{TEXT};margin:0;line-height:1.55'>"
                            f"{_pc_text}</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f"<p style='font-size:0.85rem;color:{MUTED}'>No context found for {_facts_year}.</p>",
                        unsafe_allow_html=True,
                    )

rule()

# ── Where is this word? ───────────────────────────────────────────────────────
section(
    "Word tracker",
    "Where does this word appear?",
    "Choose a word to see every song that uses it and how many times.",
)

wic_col1, wic_col2 = st.columns([1, 2])
with wic_col1:
    top50 = [w for w, _ in all_counts.most_common(50)]
    search_word  = st.selectbox("Common words", top50, index=0, key="wic_select")
    custom_word  = st.text_input("Or type your own", placeholder="e.g. guitar", key="wic_custom").strip().lower()
    target_word  = custom_word if custom_word else search_word

with wic_col2:
    st.markdown(
        f"<p style='font-size:0.85rem;color:{MUTED};margin-top:1.8rem'>"
        f"Showing results for <strong style='color:{TEXT}'>{target_word}</strong></p>",
        unsafe_allow_html=True,
    )

song_word_counts = []
for _, row in filtered.iterrows():
    cnt = clean_words(row["lyrics"]).count(target_word)
    if cnt > 0:
        song_word_counts.append({
            "title": row["title"], "era": row["era"],
            "album": row["album"], "release_date": row["release_date"],
            "genius_url": row["genius_url"], "count": cnt,
        })

if not song_word_counts:
    st.info(f'"{target_word}" doesn\'t appear in any song in the current selection.')
else:
    word_songs_df = pd.DataFrame(song_word_counts).sort_values("count", ascending=True)
    fig_wic = go.Figure(go.Bar(
        x=word_songs_df["count"],
        y=word_songs_df["title"],
        orientation="h",
        marker=dict(
            color=[ERA_COLORS.get(e, MAGENTA) for e in word_songs_df["era"]],
            line=dict(width=0),
        ),
        text=word_songs_df["count"],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        customdata=word_songs_df[["era", "album", "release_date", "genius_url"]].values,
        hovertemplate=(
            "<b>%{y}</b> — %{x}× <i>'%{meta}'</i><br>"
            "%{customdata[0]} · %{customdata[1]}<br>"
            "Released: %{customdata[2]}<br>"
            "genius.com ↗ %{customdata[3]}"
            "<extra></extra>"
        ),
        meta=target_word,
    ))
    fig_wic.update_layout(
        **chart_layout(height=max(280, len(word_songs_df) * 28 + 60)),
        yaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        xaxis=axis_style(title=f'"{target_word}" occurrences per song'),
    )

    legend_html = "  ".join(
        f'<span><span class="era-dot" style="background:{ERA_COLORS[e]}"></span>'
        f'<span style="font-size:0.78rem;color:{MUTED}">{e}</span></span>'
        for e in ERA_ORDER if e in word_songs_df["era"].values
    )
    st.markdown(f"<div style='margin-bottom:0.4rem'>{legend_html}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_wic, width='stretch')
    st.markdown(
        f"<p style='font-size:0.78rem;color:{MUTED}'>"
        f"<strong style='color:{TEXT}'>{target_word}</strong> appears in "
        f"{len(word_songs_df)} of {len(filtered)} songs "
        f"({len(word_songs_df)/len(filtered)*100:.0f}%) — "
        f"{word_songs_df['count'].sum()} total occurrences.</p>",
        unsafe_allow_html=True,
    )

rule()

# ── Heatmap ───────────────────────────────────────────────────────────────────
section(
    "Matrix",
    "Word × song heatmap",
    "How often the top words appear across every song. Columns sorted by era.",
)

hm_n_words = st.slider("Word rows", 10, 40, 20, step=5, key="hm_words")

top_hm_words = [w for w, _ in compute_word_data(filtered).most_common(hm_n_words)]

era_sort_map = {e: i for i, e in enumerate(ERA_ORDER)}
songs_sorted = filtered.copy()
songs_sorted["era_sort"] = songs_sorted["era"].map(era_sort_map)
songs_sorted = songs_sorted.sort_values(["era_sort", "title"]).reset_index(drop=True)

matrix = np.zeros((hm_n_words, len(songs_sorted)), dtype=int)
for j, (_, row) in enumerate(songs_sorted.iterrows()):
    freq = Counter(clean_words(row["lyrics"]))
    for i, w in enumerate(top_hm_words):
        matrix[i, j] = freq.get(w, 0)

short_titles = [t if len(t) <= 22 else t[:20] + "…" for t in songs_sorted["title"]]

song_meta = songs_sorted[["title", "era", "album", "release_date", "genius_url"]].values
hm_customdata = np.broadcast_to(
    song_meta[np.newaxis, :, :],
    (hm_n_words, len(songs_sorted), 5),
).copy()

fig_hm = go.Figure(go.Heatmap(
    z=matrix,
    x=short_titles,
    y=top_hm_words,
    customdata=hm_customdata,
    colorscale=[
        [0.0,  "#F8F2EC"],
        [0.25, "#F0C8D8"],
        [0.6,  "#C8005A"],
        [1.0,  "#6B0030"],
    ],
    hovertemplate=(
        "<b>%{y}</b> in <b>%{customdata[0]}</b> — %{z} times<br>"
        "%{customdata[1]} · %{customdata[2]}<br>"
        "Released: %{customdata[3]}<br>"
        "genius.com ↗ %{customdata[4]}"
        "<extra></extra>"
    ),
    showscale=True,
    colorbar=dict(
        title="count", title_font=dict(color=MUTED, size=10),
        tickfont=dict(color=MUTED, size=9),
        outlinecolor=BORDER, outlinewidth=1, thickness=10,
    ),
    xgap=1, ygap=1,
))

# Era boundary lines + labels
era_boundaries, added_eras = [], set()
current_era = songs_sorted.iloc[0]["era"]
for j, (_, row) in enumerate(songs_sorted.iterrows()):
    if row["era"] != current_era:
        era_boundaries.append(j - 0.5)
        current_era = row["era"]
    if row["era"] not in added_eras:
        fig_hm.add_annotation(
            x=j, y=hm_n_words - 0.5,
            text=f"<b>{row['era']}</b>",
            showarrow=False,
            font=dict(color=ERA_COLORS[row["era"]], size=7.5),
            xanchor="left", yanchor="bottom",
            yref="y", xref="x", yshift=4,
        )
        added_eras.add(row["era"])

for xb in era_boundaries:
    fig_hm.add_vline(x=xb, line_color=TEXT, line_width=1, opacity=0.15)

fig_hm.update_layout(
    **chart_layout(
        height=max(380, hm_n_words * 22 + 180),
        margin=dict(l=10, r=80, t=30, b=120),
    ),
    xaxis=dict(tickangle=-50, tickfont=dict(size=7.5), showgrid=False, linecolor=BORDER),
    yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=9.5)),
)
st.plotly_chart(fig_hm, width='stretch')

# ── Younger You — Sentiment Arc ───────────────────────────────────────────────
rule()
section(
    "20th Anniversary · 2026",
    "Younger You — emotional arc",
    "Line-by-line sentiment through the new song, scored with VADER.",
)

_sia = SentimentIntensityAnalyzer()

_younger_you_row = df[df["title"] == "Younger You"]
if not _younger_you_row.empty:
    _lyrics = _younger_you_row.iloc[0]["lyrics"]
    _lines = [l.strip() for l in _lyrics.splitlines() if l.strip()]
    _scores = [_sia.polarity_scores(l)["compound"] for l in _lines]

    # Smooth with a rolling average (window=3)
    _smoothed = pd.Series(_scores).rolling(3, center=True, min_periods=1).mean().tolist()

    _line_nums = list(range(1, len(_lines) + 1))

    # Colour each point by positive / negative
    _point_colors = [MAGENTA if s >= 0 else AMBER for s in _smoothed]

    fig_yx = go.Figure()

    # Filled area
    fig_yx.add_trace(go.Scatter(
        x=_line_nums,
        y=_smoothed,
        mode="lines",
        line=dict(color=MAGENTA, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(200,0,90,0.08)",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Dots with line text on hover
    fig_yx.add_trace(go.Scatter(
        x=_line_nums,
        y=_smoothed,
        mode="markers",
        marker=dict(color=_point_colors, size=8, line=dict(color="#fff", width=1.5)),
        customdata=_lines,
        hovertemplate="<i>%{customdata}</i><br>sentiment: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    # Zero baseline
    fig_yx.add_hline(y=0, line_color=BORDER, line_width=1)

    # Annotate the most positive and most negative lines
    _max_i = int(pd.Series(_smoothed).idxmax())
    _min_i = int(pd.Series(_smoothed).idxmin())
    for _i, _label, _ay in [(_max_i, "most hopeful", -40), (_min_i, "most wistful", 40)]:
        fig_yx.add_annotation(
            x=_line_nums[_i], y=_smoothed[_i],
            text=f"<b>{_label}</b>",
            showarrow=True, arrowhead=2, arrowcolor=MUTED,
            ay=_ay, ax=0,
            font=dict(size=9, color=MUTED),
        )

    fig_yx.update_layout(
        **chart_layout(height=340, margin=dict(l=10, r=10, t=10, b=40)),
        xaxis=dict(
            title="line", tickfont=dict(size=9), showgrid=False,
            linecolor=BORDER, zeroline=False,
        ),
        yaxis=dict(
            title="sentiment", tickfont=dict(size=9),
            range=[-1.05, 1.05], gridcolor=GRID_COL, zeroline=False,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["−1 negative", "−0.5", "0 neutral", "+0.5", "+1 positive"],
        ),
    )
    st.plotly_chart(fig_yx, width='stretch')
    st.markdown(
        f"<p style='font-size:0.72rem;color:{MUTED}'>"
        f"Each point is one lyric line. The curve is a 3-line rolling average. "
        f"<span style='color:{MAGENTA}'>●</span> positive &nbsp;·&nbsp; "
        f"<span style='color:{AMBER}'>●</span> negative. "
        f"Hover a dot to read the line.</p>",
        unsafe_allow_html=True,
    )

# ── Song Recommender ──────────────────────────────────────────────────────────
rule()
section(
    "Song recommender",
    "Find your song",
    "Answer three quick questions and we'll match you to a Hannah Montana track.",
)

q_col1, q_col2, q_col3 = st.columns(3)
with q_col1:
    st.markdown(f"<p style='font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:0.25rem'>How are you feeling right now?</p>", unsafe_allow_html=True)
    feeling_now = st.text_input("feeling_now", placeholder="e.g. anxious, excited, meh…", label_visibility="collapsed", key="feeling_now")
with q_col2:
    st.markdown(f"<p style='font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:0.25rem'>How do you want to feel?</p>", unsafe_allow_html=True)
    feeling_want = st.text_input("feeling_want", placeholder="e.g. pumped up, at peace…", label_visibility="collapsed", key="feeling_want")
with q_col3:
    st.markdown(f"<p style='font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:0.25rem'>What's your zodiac sign?</p>", unsafe_allow_html=True)
    zodiac = st.selectbox(
        "zodiac",
        ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"],
        label_visibility="collapsed", key="zodiac",
    )

rec_btn_col, _ = st.columns([1, 4])
with rec_btn_col:
    run_rec = st.button("Recommend a song ↗", type="primary", key="run_rec")

if run_rec:
    if not feeling_now.strip() or not feeling_want.strip():
        st.warning("Please answer the first two questions before getting a recommendation.")
    else:
        song_titles = "\n".join(f"- {t}" for t in df["title"].tolist())
        rec_prompt = (
            f"You are a Hannah Montana expert and music therapist. "
            f"Based on the user's answers, recommend exactly one Hannah Montana song from the list below.\n\n"
            f"User answers:\n"
            f"- How they're feeling now: {feeling_now}\n"
            f"- How they want to feel: {feeling_want}\n"
            f"- Zodiac sign: {zodiac}\n\n"
            f"Available songs:\n{song_titles}\n\n"
            f"Respond in exactly this format — nothing else:\n"
            f"Song: <title>\n"
            f"Why: <one sentence>\n"
        )
        with st.spinner("Finding your song…"):
            try:
                api_key = os.environ.get("MODEL_ACCESS_KEY")
                if not api_key:
                    st.error("MODEL_ACCESS_KEY environment variable is not set.")
                else:
                    rec_client = Gradient(model_access_key=api_key)
                    rec_response = rec_client.chat.completions.create(
                        messages=[{"role": "user", "content": rec_prompt}],
                        model="anthropic-claude-4.6-sonnet",
                        max_tokens=100,
                    )
                    rec_msg = rec_response.choices[0].message
                    rec_text = rec_msg.content or rec_msg.reasoning_content

                    # Parse out title and reason
                    rec_lines = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                                 for line in rec_text.strip().splitlines() if ":" in line}
                    rec_title = rec_lines.get("Song", "")
                    rec_why = rec_lines.get("Why", rec_text)

                    st.markdown(
                        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:4px;"
                        f"padding:1.4rem 1.6rem;margin-top:0.75rem;line-height:1.8'>"
                        f"<p style='font-size:1.15rem;font-weight:700;color:{MAGENTA};margin:0 0 0.4rem'>🎵 {rec_title}</p>"
                        f"<p style='margin:0;color:{TEXT}'>{rec_why}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error getting recommendation: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<hr class="rule">'
    f'<p style="font-size:0.75rem;color:{MUTED}">'
    f"Lyrics sourced from the Genius API via lyricsgenius. "
    f"80 songs · Hannah Montana (2006–2026).</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sticky-footer">'
    'made w/ ♥ in sf🌉 &nbsp;·&nbsp; github repo: '
    '<a href="https://github.com/elizabethsiegle/hannah-montana-lyric-explorer" target="_blank" rel="noopener">'
    'hannah-montana-lyric-explorer'
    '</a>'
    '</div>',
    unsafe_allow_html=True,
)
