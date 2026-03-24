import csv
import re
import ast
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Hannah Montana palette ────────────────────────────────────────────────────
PURPLE   = "#7B2D8B"
LAVENDER = "#C084FC"
TEAL     = "#06B6D4"
PINK     = "#F472B6"
GOLD     = "#FBBF24"
BG_DARK  = "#0F0A1E"
BG_MID   = "#1E1040"
TEXT_COL = "#F3E8FF"

# ── Load data ─────────────────────────────────────────────────────────────────
songs = []
with open("lyrics/hannah_montana_lyrics.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        album_raw = row.get("album", "")
        try:
            album_dict = ast.literal_eval(album_raw)
            album_name = album_dict.get("name", "")
        except Exception:
            album_name = album_raw

        # Normalize album names to major seasons
        if "Forever" in album_name:
            season = "Season 4 – Forever"
        elif "Hannah Montana 3" in album_name:
            season = "Season 3"
        elif "Hannah Montana 2" in album_name or "Non-Stop" in album_name or "Rock Star" in album_name:
            season = "Season 2"
        elif "Movie" in album_name:
            season = "The Movie"
        elif "Remixed" in album_name or "Collection" in album_name or "Best of" in album_name or "Holiday" in album_name:
            season = "Compilations"
        else:
            season = "Season 1"

        songs.append({
            "title": row["title"],
            "album": album_name,
            "season": season,
            "lyrics": row["lyrics"],
        })

# ── Text cleaning ─────────────────────────────────────────────────────────────
STOP = set(stopwords.words("english")) | {
    "oh", "yeah", "ooh", "ah", "hey", "la", "na", "whoa", "gonna", "gotta",
    "wanna", "cause", "'cause", "em", "let", "like", "know", "got", "get",
    "go", "come", "make", "one", "two", "back", "want", "still", "even",
    "said", "say", "see", "feel", "tell", "never", "always", "every",
    "right", "little", "around", "something", "way", "day", "time",
    "can", "would", "could", "think", "take", "show",
}

def clean(text):
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    return text.lower()

all_words = []
for s in songs:
    words = [w for w in clean(s["lyrics"]).split() if w not in STOP and len(w) > 2]
    all_words.extend(words)

all_lyrics_text = " ".join(all_words)
word_counts = Counter(all_words)

# Per-season word counts
season_words = {}
for s in songs:
    words = [w for w in clean(s["lyrics"]).split() if w not in STOP and len(w) > 2]
    season_words.setdefault(s["season"], []).extend(words)

# Song lengths (word count)
song_lengths = sorted(
    [(s["title"], len(s["lyrics"].split())) for s in songs],
    key=lambda x: x[1],
    reverse=True,
)

# ── Big figure canvas ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28), facecolor=BG_DARK)
fig.patch.set_facecolor(BG_DARK)

# Grid: 4 rows × 3 cols
gs = fig.add_gridspec(
    4, 3,
    hspace=0.55, wspace=0.35,
    left=0.05, right=0.97, top=0.93, bottom=0.04,
)

# ── Header ────────────────────────────────────────────────────────────────────
ax_title = fig.add_axes([0, 0.94, 1, 0.06])
ax_title.set_facecolor(BG_DARK)
ax_title.axis("off")

# Sparkle stars
rng = np.random.default_rng(42)
for _ in range(60):
    ax_title.plot(
        rng.uniform(0.01, 0.99), rng.uniform(0.05, 0.95),
        marker="*", color=GOLD,
        markersize=rng.uniform(3, 9), alpha=rng.uniform(0.4, 1.0),
        transform=ax_title.transAxes,
    )

ax_title.text(
    0.5, 0.65, "✦  HANNAH MONTANA: 20th Anniversary  ✦",
    ha="center", va="center", fontsize=28, fontweight="bold",
    color=GOLD, transform=ax_title.transAxes,
    fontfamily="DejaVu Sans",
)
ax_title.text(
    0.5, 0.18, "A lyrical deep-dive into the Best of Both Worlds  •  79 Songs  •  2006 – 2010",
    ha="center", va="center", fontsize=12, color=LAVENDER,
    transform=ax_title.transAxes,
)

# ── 1. Lyric Word Cloud (row 0, spans all 3 cols) ─────────────────────────────
ax_wc = fig.add_subplot(gs[0, :])
ax_wc.set_facecolor(BG_MID)

wc = WordCloud(
    width=2000, height=420,
    background_color=BG_MID,
    colormap="cool",
    max_words=120,
    prefer_horizontal=0.75,
    contour_width=0,
    stopwords=set(),          # already cleaned
    min_font_size=10,
    max_font_size=140,
    random_state=7,
    collocations=False,
).generate(all_lyrics_text)

ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
ax_wc.set_title(
    "Most-Used Words Across All 79 Songs",
    color=GOLD, fontsize=14, fontweight="bold", pad=8,
)

# ── 2. Top-30 Words Bar Chart (row 1, col 0-1) ───────────────────────────────
ax_bar = fig.add_subplot(gs[1, :2])
ax_bar.set_facecolor(BG_MID)

top_words = word_counts.most_common(30)
words_list, counts_list = zip(*top_words)
colors_bar = [
    PINK if i % 3 == 0 else (TEAL if i % 3 == 1 else LAVENDER)
    for i in range(len(words_list))
]

bars = ax_bar.barh(
    range(len(words_list)), counts_list,
    color=colors_bar, edgecolor="none", height=0.7,
)
ax_bar.set_yticks(range(len(words_list)))
ax_bar.set_yticklabels(words_list, color=TEXT_COL, fontsize=9)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Occurrences", color=LAVENDER, fontsize=9)
ax_bar.set_title("Top 30 Lyric Words", color=GOLD, fontsize=13, fontweight="bold")
ax_bar.tick_params(colors=LAVENDER, labelsize=8)
ax_bar.spines[:].set_color(PURPLE)
ax_bar.set_facecolor(BG_MID)
for spine in ax_bar.spines.values():
    spine.set_color(PURPLE)
ax_bar.xaxis.label.set_color(LAVENDER)
ax_bar.tick_params(axis="x", colors=LAVENDER)

# Add count labels
for bar, count in zip(bars, counts_list):
    ax_bar.text(
        bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
        str(count), va="center", color=TEXT_COL, fontsize=7,
    )

# ── 3. Songs per Season Donut (row 1, col 2) ─────────────────────────────────
ax_donut = fig.add_subplot(gs[1, 2])
ax_donut.set_facecolor(BG_MID)

season_order = ["Season 1", "Season 2", "Season 3", "The Movie", "Season 4 – Forever", "Compilations"]
season_colors = [PINK, TEAL, LAVENDER, GOLD, "#A78BFA", "#34D399"]
season_counts = [sum(1 for s in songs if s["season"] == sn) for sn in season_order]

wedges, texts, autotexts = ax_donut.pie(
    season_counts,
    labels=None,
    colors=season_colors,
    autopct="%1.0f%%",
    pctdistance=0.78,
    startangle=120,
    wedgeprops={"width": 0.55, "edgecolor": BG_DARK, "linewidth": 2},
)
for at in autotexts:
    at.set_color(BG_DARK)
    at.set_fontsize(8)
    at.set_fontweight("bold")

ax_donut.set_title("Songs by Era", color=GOLD, fontsize=13, fontweight="bold")
legend_patches = [
    mpatches.Patch(color=c, label=f"{sn} ({ct})")
    for sn, c, ct in zip(season_order, season_colors, season_counts)
]
ax_donut.legend(
    handles=legend_patches, loc="lower center",
    bbox_to_anchor=(0.5, -0.38), ncol=1,
    fontsize=7, frameon=False, labelcolor=TEXT_COL,
)

# ── 4. Top Unique Words per Season (row 2, spans all cols) ───────────────────
ax_theme = fig.add_subplot(gs[2, :])
ax_theme.set_facecolor(BG_MID)
ax_theme.axis("off")

ax_theme.text(
    0.5, 1.0, "Signature Words by Era  (words that appear most exclusively in each era)",
    ha="center", va="top", fontsize=13, fontweight="bold",
    color=GOLD, transform=ax_theme.transAxes,
)

show_seasons = [s for s in season_order if s != "Compilations"]
s_colors = dict(zip(season_order, season_colors))

global_freq = Counter(all_words)

col_width = 1.0 / len(show_seasons)
for i, sn in enumerate(show_seasons):
    wds = season_words.get(sn, [])
    if not wds:
        continue
    local_freq = Counter(wds)
    # TF-IDF-like: local_freq / global_freq
    scores = {
        w: local_freq[w] / (global_freq[w] + 1)
        for w in local_freq if len(w) > 2 and w not in STOP
    }
    top = sorted(scores, key=scores.get, reverse=True)[:8]

    cx = col_width * i + col_width / 2
    ax_theme.text(
        cx, 0.88, sn,
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=s_colors[sn], transform=ax_theme.transAxes,
    )
    for j, w in enumerate(top):
        size = 12 - j * 0.8
        alpha = 1.0 - j * 0.08
        ax_theme.text(
            cx, 0.72 - j * 0.085, w,
            ha="center", va="top", fontsize=size,
            color=s_colors[sn], alpha=alpha,
            transform=ax_theme.transAxes,
            fontweight="bold" if j == 0 else "normal",
        )

# Vertical dividers
for i in range(1, len(show_seasons)):
    ax_theme.plot(
        [col_width * i, col_width * i], [0, 1],
        color=PURPLE, linewidth=0.8, alpha=0.5,
        transform=ax_theme.transAxes,
    )

# ── 5. Longest Songs Bar (row 3, col 0-1) ────────────────────────────────────
ax_long = fig.add_subplot(gs[3, :2])
ax_long.set_facecolor(BG_MID)

top_n = 15
top_songs = song_lengths[:top_n]
titles_s, lengths_s = zip(*top_songs)
short_titles = [t if len(t) <= 28 else t[:26] + "…" for t in titles_s]

grad_colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
bars2 = ax_long.barh(range(top_n), lengths_s, color=grad_colors, edgecolor="none", height=0.7)
ax_long.set_yticks(range(top_n))
ax_long.set_yticklabels(short_titles, color=TEXT_COL, fontsize=8.5)
ax_long.invert_yaxis()
ax_long.set_xlabel("Word Count", color=LAVENDER, fontsize=9)
ax_long.set_title("Longest Songs by Word Count", color=GOLD, fontsize=13, fontweight="bold")
ax_long.spines[:].set_color(PURPLE)
ax_long.set_facecolor(BG_MID)
for spine in ax_long.spines.values():
    spine.set_color(PURPLE)
ax_long.tick_params(colors=LAVENDER, labelsize=8)
ax_long.xaxis.label.set_color(LAVENDER)
ax_long.tick_params(axis="x", colors=LAVENDER)

for bar, count in zip(bars2, lengths_s):
    ax_long.text(
        bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
        str(count), va="center", color=TEXT_COL, fontsize=7.5,
    )

# ── 6. Love vs. Star vs. Free word families (row 3, col 2) ──────────────────
ax_theme2 = fig.add_subplot(gs[3, 2])
ax_theme2.set_facecolor(BG_MID)

themes = {
    "Love & Heart": ["love", "heart", "crush", "kiss", "baby", "forever"],
    "Fame & Star":  ["star", "fame", "world", "dream", "lights", "shine"],
    "Friends & True": ["friend", "true", "together", "real", "trust", "care"],
    "Free & Wild": ["free", "fly", "wild", "run", "dance", "rock"],
}
theme_cols = [PINK, GOLD, TEAL, LAVENDER]
theme_vals = []
theme_labels = []

for (label, keywords), col in zip(themes.items(), theme_cols):
    count = sum(word_counts.get(k, 0) for k in keywords)
    theme_vals.append(count)
    theme_labels.append(label)

bars3 = ax_theme2.bar(
    range(len(theme_labels)), theme_vals,
    color=theme_cols, edgecolor="none", width=0.6,
)
ax_theme2.set_xticks(range(len(theme_labels)))
ax_theme2.set_xticklabels(theme_labels, color=TEXT_COL, fontsize=7.5, rotation=12, ha="right")
ax_theme2.set_ylabel("Word Occurrences", color=LAVENDER, fontsize=8)
ax_theme2.set_title("Song Themes Frequency", color=GOLD, fontsize=13, fontweight="bold")
ax_theme2.spines[:].set_color(PURPLE)
ax_theme2.set_facecolor(BG_MID)
for spine in ax_theme2.spines.values():
    spine.set_color(PURPLE)
ax_theme2.tick_params(colors=LAVENDER, labelsize=8)
ax_theme2.yaxis.label.set_color(LAVENDER)
ax_theme2.tick_params(axis="y", colors=LAVENDER)

for bar, val in zip(bars3, theme_vals):
    ax_theme2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
        str(val), ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.015,
    "Made with  ♥  for the 20th Anniversary of Hannah Montana  •  Data from Genius Lyrics API",
    ha="center", fontsize=9, color=LAVENDER, style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "lyrics/hannah_montana_20th_anniversary.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
print(f"Saved → {out}")
plt.close(fig)
