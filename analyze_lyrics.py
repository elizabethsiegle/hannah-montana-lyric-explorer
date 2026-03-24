import os
import csv
import ast
import random
from dotenv import load_dotenv
from gradient import Gradient

load_dotenv()

client = Gradient(
    model_access_key=os.environ.get("MODEL_ACCESS_KEY"),
)

# ── Pick a random song from the CSV ──────────────────────────────────────────
songs = []
with open("lyrics/hannah_montana_lyrics.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["lyrics"].strip():
            songs.append({"title": row["title"], "lyrics": row["lyrics"]})

song = random.choice(songs)
# Trim to first ~800 words so we stay within context
lyrics_snippet = " ".join(song["lyrics"].split()[:800])

print(f"Analyzing: {song['title']}\n{'─' * 50}")

# ── Analysis prompt ───────────────────────────────────────────────────────────
prompt = f"""Analyze the following Hannah Montana song lyrics and provide insights on the themes, tone, and overall vibe of the song. Be concise but thorough in your analysis. Return only the analysis and no reasoning whatsoever. Do not think it through, just return the analysis. Do not return any text that is not part of the analysis.

Song: "{song['title']}"
Lyrics: {lyrics_snippet}

"""

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model= "openai-gpt-oss-120b",
    max_tokens=200,
)

msg = response.choices[0].message
print(msg.content or msg.reasoning_content)
