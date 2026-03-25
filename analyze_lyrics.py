import os
import csv
from dotenv import load_dotenv
from gradient import Gradient

load_dotenv()

client = Gradient(
    model_access_key=os.environ.get("MODEL_ACCESS_KEY"),
)

# ── Load all song titles from the CSV ────────────────────────────────────────
songs = []
with open("lyrics/hannah_montana_lyrics.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["lyrics"].strip():
            songs.append(row["title"])

song_list = "\n".join(f"- {title}" for title in songs)

# ── Ask the user a few questions ─────────────────────────────────────────────
print("Let's find your perfect Hannah Montana song!\n")
feeling_now = input("How are you feeling right now? ")
feeling_want = input("How do you want to feel? ")
zodiac = input("What is your zodiac sign? ")

print("\n✨ Finding your song...\n")

# ── Recommendation prompt ─────────────────────────────────────────────────────
prompt = f"""You are a Hannah Montana expert and music therapist. Based on the user's answers below, recommend exactly one Hannah Montana song from the list provided. Return only the song title, followed by a one-sentence explanation of why it fits.

User answers:
- How they're feeling now: {feeling_now}
- How they want to feel: {feeling_want}
- Zodiac sign: {zodiac}

Available Hannah Montana songs:
{song_list}

Respond in this format:
Song: <title>
Why: <one sentence>
"""

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="anthropic-claude-4.6-sonnet",
    max_tokens=100,
)

msg = response.choices[0].message
print(msg.content or msg.reasoning_content)
