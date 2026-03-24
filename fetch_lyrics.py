import lyricsgenius
import csv
import time
import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

genius = lyricsgenius.Genius(
    ACCESS_TOKEN,
    remove_section_headers=True,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    timeout=15,
    retries=3,
)

output_path = "/Users/lsiegle/Desktop/demos/hannahmontana/lyrics/hannah_montana_lyrics.csv"

print("Fetching Hannah Montana songs...")
artist = genius.search_artist("Hannah Montana", max_songs=100, sort="popularity")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "album", "year", "lyrics"])
    writer.writeheader()
    for song in artist.songs:
        writer.writerow({
            "title": song.title,
            "album": song.album if song.album else "",
            "year": getattr(song, "year", "") or "",
            "lyrics": song.lyrics if song.lyrics else "",
        })

print(f"\nDone! Saved {len(artist.songs)} songs to {output_path}")
