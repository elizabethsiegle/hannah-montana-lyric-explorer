# Hannah Montana Lyric Explorer in honor of the 20th anniversary!

A deep dive into 79 Hannah Montana songs spanning 2006-2010, with interactive visualizations and AI-powered insights.

## What's Inside

- **Word clouds** filtered by era with custom color schemes
- **Frequency analysis** of the most-used words across all songs
- **Era breakdowns** showing how Hannah's vocabulary evolved
- **Song explorer** with lyrics and word analysis
- **AI critic** powered by Nvidia's Nemotron model for witty song reviews

## Tech Stack

- **Visualizations**: Plotly, Matplotlib, WordCloud
- **Data**: 79+ songs from Genius API via lyricsgenius
- **AI Reviews**: Nvidia Nemotron on DigitalOcean
- **Deployment**: [DigitalOcean App Platform](https://www.digitalocean.com/products/app-platform)

The color palette and typography are inspired by the actual Hannah Montana branding from the mid-2000s.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need API keys for Genius (lyrics) and Gradient AI (song reviews) in your `.env` file.

## Data Notes

Songs are categorized into eras: Season 1, Season 2, Season 3, The Movie, Season 4 – Forever, and Compilations. The word analysis filters out common stopwords and focuses on meaningful lyrics that capture Hannah's themes of fame, friendship, and growing up.