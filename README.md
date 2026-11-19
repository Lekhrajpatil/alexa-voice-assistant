# Alexa Voice Assistant

A voice-controlled assistant I built using OpenAI Whisper, a custom-trained intent classifier, and Streamlit. You speak a command, it transcribes your speech, figures out what you want, takes action, and speaks the response back to you.

## What it does

- Records your voice from the microphone
- Transcribes it using OpenAI Whisper (runs fully offline after the first download)
- Classifies your intent using a TF-IDF vectorizer and MLP neural network trained on my own dataset
- Performs the matching action — opens websites, tells the time, plays music, sets reminders, tells jokes, and more
- Speaks the response back using Windows text-to-speech

## Supported commands

| What you say | What it does |
|---|---|
| "play some music" | Opens YouTube music search |
| "what time is it" | Tells current date and time |
| "latest news" | Opens Google News |
| "tell me a joke" | Tells a joke |
| "set a timer for 5 minutes" | Confirms timer set |
| "weather in Mumbai" | Opens Google weather for that city |
| "open YouTube" | Opens YouTube |
| "what is 25 plus 17" | Calculates and answers |
| "add milk to shopping list" | Adds item to in-session list |
| "suggest a movie" | Opens IMDB Top 250 |
| "remind me about the meeting" | Saves reminder |

28 intents total — alarm, calculator, calendar, date/time, dictionary, directions, facts, games, general QA, jokes, movies, music, news, open website, personality, podcast, radio, reminder, routine, shopping list, smart home, timer, to-do, traffic, translation, unit conversion, weather, device status.

## Project structure

```
alexa_project_Lekhraj/
├── alexa.py          - main application
├── alexa_data.csv    - training dataset (4785 labelled samples, 28 intents)
├── requirements.txt  - dependencies
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/lekhrajpatil/alexa-voice-assistant.git
cd alexa-voice-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
streamlit run alexa.py
```

The Whisper `base` model (~140 MB) downloads automatically on first run and is cached at `~/.cache/whisper/`. After that it works fully offline.

> **Note:** pyttsx3 text-to-speech uses Windows SAPI5. This project runs on Windows. For Linux/Mac you would swap pyttsx3 for a compatible TTS library.

## How the intent model works

I collected voice command samples covering 28 intents and augmented them to around 4800 rows. Each prompt is cleaned and vectorized with TF-IDF, then fed into a two-layer MLP (`50 → 25` hidden units). Training takes under 10 seconds on CPU and accuracy on the training set exceeds 98%.

The model is retrained every time the app starts (takes ~8 seconds) so there is no stale model file to manage.

## Tech stack

- [OpenAI Whisper](https://github.com/openai/whisper) — speech to text
- [scikit-learn](https://scikit-learn.org/) — TF-IDF + MLP intent classifier
- [Streamlit](https://streamlit.io/) — web UI
- [sounddevice](https://python-sounddevice.readthedocs.io/) — microphone recording
- [scipy](https://scipy.org/) — WAV I/O and audio resampling
- [pyttsx3](https://pyttsx3.readthedocs.io/) — text to speech

## Author

**Lekhraj Patil**
