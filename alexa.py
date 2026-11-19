"""
Alexa Voice Assistant
Author      : Lekhraj Patil
Description : A browser-based voice assistant. Record your voice in the
              browser, Whisper transcribes it, a TF-IDF + MLP classifier
              detects the intent, an action is taken, and the response is
              played back as audio — all inside Streamlit, no local
              dependencies like ffmpeg or pyttsx3 required.
"""

import io
import os
import re
import base64
import random
import datetime
import tempfile
import webbrowser
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import whisper
from gtts import gTTS
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample_poly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


DATA_CSV = os.getenv(
    "ALEXA_DATA_CSV",
    str(Path(__file__).with_name("alexa_data.csv"))
)

st.set_page_config(page_title="Alexa — Lekhraj Patil", layout="centered")

st.title("🎙️ Alexa Voice Assistant")
st.markdown("**Built by Lekhraj Patil**")
st.caption("Record your voice → Whisper transcribes → Intent classified → Action taken → Response spoken")


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper():
    return whisper.load_model("tiny")


@st.cache_resource(show_spinner="Training intent classifier…")
def load_classifier():
    df = pd.read_csv(DATA_CSV)
    df["prompt_clean"] = df["prompt"].str.replace(r"\s+\d+$", "", regex=True)
    vec = TfidfVectorizer()
    X = vec.fit_transform(df["prompt_clean"])
    clf = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    clf.fit(X, df["intent"])
    return vec, clf


whisper_model   = load_whisper()
vectorizer, clf = load_classifier()


def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    sample_rate, raw = wav_read(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if raw.ndim > 1:
        raw = raw.mean(axis=1)

    if raw.dtype == np.int16:
        samples = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        samples = raw.astype(np.float32) / 2147483648.0
    else:
        samples = raw.astype(np.float32)

    target = 16000
    if sample_rate != target:
        g = gcd(sample_rate, target)
        samples = resample_poly(samples, target // g, sample_rate // g).astype(np.float32)

    result = whisper_model.transcribe(samples, fp16=False)
    return result["text"].strip().lower()


def classify(text: str) -> str:
    return clf.predict(vectorizer.transform([text]))[0]


def speak(text: str):
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )


def try_math(text: str):
    cleaned = re.sub(r"\s+\d+$", "", text.lower()
                     .replace("what is", "")
                     .replace("calculate", "")
                     .strip())
    nums = list(map(int, re.findall(r"\d+", cleaned)))
    if len(nums) != 2:
        return None
    a, b = nums
    if "plus"   in cleaned or "+" in cleaned: return f"The answer is {a + b}"
    if "minus"  in cleaned or "-" in cleaned: return f"The answer is {a - b}"
    if "times"  in cleaned or "multiply" in cleaned: return f"The answer is {a * b}"
    if "divide" in cleaned or "/" in cleaned:
        return f"The answer is {a / b:.2f}" if b != 0 else "Cannot divide by zero"
    return None


shopping_items: list[str] = []


def run_action(intent: str, prompt: str) -> str:
    text = prompt.lower()

    if intent == "play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Opening music on YouTube for you"

    if intent == "open_website":
        if "youtube" in text:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"
        if "google" in text:
            webbrowser.open("https://www.google.com")
            return "Opening Google"
        if "github" in text:
            webbrowser.open("https://github.com")
            return "Opening GitHub"
        return "Tell me which website you want to open"

    if intent == "play_radio":
        webbrowser.open("https://www.radio.com")
        return "Tuning into radio"

    if intent == "podcast_audiobook":
        webbrowser.open("https://open.spotify.com/search/podcasts")
        return "Opening podcasts on Spotify"

    if intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today's top headlines"

    if intent == "weather":
        match = re.search(r"(in|of|for)\s+([a-zA-Z\s]+)", text)
        city  = match.group(2).strip() if match else ""
        query = f"weather+in+{city.replace(' ', '+')}" if city else "weather+today"
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Showing weather for {city.title()}" if city else "Here is today's weather forecast"

    if intent == "date_time":
        now = datetime.datetime.now()
        return now.strftime("It is %I:%M %p on %B %d, %Y")

    if intent == "calculator":
        result = try_math(text)
        if result:
            return result
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Let me search that for you"

    if intent == "general_qa":
        if any(w in text for w in ("temperature", "weather", "forecast")):
            match = re.search(r"(in|of|for)\s+([a-zA-Z\s]+)", text)
            city  = match.group(2).strip() if match else ""
            query = f"weather+in+{city.replace(' ', '+')}" if city else "weather+today"
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Showing weather for {city.title()}" if city else "Here is the weather"
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Here is what I found on the web"

    if intent == "facts":
        facts = [
            "Honey never spoils — archaeologists found 3000 year old honey in Egyptian tombs.",
            "Bananas are technically berries, but strawberries are not.",
            "Octopuses have three hearts and blue blood.",
            "A group of flamingos is called a flamboyance.",
            "Wombats are the only animals that produce cube shaped droppings.",
        ]
        return random.choice(facts)

    if intent == "dictionary":
        webbrowser.open(f"https://www.dictionary.com/browse/{prompt.replace(' ', '%20')}")
        return "Opening the dictionary for you"

    if intent == "translation":
        webbrowser.open(f"https://translate.google.com/?sl=auto&tl=en&text={prompt.replace(' ', '%20')}")
        return "Opening Google Translate"

    if intent == "unit_conversion":
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Here is the unit conversion"

    if intent == "alarm":
        match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
        if match:
            h  = int(match.group(1))
            m  = int(match.group(2) or 0)
            ap = (match.group(3) or "am").upper()
            return f"Alarm set for {h}:{m:02d} {ap}"
        return "Alarm has been set"

    if intent == "set_timer":
        numbers = re.findall(r"\d+", text)
        mins = int(numbers[0]) if numbers else 1
        return f"Timer set for {mins} minute{'s' if mins != 1 else ''}"

    if intent == "reminder":
        return "Got it, I will remind you"

    if intent == "calendar":
        return "You have no upcoming events scheduled today"

    if intent == "shopping_list":
        item = re.sub(r"^(add|put|buy)\s+", "", text)
        item = re.sub(r"\s+(to\s+)?(my\s+)?(shopping\s+list)?\.?$", "", item).strip()
        shopping_items.append(item)
        return f"Added {item} to your shopping list"

    if intent == "routine":
        return "Your daily routine is ready"

    if intent == "todo_task":
        return "Task added to your to-do list"

    if intent == "smart_home":
        return "Smart home command sent"

    if intent == "device_status":
        return "All your devices are online"

    if intent == "traffic":
        webbrowser.open("https://www.google.com/maps/dir/Home/Work")
        return "Checking traffic on your route"

    if intent == "directions":
        destination = re.sub(r"^(directions?\s+to|navigate\s+to|way\s+to)\s+", "", text).strip()
        webbrowser.open(f"https://www.google.com/maps/search/{destination.replace(' ', '+')}")
        return f"Getting directions to {destination}"

    if intent == "personality":
        replies = [
            "I am Alexa, built by Lekhraj Patil. Always here to help!",
            "Doing great and ready to assist you.",
            "I was trained and deployed by Lekhraj. What can I do for you?",
        ]
        return random.choice(replies)

    if intent == "jokes_fun":
        jokes = [
            "Why did the computer catch a cold? Because it left its Windows open!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I asked my assistant for a joke. It said — loading humour module. Done. Here it is.",
        ]
        return random.choice(jokes)

    if intent == "games_trivia":
        return "Sure, let us play! What topic do you want — science, history or sports?"

    if intent == "movies":
        webbrowser.open("https://www.imdb.com/chart/top/")
        return "Here are some highly rated movies for you"

    return "Sorry, I could not understand that. Can you say it again?"


st.markdown("---")
st.markdown("### How to use")
st.markdown(
    "1. Click **Record** below and allow microphone access  \n"
    "2. Speak your command  \n"
    "3. Click **Stop** then **Submit**  \n"
    "4. Alexa will transcribe, understand and respond"
)
st.markdown("---")

audio_input = st.audio_input("🎤 Record your voice command")

if audio_input is not None:
    with st.spinner("Transcribing…"):
        transcript = transcribe(audio_input.getvalue())

    st.markdown(f"**You said:** {transcript}")

    intent = classify(transcript)
    st.markdown(f"**Intent detected:** `{intent}`")

    response = run_action(intent, transcript)
    st.success(f"🤖  Alexa says: {response}")

    with st.spinner("Generating audio response…"):
        speak(response)

st.markdown("---")
st.caption("Alexa Voice Assistant · Lekhraj Patil · Whisper + Scikit-learn + Streamlit")
