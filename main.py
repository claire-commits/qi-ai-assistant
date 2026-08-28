from __future__ import annotations

import datetime as dt
import ast
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import requests


load_dotenv(Path(__file__).with_name(".env"))


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.notes: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.notes = [str(item) for item in data]
            except (json.JSONDecodeError, OSError):
                self.notes = []

    def add(self, text: str) -> str:
        note = text.strip()
        if not note:
            return "I could not save an empty note."
        self.notes.append(note)
        self.path.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
        return f"Saved: \"{note}\""

    def list_notes(self) -> str:
        if not self.notes:
            return "I do not have any saved notes yet."
        return "Saved notes:\n- " + "\n- ".join(self.notes)


class ProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.facts: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.facts = {str(key): str(value) for key, value in data.items()}
            except (json.JSONDecodeError, OSError):
                self.facts = {}

    def set_fact(self, key: str, value: str) -> str:
        self.facts[key] = value.strip()
        self.path.write_text(json.dumps(self.facts, indent=2), encoding="utf-8")
        return f"I will remember that your {key} is {value.strip()}."

    def summary(self) -> str:
        if not self.facts:
            return "I do not know any personal details about you yet."
        return "What I know about you:\n" + "\n".join(
            f"- {key}: {value}" for key, value in self.facts.items()
        )

    def context(self) -> str:
        if not self.facts:
            return "No personal profile details have been saved."
        return "\n".join(f"- {key}: {value}" for key, value in self.facts.items())

    def forget(self) -> str:
        self.facts.clear()
        self.path.write_text("{}", encoding="utf-8")
        return "I have forgotten your saved personal details."


class ReminderStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.reminders: list[dict[str, str | bool]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.reminders = [item for item in data if isinstance(item, dict)]
            except (json.JSONDecodeError, OSError):
                self.reminders = []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.reminders, indent=2), encoding="utf-8")

    def add(self, text: str, due_at: dt.datetime) -> str:
        self.reminders.append({
            "text": text.strip(),
            "due_at": due_at.isoformat(timespec="minutes"),
            "done": False,
        })
        self._save()
        return f"Reminder saved for {due_at:%A, %d %B at %H:%M}: {text.strip()}"

    def list_pending(self) -> str:
        pending = [item for item in self.reminders if not item.get("done", False)]
        if not pending:
            return "You do not have any pending reminders."
        lines = []
        for item in pending:
            due_at = dt.datetime.fromisoformat(str(item["due_at"]))
            lines.append(f"- {due_at:%d %b %Y %H:%M}: {item['text']}")
        return "Pending reminders:\n" + "\n".join(lines)

    def due(self) -> list[str]:
        now = dt.datetime.now()
        due_text = []
        changed = False
        for item in self.reminders:
            if item.get("done", False):
                continue
            if dt.datetime.fromisoformat(str(item["due_at"])) <= now:
                due_text.append(str(item["text"]))
                item["done"] = True
                changed = True
        if changed:
            self._save()
        return due_text


class Assistant:
    def __init__(self, name: str | None = None) -> None:
        self.name = name or os.getenv("QI_NAME", "Qi")
        self.voice_enabled = os.getenv("QI_VOICE_ENABLED", "false").lower() == "true"
        self.memory = MemoryStore(Path(__file__).with_name("assistant_memory.json"))
        self.profile = ProfileStore(Path(__file__).with_name("assistant_profile.json"))
        self.reminders = ReminderStore(Path(__file__).with_name("assistant_reminders.json"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = self._build_client()
        self._speech_engine = None
        self._speech_lock = threading.Lock()

    def _build_client(self) -> OpenAI | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return None

        return OpenAI(api_key=api_key, base_url=self.base_url)

    def speak(self, text: str) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            with self._speech_lock:
                self._speech_engine = engine
            preferred_voice = os.getenv("QI_VOICE_NAME", "Scottish").lower()
            fallback_voice = None
            for voice in engine.getProperty("voices"):
                voice_details = f"{voice.id} {voice.name}".lower()
                if preferred_voice in voice_details or "scotland" in voice_details:
                    engine.setProperty("voice", voice.id)
                    break
                if fallback_voice is None and "great britain" in voice_details:
                    fallback_voice = voice.id
            else:
                if fallback_voice:
                    engine.setProperty("voice", fallback_voice)
            segments = [segment.strip() for segment in text.splitlines() if segment.strip()]
            for index, segment in enumerate(segments):
                engine.say(segment)
                engine.runAndWait()
                if index < len(segments) - 1:
                    time.sleep(1)
        except Exception:
            pass
        finally:
            with self._speech_lock:
                self._speech_engine = None

    def stop_speaking(self) -> None:
        with self._speech_lock:
            engine = self._speech_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speech_text(self, prompt: str, response: str) -> str:
        if "news" in prompt.lower() or "headlines" in prompt.lower():
            response = "\n".join(
                re.sub(r"^\s*[-*]\s*", "", line)
                for line in response.splitlines()[1:]
            )
        return re.sub(r"https?://\S+", "", response).strip()

    def _calculate(self, expression: str) -> str:
        allowed_operators = {
            ast.Add: lambda left, right: left + right,
            ast.Sub: lambda left, right: left - right,
            ast.Mult: lambda left, right: left * right,
            ast.Div: lambda left, right: left / right,
            ast.Pow: lambda left, right: left ** right,
            ast.Mod: lambda left, right: left % right,
            ast.USub: lambda value: -value,
        }

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](evaluate(node.operand))
            raise ValueError

        try:
            result = evaluate(ast.parse(expression, mode="eval").body)
            return f"{expression} = {result:g}"
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            return "I could not calculate that. Try something like: calculate 12 * (3 + 4)."

    def _weather_response(self, prompt: str) -> str:
        match = re.search(r"\b(?:in|for|near)\s+(.+?)(?:\s+today|\s+now)?[?.!]*$", prompt, re.IGNORECASE)
        city = match.group(1).strip() if match else "Aberdeen"
        try:
            location = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            ).json()
            results = location.get("results", [])
            if not results:
                return f"I could not find a location called {city}."

            place = results[0]
            current = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                },
                timeout=10,
            ).json()["current"]
            descriptions = {
                0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
                55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
                71: "light snow", 73: "snow", 75: "heavy snow", 80: "light showers",
                81: "showers", 82: "heavy showers", 95: "thunderstorms",
            }
            condition = descriptions.get(current["weather_code"], "mixed conditions")
            return (
                f"In {place['name']} today: {current['temperature_2m']}°C, feels like "
                f"{current['apparent_temperature']}°C with {condition}. "
                f"Wind is {current['wind_speed_10m']} km/h."
            )
        except (requests.RequestException, KeyError, TypeError, ValueError):
            return "I could not reach the live weather service right now."

    def _news_response(self, local: bool = False) -> str:
        feed_url = (
            "https://news.google.com/rss/search?q=Aberdeen+Scotland&hl=en-GB&gl=GB&ceid=GB:en"
            if local
            else "https://feeds.bbci.co.uk/news/world/rss.xml"
        )
        label = "local Aberdeen and North East Scotland" if local else "global"
        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            headlines = []
            for item in root.findall("./channel/item")[:5]:
                title = item.findtext("title")
                if title:
                    headlines.append(f"- {title}")
            if not headlines:
                return f"I could not find any {label} headlines right now."
            return f"Latest {label} headlines:\n" + "\n".join(headlines)
        except (requests.RequestException, ET.ParseError):
            return f"I could not reach the {label} news feed right now."

    def _reminder_response(self, prompt: str) -> str:
        match = re.match(r"remind me to (.+?) at (.+)$", prompt, re.IGNORECASE)
        if not match:
            return "Use this format: remind me to call Mum at 2026-08-29 18:30"
        text, due_text = match.groups()
        try:
            due_at = dt.datetime.fromisoformat(due_text.strip())
        except ValueError:
            return "I could not understand that date. Use YYYY-MM-DD HH:MM."
        return self.reminders.add(text, due_at)

    def _profile_response(self, prompt: str) -> str:
        match = re.match(r"my (name|location|city|job|work|favourite colour|favorite color) is (.+)$", prompt, re.IGNORECASE)
        if match:
            key, value = match.groups()
            return self.profile.set_fact(key.lower(), value)
        match = re.match(r"i live in (.+)$", prompt, re.IGNORECASE)
        if match:
            return self.profile.set_fact("location", match.group(1))
        match = re.match(r"i work as (.+)$", prompt, re.IGNORECASE)
        if match:
            return self.profile.set_fact("job", match.group(1))
        if prompt.lower() in {"what do you know about me", "what do you remember about me", "my profile"}:
            return self.profile.summary()
        if prompt.lower() in {"forget me", "forget what you know about me", "clear my profile"}:
            return self.profile.forget()
        return "Tell me a fact like: my name is Claire, or I live in Aberdeen."

    def _local_response(self, user_input: str) -> str:
        text = user_input.strip()
        lower = text.lower()

        if any(word in lower for word in ["hello", "hi", "hey"]):
            return f"Hello! I'm {self.name}, your first AI assistant. How can I help today?"

        if any(word in lower for word in ["what can you do", "help", "commands"]):
            return (
                "I can greet you, tell the time, share a joke, help with planning, "
                "and save simple notes for you."
            )

        if any(word in lower for word in ["time", "clock"]):
            now = dt.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."

        if "date" in lower or lower == "day" or "what day" in lower:
            today = dt.date.today().strftime("%A, %d %B %Y")
            return f"Today is {today}."

        if "joke" in lower:
            jokes = [
                "Why did the developer go broke? Because they used up all their cache.",
                "I told my code to be funny, and it returned a syntax error.",
                "A computer and a life coach walked into a bar. The life coach said: 'Let's focus on your goals.'",
            ]
            return random.choice(jokes)

        if lower.startswith("calculate "):
            return self._calculate(text[10:].strip())

        if lower.startswith("remind me to "):
            return self._reminder_response(text)

        if (lower.startswith("my ") and " is " in lower) or lower.startswith("i live in ") or lower.startswith("i work as "):
            return self._profile_response(text)

        if lower in {"what do you know about me", "what do you remember about me", "my profile", "forget me", "forget what you know about me", "clear my profile"}:
            return self._profile_response(text)

        if lower in {"reminders", "show reminders", "my reminders"}:
            return self.reminders.list_pending()

        if lower.startswith("remember "):
            return self.memory.add(text[9:].strip())

        if lower.startswith("save "):
            return self.memory.add(text[5:].strip())

        if lower == "notes" or lower.startswith("show notes"):
            return self.memory.list_notes()

        if any(word in lower for word in ["plan", "task", "goal", "project"]):
            return (
                "A simple plan is: define the goal, make one small win, test it, and then improve it. "
                "Your first AI assistant project is a great example."
            )

        if "thanks" in lower or "thank you" in lower:
            return "You're welcome! I'm happy to help you build your project."

        if "name" in lower:
            return f"My name is {self.name}. I am a small starter assistant built in Python."

        return (
            "I'm still learning, but I can help with quick questions, small tasks, and notes. "
            "Try asking for the time, a joke, or help."
        )

    def _api_response(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        if self.client is None:
            return (
                "No AI provider is configured. Set OPENAI_API_KEY in .env and restart the app."
            )

        try:
            messages = [{
                "role": "system",
                "content": (
                    f"You are {self.name}, a helpful AI assistant. Keep responses concise, clear, and practical. "
                    "If the user asks about coding, give direct, beginner-friendly guidance. "
                    "Use these user-provided personal details when relevant, and do not invent any others:\n"
                    f"{self.profile.context()}"
                ),
            }]
            if history:
                messages.extend(history[-12:])
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            result = response.choices[0].message.content
            return result.strip() if result else "I’m here — I just didn’t get a useful reply back."
        except Exception as exc:
            return f"I could not reach {self.base_url}: {exc}. Make sure Ollama is running."

    def respond(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        text = user_input.strip()
        if not text:
            return "I didn't catch that. Try asking for help or say hello."

        lower = text.lower()
        if lower in {"exit", "quit", "bye", "goodbye"}:
            return "Goodbye! I'll be here when you want to keep building."

        if "weather" in lower:
            return self._weather_response(text)

        if "news" in lower or "headlines" in lower:
            return self._news_response(
                local="local" in lower or "aberdeen" in lower or "scotland" in lower
            )

        if lower.startswith("remind me to ") or lower in {"reminders", "show reminders", "my reminders"}:
            return self._local_response(text)

        local_commands = [
            "remember ",
            "save ",
            "notes",
            "show notes",
            "hello",
            "hi",
            "hey",
            "what can you do",
            "help",
            "commands",
            "time",
            "clock",
            "date",
            "what day",
            "joke",
            "calculate ",
            "remind me to ",
            "reminders",
            "show reminders",
            "my reminders",
            "plan",
            "task",
            "goal",
            "project",
            "thanks",
            "thank you",
            "name",
        ]

        if any(word in lower for word in local_commands):
            return self._local_response(text)

        return self._api_response(text, history)


def run_gui() -> None:
    import tkinter as tk
    from tkinter import scrolledtext, ttk

    assistant = Assistant()
    root = tk.Tk()
    root.title(f"{assistant.name} AI Assistant")
    root.geometry("900x680")
    root.minsize(640, 480)
    root.configure(bg="#0b1220")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("App.TFrame", background="#0b1220")
    style.configure(
        "Header.TLabel",
        background="#0b1220",
        foreground="#f8fafc",
        font=("Segoe UI Semibold", 20),
    )
    style.configure(
        "Muted.TLabel",
        background="#0b1220",
        foreground="#8fa3bf",
        font=("Segoe UI", 9),
    )
    style.configure(
        "Send.TButton",
        background="#3b82f6",
        foreground="white",
        padding=(18, 9),
        font=("Segoe UI Semibold", 10),
    )
    style.configure(
        "Voice.TCheckbutton",
        background="#0b1220",
        foreground="#c7d7ee",
        padding=(18, 9),
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "Voice.TCheckbutton",
        background=[("active", "#0b1220"), ("pressed", "#0b1220")],
        foreground=[("active", "#c7d7ee"), ("pressed", "#c7d7ee")],
    )

    history: list[dict[str, str]] = []
    busy = False
    container = ttk.Frame(root, style="App.TFrame", padding=18)
    container.pack(fill="both", expand=True)
    header = ttk.Frame(container, style="App.TFrame")
    header.pack(fill="x", pady=(0, 12))
    ttk.Label(header, text=assistant.name, style="Header.TLabel").pack(side="left")
    status = ttk.Label(header, text=f"Local Ollama • {assistant.model}",
                       style="Muted.TLabel")
    status.pack(side="right", pady=6)

    chat = scrolledtext.ScrolledText(
        container, wrap=tk.WORD, state="disabled", bg="#111d30", fg="#e2e8f0",
        insertbackground="white", relief="flat", padx=18, pady=16,
        font=("Segoe UI", 11),
    )
    chat.pack(fill="both", expand=True)
    chat.tag_configure("you", foreground="#93c5fd", font=("Segoe UI", 11, "bold"))
    chat.tag_configure("qi", foreground="#86efac", font=("Segoe UI", 11, "bold"))

    def add_message(sender: str, text: str, tag: str) -> None:
        chat.configure(state="normal")
        chat.insert(tk.END, f"{sender}\n", tag)
        chat.insert(tk.END, f"{text}\n\n")
        chat.configure(state="disabled")
        chat.see(tk.END)

    add_message(assistant.name, f"Hello! I’m connected to your local Ollama model. How can I help?", "qi")
    composer = ttk.Frame(container, style="App.TFrame")
    composer.pack(fill="x", pady=(12, 0))
    speak_var = tk.BooleanVar(value=assistant.voice_enabled)
    ttk.Checkbutton(
        composer,
        text="Speak replies",
        variable=speak_var,
        style="Voice.TCheckbutton",
    ).pack(side="right", padx=(10, 0))
    entry = ttk.Entry(composer, font=("Segoe UI", 11))
    entry.pack(side="left", fill="x", expand=True, ipady=7)
    send = ttk.Button(composer, text="Send", style="Send.TButton")
    send.pack(side="left", padx=(10, 0))
    stop = ttk.Button(composer, text="Stop Morris", command=assistant.stop_speaking)
    stop.pack(side="left", padx=(8, 0))

    def set_busy(value: bool) -> None:
        nonlocal busy
        busy = value
        state = "disabled" if value else "normal"
        entry.configure(state=state)
        send.configure(state=state)
        status.configure(text=f"Thinking with {assistant.model}..." if value
                          else f"Local Ollama • {assistant.model}")

    def complete_response(user_text: str, response: str) -> None:
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ])
        add_message(assistant.name, response, "qi")
        set_busy(False)
        entry.focus_set()

    def send_message(_event: object | None = None) -> str:
        if busy:
            return "break"
        user_text = entry.get().strip()
        if not user_text:
            return "break"
        entry.delete(0, tk.END)
        add_message("You", user_text, "you")
        set_busy(True)
        speak_response = speak_var.get()

        def work() -> None:
            response = assistant.respond(user_text, history)
            if speak_response:
                assistant.speak(assistant.speech_text(user_text, response))
            root.after(0, complete_response, user_text, response)

        threading.Thread(target=work, daemon=True).start()
        return "break"

    def check_reminders() -> None:
        for reminder in assistant.reminders.due():
            message = f"Reminder: {reminder}"
            add_message(assistant.name, message, "qi")
            if speak_var.get():
                threading.Thread(target=assistant.speak, args=(message,), daemon=True).start()
        root.after(30_000, check_reminders)

    send.configure(command=send_message)
    entry.bind("<Return>", send_message)
    entry.focus_set()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.after(1_000, check_reminders)
    root.mainloop()


def get_user_input(prompt: str) -> str:
    if sys.stdin.isatty():
        return input(prompt)

    try:
        line = sys.stdin.readline()
        if not line:
            return "exit"
        return line.rstrip("\r\n")
    except EOFError:
        return "exit"


def main() -> None:
    if "--gui" in sys.argv or ("--cli" not in sys.argv and sys.stdin.isatty()):
        run_gui()
        return
    assistant = Assistant()
    print("Welcome to Qi AI Assistant!")
    print(f"Connected to {assistant.base_url} using {assistant.model}. Type 'exit' to quit.")

    while True:
        user_input = get_user_input("You: ")
        response = assistant.respond(user_input)
        print(f"Assistant: {response}")

        if user_input.strip().lower() in {"exit", "quit", "bye", "goodbye"}:
            break


if __name__ == "__main__":
    main()
