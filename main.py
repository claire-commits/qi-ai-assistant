from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


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


class Assistant:
    def __init__(self, name: str = "Qi") -> None:
        self.name = name
        self.memory = MemoryStore(Path(__file__).with_name("assistant_memory.json"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = self._build_client()

    def _build_client(self) -> OpenAI | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return None

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAI(api_key=api_key, base_url=base_url)

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

        if any(word in lower for word in ["date", "day"]):
            today = dt.date.today().strftime("%A, %d %B %Y")
            return f"Today is {today}."

        if "joke" in lower:
            jokes = [
                "Why did the developer go broke? Because they used up all their cache.",
                "I told my code to be funny, and it returned a syntax error.",
                "A computer and a life coach walked into a bar. The life coach said: 'Let's focus on your goals.'",
            ]
            return random.choice(jokes)

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

    def _api_response(self, prompt: str) -> str:
        if self.client is None:
            return (
                "I’m running in local-only mode right now. To enable a real LLM response, "
                "create a .env file and set OPENAI_API_KEY."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Qi, a helpful AI assistant. Keep responses concise, clear, and practical. "
                            "If the user asks about coding, give direct, beginner-friendly guidance."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            result = response.choices[0].message.content
            return result.strip() if result else "I’m here — I just didn’t get a useful reply back."
        except Exception as exc:
            return f"I hit an API error: {exc}. Please check your OPENAI_API_KEY and network connection."

    def respond(self, user_input: str) -> str:
        text = user_input.strip()
        if not text:
            return "I didn't catch that. Try asking for help or say hello."

        lower = text.lower()
        if lower in {"exit", "quit", "bye", "goodbye"}:
            return "Goodbye! I'll be here when you want to keep building."

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
            "day",
            "joke",
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

        return self._api_response(text)


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
    assistant = Assistant()
    print("Welcome to Qi AI Assistant!")
    print("Type 'exit' to quit. If OPENAI_API_KEY is set, the assistant will use the real API.")

    while True:
        user_input = get_user_input("You: ")
        response = assistant.respond(user_input)
        print(f"Assistant: {response}")

        if user_input.strip().lower() in {"exit", "quit", "bye", "goodbye"}:
            break


if __name__ == "__main__":
    main()
