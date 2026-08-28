# qi-ai-assistant

My first AI-powered assistant project.

## What it does

This assistant can:

- greet the user
- answer time/date questions
- share short jokes
- save notes locally in a JSON file
- use a real OpenAI-compatible API when `OPENAI_API_KEY` is present

## Step 1: create your virtual environment

```bash
cd qi-ai-assistant
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Step 2: install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: configure Ollama

Copy the example file and fill in your real key:

```bash
copy .env.example .env
```

Or on macOS/Linux:

```bash
cp .env.example .env
```

Update `.env` for your local Ollama model:

```env
OPENAI_API_KEY=local
OPENAI_MODEL=llama3.1:8b
OPENAI_BASE_URL=http://localhost:11434/v1
```

Use a model installed on your computer. Check with `ollama list`, or download one with:

```powershell
ollama pull llama3.1:8b
```

## Step 4: run the app

```bash
python main.py
```

Running `python main.py` opens the desktop chat UI. Use `python main.py --cli` for the terminal version.

You can say things like:

- hello
- what can you do
- tell me a joke
- remember my project deadline is Friday
- show notes
- how do I build a Python app?

## Step 5: use the local model

Once Ollama is running and the model is installed, the assistant sends messages to your local model. No cloud API key is required.

If you do not provide a key, the assistant still works in local mode for simple built-in commands.

## Notes

- The local note memory is stored in `assistant_memory.json`.
- The repo ignores runtime data with `.gitignore`.
