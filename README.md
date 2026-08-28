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

## Step 3: add your API key

Copy the example file and fill in your real key:

```bash
copy .env.example .env
```

Or on macOS/Linux:

```bash
cp .env.example .env
```

Then update `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

You can also point `OPENAI_BASE_URL` to a compatible provider, such as OpenRouter or a local OpenAI-compatible server.

## Step 4: run the app

```bash
python main.py
```

You can say things like:

- hello
- what can you do
- tell me a joke
- remember my project deadline is Friday
- show notes
- how do I build a Python app?

## Step 5: use the real API

Once the key is configured, the assistant will send your message to the model and return a real AI response instead of the built-in local fallback.

If you do not provide a key, the assistant still works in local mode for simple built-in commands.

## Notes

- The local note memory is stored in `assistant_memory.json`.
- The repo ignores runtime data with `.gitignore`.
