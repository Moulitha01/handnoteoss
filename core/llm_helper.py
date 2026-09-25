"""Local open-source LLM helper (via Ollama) for OCR correction and note organizing.

Setup (once):
    1. Install Ollama from https://ollama.com
    2. ollama pull qwen2.5:3b        (or llama3.2:3b - both run OK on CPU)
Ollama then serves an API on http://localhost:11434. No extra pip packages needed.

If Ollama isn't running, every function degrades gracefully and the app keeps working.
"""
import difflib
import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def _post(payload, timeout=45):
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError):
        return None


def is_available():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Correction
# --------------------------------------------------------------------------- #
CORRECT_PROMPT = """You are fixing OCR output from handwritten notes.
Fix only obvious recognition or spelling mistakes, using the surrounding context.
Rules:
- Do NOT add, remove, reorder or paraphrase content.
- Keep the same number of lines and the same line breaks.
- If you are unsure about a word, leave it unchanged.
- Return ONLY the corrected text, nothing else.

OCR text:
{text}"""


def correct_text(raw):
    """Return {"original", "corrected", "changed", "used_llm"}.

    A safety check rejects the LLM output if it drifts too far from the OCR text,
    so the model can't quietly rewrite the note.
    """
    result = {"original": raw, "corrected": raw, "changed": False, "used_llm": False}
    if not raw or not raw.strip():
        return result

    out = _post({
        "model": OLLAMA_MODEL,
        "prompt": CORRECT_PROMPT.format(text=raw),
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": int(len(raw) / 2) + 60},
    })
    if not out:
        return result
    result["used_llm"] = True

    out = out.strip().strip('"').strip()
    similarity = difflib.SequenceMatcher(None, raw.lower(), out.lower()).ratio()
    too_long = len(out) > 1.4 * len(raw) + 20
    if similarity < 0.6 or too_long or not out:
        return result  # LLM rewrote too much - keep the original

    result["corrected"] = out
    result["changed"] = out != raw
    return result


# --------------------------------------------------------------------------- #
# Organizing: title, tags, summary
# --------------------------------------------------------------------------- #
ORGANIZE_PROMPT = """Read this note and produce metadata for organizing it.
Return ONLY valid JSON with exactly these keys:
{{"title": "<max 8 words>", "tags": ["<3 to 5 short lowercase tags>"], "summary": "<1-2 sentences>"}}

Note:
{text}"""


def organize_note(text):
    """Return {"title", "tags", "summary"}. Empty values if the LLM is unavailable."""
    empty = {"title": "", "tags": [], "summary": ""}
    if not text or not text.strip():
        return empty

    out = _post({
        "model": OLLAMA_MODEL,
        "prompt": ORGANIZE_PROMPT.format(text=text),
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_predict": 150},
    })
    if not out:
        return empty
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", out, re.S)
        if not m:
            return empty
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return empty

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return {
        "title": str(data.get("title", "")).strip()[:80],
        "tags": [str(t).strip().lower() for t in tags if str(t).strip()][:5],
        "summary": str(data.get("summary", "")).strip(),
    }