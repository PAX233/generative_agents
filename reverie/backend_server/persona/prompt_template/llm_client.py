"""
DeepSeek client (OpenAI-compatible chat completions) for generative agents.

gpt_structure.py uses this module instead of the OpenAI SDK. Every request is
logged to llm_usage.jsonl (prompt/completion tokens and an estimated cost) so
the cost of a simulation run can be audited afterwards.

Configuration lives in utils.py (gitignored) or the DEEPSEEK_API_KEY
environment variable.
"""
import datetime
import json
import os
import time

import requests

try:
  import utils as _utils
except Exception:  # pragma: no cover - utils.py is intentionally gitignored
  _utils = None


def _cfg(name, default):
  if _utils is None:
    return default
  value = getattr(_utils, name, None)
  return default if value in (None, "") else value


def api_key():
  key = os.environ.get("DEEPSEEK_API_KEY", "")
  if key:
    return key.strip()
  return str(_cfg("deepseek_api_key", "")).strip()


def _base_url():
  return str(_cfg("llm_base_url", "https://api.deepseek.com")).rstrip("/")


def _model():
  return str(_cfg("llm_model", "deepseek-chat"))


def _model_large():
  return str(_cfg("llm_model_large", _model()))


def _usage_log_path():
  return str(_cfg("llm_usage_log", "llm_usage.jsonl"))


_TOTALS = {
  "calls": 0,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "cache_hit_tokens": 0,
  "cache_miss_tokens": 0,
  "estimated_cost_cny": 0.0,
  "elapsed_seconds": 0.0,
}


def _price(tokens, per_million):
  return tokens / 1_000_000.0 * float(per_million)


def _record_usage(model, usage, tag, elapsed):
  prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
  completion_tokens = int(usage.get("completion_tokens", 0) or 0)
  cache_hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
  cache_miss = int(usage.get("prompt_cache_miss_tokens", 0) or
                   max(prompt_tokens - cache_hit, 0))
  cost = (_price(cache_hit, _cfg("llm_price_input_cached", 0.5)) +
          _price(cache_miss, _cfg("llm_price_input", 2.0)) +
          _price(completion_tokens, _cfg("llm_price_output", 8.0)))

  _TOTALS["calls"] += 1
  _TOTALS["prompt_tokens"] += prompt_tokens
  _TOTALS["completion_tokens"] += completion_tokens
  _TOTALS["cache_hit_tokens"] += cache_hit
  _TOTALS["cache_miss_tokens"] += cache_miss
  _TOTALS["estimated_cost_cny"] += cost
  _TOTALS["elapsed_seconds"] += elapsed

  record = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "tag": tag,
    "model": model,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "prompt_cache_hit_tokens": cache_hit,
    "prompt_cache_miss_tokens": cache_miss,
    "estimated_cost_cny": round(cost, 6),
    "elapsed_seconds": round(elapsed, 3),
  }
  try:
    with open(_usage_log_path(), "a", encoding="utf-8") as f:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")
  except OSError:
    pass

  if tag:
    print(f"[llm] {tag:<28} in={prompt_tokens:<6} out={completion_tokens:<5} "
          f"{elapsed:5.1f}s  total=CNY{_TOTALS['estimated_cost_cny']:.3f}")


CONTINUATION_SUFFIX = (
  "\n\nContinue the text above exactly where it stops. Reply with only the "
  "continuation itself: do not repeat the preceding text, do not add quotes, "
  "labels or explanations.")


def _log_raw(tag, prompt, response, payload):
  """Optional debugging dump: GA_LLM_LOG=<path>."""
  path = os.environ.get("GA_LLM_LOG", "").strip()
  if not path:
    return
  try:
    with open(path, "a", encoding="utf-8") as f:
      f.write(json.dumps({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "prompt": prompt,
        "response": response,
        "payload": {k: payload[k] for k in ("model", "max_tokens",
                                            "temperature", "stop")
                    if k in payload},
      }, ensure_ascii=False) + "\n")
  except OSError:
    pass


def _clean_stop(stop):
  if not stop:
    return None
  # Note: "\n" is a valid stop sequence for DeepSeek, and the upstream
  # prompts rely on it heavily, so only empty strings are dropped here.
  seqs = [s for s in stop if isinstance(s, str) and s != ""]
  return seqs or None


def chat(prompt,
         model=None,
         max_tokens=1024,
         temperature=0.0,
         top_p=1.0,
         stop=None,
         frequency_penalty=0.0,
         presence_penalty=0.0,
         tag=None,
         continuation=False):
  """Send one prompt as a single user message and return the text reply."""
  key = api_key()
  if not key:
    raise RuntimeError(
      "DEEPSEEK_API_KEY is empty. Put your key in reverie/backend_server/utils.py "
      "or export DEEPSEEK_API_KEY before running the simulation.")

  model = model or _model()
  if continuation:
    prompt = prompt.rstrip() + CONTINUATION_SUFFIX
  payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(min(max(int(max_tokens), 1), 8192)),
    "temperature": float(temperature),
    "top_p": float(top_p),
    "stream": False,
  }
  stop_seqs = _clean_stop(stop)
  if stop_seqs:
    payload["stop"] = stop_seqs
  if frequency_penalty:
    payload["frequency_penalty"] = float(frequency_penalty)
  if presence_penalty:
    payload["presence_penalty"] = float(presence_penalty)

  url = f"{_base_url()}/chat/completions"
  headers = {"Authorization": f"Bearer {key}",
             "Content-Type": "application/json"}
  timeout = float(_cfg("llm_timeout", 120))
  retries = int(_cfg("llm_max_retries", 6))
  started = time.time()

  last_error = None
  for attempt in range(retries):
    attempt_started = time.time()
    try:
      resp = requests.post(url, headers=headers, json=payload,
                           timeout=(15, timeout))
    except requests.RequestException as exc:
      last_error = f"network error: {exc}"
    else:
      if resp.status_code == 200:
        data = resp.json()
        usage = data.get("usage") or {}
        _record_usage(model, usage, tag, time.time() - attempt_started)
        choices = data.get("choices") or []
        if not choices:
          raise RuntimeError(f"empty choices in response: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        _log_raw(tag, prompt, content, payload)
        return content
      body = resp.text[:500]
      last_error = f"HTTP {resp.status_code}: {body}"
      if resp.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {body}")

    sleep_for = min(60.0, 2.0 * (2 ** attempt)) * (0.75 + 0.5 * (attempt % 2))
    print(f"[llm] retry {attempt + 1}/{retries} in {sleep_for:.1f}s "
          f"({last_error})")
    time.sleep(sleep_for)

  raise RuntimeError(f"DeepSeek API failed after {retries} attempts: "
                     f"{last_error} (elapsed {time.time() - started:.0f}s)")


def chat_large(prompt, **kwargs):
  kwargs.setdefault("model", _model_large())
  return chat(prompt, **kwargs)


def totals():
  return dict(_TOTALS)


def summary():
  t = totals()
  return (f"LLM calls: {t['calls']} | input: {t['prompt_tokens']} tok "
          f"(cache hit {t['cache_hit_tokens']}) | output: "
          f"{t['completion_tokens']} tok | approx CNY "
          f"{t['estimated_cost_cny']:.3f} | {t['elapsed_seconds']:.0f}s")
