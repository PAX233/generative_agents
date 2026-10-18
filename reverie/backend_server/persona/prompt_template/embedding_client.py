"""
Local text-embedding backend for generative agents.

DeepSeek's public API only serves chat completions (there is no
text-embedding endpoint), while the original project relied on OpenAI's
text-embedding-ada-002. To keep the same retrieval behaviour without an
external embedding service, this module runs a small sentence-transformer
( all-MiniLM-L6-v2, 384-d, exported to ONNX) locally with onnxruntime.

If the ONNX model is missing, a deterministic feature-hashing embedder is
used instead so that the simulation still runs (lower semantic quality).
"""
import hashlib
import os
import re
import threading

import numpy as np

try:
  import utils as _utils
except Exception:  # pragma: no cover - utils.py intentionally gitignored
  _utils = None


def _cfg(name, default):
  if _utils is None:
    return default
  return getattr(_utils, name, default)


_TOKENIZER = None
_SESSION = None
_DIM = None
_BACKEND = None
_CACHE = {}
_LOCK = threading.Lock()

_HASH_DIM = 384


def _model_dir():
  d = _cfg("embedding_model_dir", None)
  if d:
    return os.path.abspath(d)
  here = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(os.path.join(here, "..", "..", "..", "..",
                                      "models", "all-MiniLM-L6-v2"))


def _init_onnx():
  global _TOKENIZER, _SESSION, _DIM, _BACKEND
  from tokenizers import Tokenizer
  import onnxruntime as ort

  model_dir = _model_dir()
  onnx_path = os.path.join(model_dir, "model.onnx")
  tok_path = os.path.join(model_dir, "tokenizer.json")
  if not (os.path.exists(onnx_path) and os.path.exists(tok_path)):
    raise FileNotFoundError(f"embedding model not found in {model_dir}")

  tokenizer = Tokenizer.from_file(tok_path)
  tokenizer.enable_truncation(max_length=int(_cfg("embedding_max_tokens", 256)))
  tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

  sess_options = ort.SessionOptions()
  sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  sess_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
  session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                 providers=["CPUExecutionProvider"])
  out_shape = session.get_outputs()[0].shape
  dim = out_shape[-1] if isinstance(out_shape[-1], int) else None

  _TOKENIZER = tokenizer
  _SESSION = session
  _DIM = dim or 384
  _BACKEND = "onnx:all-MiniLM-L6-v2"


def _init_backend():
  global _BACKEND, _DIM
  if _BACKEND is not None:
    return
  backend = str(_cfg("embedding_backend", "onnx")).lower()
  if backend == "onnx":
    try:
      _init_onnx()
      return
    except Exception as exc:  # pragma: no cover - depends on local files
      print(f"[embedding] ONNX model unavailable ({exc}); "
            f"falling back to feature hashing")
  _BACKEND = "hash"
  _DIM = _HASH_DIM


def embedding_dim():
  """Dimension of the vectors produced by the configured backend."""
  _init_backend()
  return _DIM


def backend_name():
  _init_backend()
  return _BACKEND


def _hash_vector(text):
  tokens = re.findall(r"[a-z0-9']+", text.lower())
  vec = np.zeros(_HASH_DIM, dtype=np.float32)
  for tok in tokens:
    h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
    sign = 1.0 if (h >> 8) & 1 else -1.0
    vec[h % _HASH_DIM] += sign
  norm = np.linalg.norm(vec)
  if norm > 0:
    vec = vec / norm
  return vec


def _onnx_forward(texts):
  encodings = _TOKENIZER.encode_batch(texts)
  input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
  attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)
  feeds = {}
  for inp in _SESSION.get_inputs():
    if inp.name == "input_ids":
      feeds[inp.name] = input_ids
    elif inp.name == "attention_mask":
      feeds[inp.name] = attention
    elif inp.name == "token_type_ids":
      feeds[inp.name] = np.zeros_like(input_ids)
    else:  # unknown input: best effort
      feeds[inp.name] = input_ids
  outputs = _SESSION.run(None, feeds)
  hidden = outputs[0]  # (batch, seq, dim)
  mask = attention[..., None].astype(np.float32)
  summed = (hidden * mask).sum(axis=1)
  counts = np.clip(mask.sum(axis=1), 1e-9, None)
  vecs = summed / counts
  norms = np.linalg.norm(vecs, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  return vecs / norms


def embed_batch(texts):
  """Embed a list of strings, returns a list of python float lists."""
  _init_backend()
  clean = []
  for t in texts:
    t = str(t).replace("\n", " ").strip()
    clean.append(t if t else "this is blank")

  missing = [t for t in dict.fromkeys(clean) if t not in _CACHE]
  if missing:
    if _BACKEND.startswith("onnx"):
      with _LOCK:
        for start in range(0, len(missing), 64):
          chunk = missing[start:start + 64]
          try:
            vecs = _onnx_forward(chunk)
          except Exception as exc:
            print(f"[embedding] ONNX inference failed ({exc}); using hashing")
            vecs = np.stack([_hash_vector(t) for t in chunk])
          for t, v in zip(chunk, vecs):
            _CACHE[t] = [round(float(x), 7) for x in v]
    else:
      for t in missing:
        _CACHE[t] = [round(float(x), 7) for x in _hash_vector(t)]

  return [_CACHE[t] for t in clean]


def embed(text):
  """Embed a single string, returns a python float list."""
  return embed_batch([text])[0]


def cache_size():
  return len(_CACHE)
