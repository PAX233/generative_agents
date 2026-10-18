"""
Download the local sentence-embedding model used instead of OpenAI's
text-embedding-ada-002 (which DeepSeek does not provide).

Model: sentence-transformers/all-MiniLM-L6-v2, exported to ONNX (384-d).

Usage:
    python scripts/download_embedding_model.py
    python scripts/download_embedding_model.py --mirror   # use hf-mirror.com

The files are written to <repo>/models/all-MiniLM-L6-v2 (gitignored).
"""
import argparse
import os
import sys

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(REPO_ROOT, "models", "all-MiniLM-L6-v2")

FILES = [
  ("onnx/model.onnx", "model.onnx"),
  ("tokenizer.json", "tokenizer.json"),
  ("tokenizer_config.json", "tokenizer_config.json"),
  ("special_tokens_map.json", "special_tokens_map.json"),
  ("config.json", "config.json"),
]

ENDPOINTS = ["https://huggingface.co",
             "https://hf-mirror.com"]


def download(url, dest, session):
  tmp = dest + ".part"
  with session.get(url, stream=True, timeout=(20, 300)) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length") or 0)
    done = 0
    with open(tmp, "wb") as f:
      for chunk in r.iter_content(chunk_size=1 << 20):
        if chunk:
          f.write(chunk)
          done += len(chunk)
          if total:
            pct = 100.0 * done / total
            sys.stdout.write(f"\r  {os.path.basename(dest)}: {pct:5.1f}% "
                             f"({done/1e6:.1f}/{total/1e6:.1f} MB)")
            sys.stdout.flush()
  sys.stdout.write("\n")
  os.replace(tmp, dest)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dest", default=DEFAULT_DEST)
  parser.add_argument("--mirror", action="store_true",
                      help="prefer hf-mirror.com (useful from mainland China)")
  parser.add_argument("--no-model", action="store_true",
                      help="skip the 90 MB onnx file (files only)")
  args = parser.parse_args()

  os.makedirs(args.dest, exist_ok=True)
  endpoints = list(reversed(ENDPOINTS)) if args.mirror else list(ENDPOINTS)
  session = requests.Session()

  for remote, local_name in FILES:
    if args.no_model and local_name == "model.onnx":
      continue
    dest = os.path.join(args.dest, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
      print(f"skip {local_name} (already present)")
      continue
    last_error = None
    for endpoint in endpoints:
      url = f"{endpoint}/sentence-transformers/all-MiniLM-L6-v2/resolve/main/{remote}"
      try:
        print(f"downloading {remote} from {endpoint}")
        download(url, dest, session)
        last_error = None
        break
      except Exception as exc:  # try the next endpoint
        last_error = exc
        print(f"  failed: {exc}")
    if last_error is not None:
      sys.exit(f"could not download {remote}: {last_error}")

  print(f"model ready in {args.dest}")


if __name__ == "__main__":
  main()
