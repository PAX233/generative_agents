"""
Headless stand-in for the browser frontend.

The upstream project expects an open browser tab at /simulator_home: the
Phaser page posts the persona tiles to Django (/process_environment) and polls
for the backend's movement plan (/update_environment) on every game step.
This script performs the same exchange without a browser, which makes
automated / remote runs possible.

Usage (from the repository root, with the Django server already running):

    python scripts/virtual_frontend.py --sim-code my-simulation --steps 30

The simulation itself (reverie/backend_server/reverie.py) is untouched: it
still receives exactly the environment / movement files it expects.
"""
import argparse
import json
import os
import sys
import time

import requests

DEFAULT_SERVER = "http://127.0.0.1:8000"


def load_json(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontend-root", default="environment/frontend_server",
                      help="path to environment/frontend_server")
  parser.add_argument("--sim-code", required=True,
                      help="name of the new simulation (same as in reverie.py)")
  parser.add_argument("--steps", type=int, required=True,
                      help="number of game steps to drive")
  parser.add_argument("--start-step", type=int, default=0)
  parser.add_argument("--server", default=DEFAULT_SERVER)
  parser.add_argument("--timeout", type=float, default=180.0,
                      help="seconds to wait for one backend response")
  parser.add_argument("--maze", default="the_ville")
  args = parser.parse_args()

  sim_dir = os.path.join(args.frontend_root, "storage", args.sim_code)
  env_path = os.path.join(sim_dir, "environment", f"{args.start_step}.json")
  if not os.path.exists(env_path):
    sys.exit(f"environment file not found: {env_path}")

  positions = {}
  for name, value in load_json(env_path).items():
    positions[name] = [int(value["x"]), int(value["y"])]

  print(f"[frontend] sim={args.sim_code} step={args.start_step} "
        f"personas={list(positions)}")

  step = args.start_step
  end_step = args.start_step + args.steps
  while step < end_step:
    payload = {
      "step": step,
      "sim_code": args.sim_code,
      "environment": {
        name: {"maze": args.maze, "x": pos[0], "y": pos[1]}
        for name, pos in positions.items()
      },
    }
    try:
      requests.post(f"{args.server}/process_environment/", json=payload,
                    timeout=30)
    except requests.RequestException as exc:
      sys.exit(f"[frontend] cannot reach Django at {args.server}: {exc}")

    movement_path = os.path.join(sim_dir, "movement", f"{step}.json")
    deadline = time.time() + args.timeout
    movement = None
    while time.time() < deadline:
      if os.path.exists(movement_path):
        try:
          movement = load_json(movement_path)
          break
        except (json.JSONDecodeError, OSError):
          pass
      time.sleep(0.2)

    if movement is None:
      sys.exit(f"[frontend] backend did not produce {movement_path} "
               f"within {args.timeout:.0f}s")

    for name, data in movement.get("persona", {}).items():
      positions[name] = list(data["movement"])

    curr_time = movement.get("meta", {}).get("curr_time", "?")
    print(f"[frontend] step {step} -> {curr_time} "
          f"{ {n: p for n, p in positions.items()} }", flush=True)
    step += 1

  print("[frontend] done")


if __name__ == "__main__":
  main()
