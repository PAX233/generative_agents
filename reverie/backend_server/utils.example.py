"""
Copy this file to utils.py (same folder) and fill in your own settings.

utils.py is listed in .gitignore and must never be committed: it holds the
API key. The repository only tracks this example file.

The DeepSeek key can also be supplied through the DEEPSEEK_API_KEY
environment variable, which takes precedence over the value below.
"""
import os

# ---------------------------------------------------------------------------
# LLM (DeepSeek, OpenAI-compatible API)
# ---------------------------------------------------------------------------
# Reads DEEPSEEK_API_KEY first; paste your key below only if you prefer a file.
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
key_owner = "<Your Name>"

llm_base_url = "https://api.deepseek.com"
llm_model = "deepseek-chat"          # used by every prompt in the simulation
llm_model_large = "deepseek-chat"    # used by the GPT-4 style helper
llm_timeout = 120                    # seconds per HTTP request
llm_max_retries = 6

# Cost accounting (CNY per 1M tokens). Adjust to the current official prices.
llm_price_input = 2.0                # cache miss input price
llm_price_input_cached = 0.5         # cache hit input price
llm_price_output = 8.0

# ---------------------------------------------------------------------------
# Local sentence-embedding model (DeepSeek has no embeddings endpoint)
# ---------------------------------------------------------------------------
# The ONNX model files live under <repo>/models/all-MiniLM-L6-v2.
embedding_model_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "all-MiniLM-L6-v2")
embedding_max_tokens = 256
embedding_backend = os.environ.get("GA_EMBEDDING_BACKEND", "onnx")  # onnx | hash

# ---------------------------------------------------------------------------
# Paths used by the simulation (same values as the upstream README)
# ---------------------------------------------------------------------------
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
