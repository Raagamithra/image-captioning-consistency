import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# API CONFIGURATION
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Add it to your .env file."
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================
# MODEL CONFIGURATION
# =========================

MODEL_NAME = (
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)


# =========================
# EXPERIMENT CONFIGURATION
# =========================

PROMPT = "Describe this image in one sentence."

# Target successful generations per image
NUM_RUNS = 2

# Initial baseline temperature
TEMPERATURE = 0.0


# =========================
# PATH CONFIGURATION
# =========================

IMAGE_DIR = "data/my_baseline_images"

# Keep this folder name to continue using the saved results
RESULT_DIR = "experiments/baseline_batch_results"

JSON_OUTPUT = f"{RESULT_DIR}/baseline_results_t{TEMPERATURE}.json"
CSV_OUTPUT = f"{RESULT_DIR}/baseline_results_t{TEMPERATURE}.csv"