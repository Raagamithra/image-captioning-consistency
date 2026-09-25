import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config_batch import (
    IMAGE_DIR,
    RESULT_DIR,
    JSON_OUTPUT,
    CSV_OUTPUT,
    PROMPT,
    NUM_RUNS,
    TEMPERATURE,
    MODEL_NAME,
)

from src.model import generate_caption


# =========================
# BATCH / RATE LIMIT SETTINGS
# =========================

# Maximum new generation attempts in one program launch.
# Run the script again to continue with the next batch.
MAX_NEW_CALLS_PER_SESSION = 5

# Pause between API calls. This is not a guarantee against
# provider rate limits; increase it if your account requires it.
PAUSE_BETWEEN_CALLS_SECONDS = 3

# How many times to retry a failed generation during one launch.
# Failed runs remain recorded and can be retried on a later launch.
MAX_ATTEMPTS_PER_RUN_PER_SESSION = 1


# =========================
# IMAGE DISCOVERY
# =========================

def get_image_files():
    """Return supported image files in sorted order."""
    image_dir = Path(IMAGE_DIR)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    ]

    return images


# =========================
# RESULT FILE HANDLING
# =========================

def load_results():
    """
    Load existing results from JSON if present.
    Return an empty list if this is a new experiment.
    """
    json_path = Path(JSON_OUTPUT)

    if not json_path.exists():
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            results = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse existing results file: {json_path}. "
            "Make a backup and inspect the file before continuing."
        ) from error

    if not isinstance(results, list):
        raise ValueError(
            f"Expected a JSON list in {json_path}"
        )

    return results


def save_results(results):
    """
    Save results to JSON and CSV.
    JSON is written first; both files are updated after each attempt.
    """
    result_dir = Path(RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(JSON_OUTPUT)
    csv_path = Path(CSV_OUTPUT)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
    )


# =========================
# RUN TRACKING
# =========================

def successful_run_keys(results):
    """
    Return (image_id, run_number) pairs that have successful,
    non-empty captions.
    """
    completed = set()

    for row in results:
        if row.get("status") != "success":
            continue

        image_id = row.get("image_id")
        run_number = row.get("run_number")
        output = row.get("output")

        if not image_id or run_number is None:
            continue

        if not isinstance(output, str) or not output.strip():
            continue

        completed.add((str(image_id), int(run_number)))

    return completed


def existing_error_keys(results):
    """Return image/run pairs with a previously recorded error."""
    failed = set()

    for row in results:
        if row.get("status") != "error":
            continue

        image_id = row.get("image_id")
        run_number = row.get("run_number")

        if image_id and run_number is not None:
            failed.add((str(image_id), int(run_number)))

    return failed


def next_result_record(
    image_path,
    run_number,
    status,
    elapsed_time,
    attempts,
    output=None,
    error=None,
):
    """Create a consistent result row."""
    result = {
        "image_id": image_path.stem,
        "image_file": image_path.name,
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "temperature": TEMPERATURE,
        "run_number": run_number,
        "attempts": attempts,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "generation_time_seconds": round(
            elapsed_time,
            3,
        ),
        "output": output,
        "status": status,
    }

    if error is not None:
        result["error"] = str(error)

    return result


# =========================
# BASELINE EXPERIMENT
# =========================

def run_baseline():
    images = get_image_files()

    if not images:
        raise ValueError(
            f"No supported images found in {IMAGE_DIR}"
        )

    results = load_results()
    completed = successful_run_keys(results)

    # This session tracks how many generation calls it has made.
    new_calls_this_session = 0

    print("=" * 64)
    print("IMAGE-TO-TEXT BASELINE EXPERIMENT")
    print("=" * 64)
    print(f"Model                  : {MODEL_NAME}")
    print(f"Prompt                 : {PROMPT}")
    print(f"Temperature            : {TEMPERATURE}")
    print(f"Images                 : {len(images)}")
    print(f"Target runs per image  : {NUM_RUNS}")
    print(f"Successful runs saved  : {len(completed)}")
    print(f"Max new calls/session  : {MAX_NEW_CALLS_PER_SESSION}")
    print("=" * 64)

    # Check that existing results match the current experiment settings.
    # This prevents accidentally resuming a file made with another model,
    # prompt, or temperature.
    for row in results:
        if row.get("status") not in {"success", "error"}:
            continue

        if row.get("model") != MODEL_NAME:
            raise ValueError(
                "Existing result file contains a different model. "
                "Use a separate output folder for each model."
            )

        if row.get("prompt") != PROMPT:
            raise ValueError(
                "Existing result file contains a different prompt. "
                "Use a separate output folder for each prompt."
            )

        if row.get("temperature") != TEMPERATURE:
            raise ValueError(
                "Existing result file contains a different temperature. "
                "Use a separate output folder for each temperature."
            )

    for image_index, image_path in enumerate(images, start=1):
        image_id = image_path.stem

        print()
        print(
            f"[Image {image_index}/{len(images)}] "
            f"{image_path.name}"
        )

        for run_number in range(1, NUM_RUNS + 1):
            run_key = (image_id, run_number)

            if run_key in completed:
                print(
                    f"  Run {run_number}/{NUM_RUNS}: "
                    "already successful; skipping"
                )
                continue

            if new_calls_this_session >= MAX_NEW_CALLS_PER_SESSION:
                print()
                print("Session batch limit reached.")
                print("Your progress has been saved.")
                print("Run the script again later to continue.")
                print(f"JSON: {JSON_OUTPUT}")
                print(f"CSV : {CSV_OUTPUT}")
                return

            print(
                f"  Run {run_number}/{NUM_RUNS}: "
                "generating..."
            )

            new_calls_this_session += 1
            start_time = time.time()

            try:
                generation_result = generate_caption(
                    image_path=str(image_path),
                    prompt=PROMPT,
                    temperature=TEMPERATURE,
                )

                caption = generation_result.get("output")
                attempts = generation_result.get("attempts", 1)

                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(
                        "The model returned an empty caption."
                    )

                elapsed_time = time.time() - start_time

                result = next_result_record(
                    image_path=image_path,
                    run_number=run_number,
                    status="success",
                    elapsed_time=elapsed_time,
                    attempts=attempts,
                    output=caption.strip(),
                )

                # Only a valid successful result is marked complete.
                completed.add(run_key)

                print(f"    SUCCESS: {caption.strip()}")
                print(f"    Internal attempts: {attempts}")

            except Exception as error:
                elapsed_time = time.time() - start_time

                result = next_result_record(
                    image_path=image_path,
                    run_number=run_number,
                    status="error",
                    elapsed_time=elapsed_time,
                    attempts=MAX_ATTEMPTS_PER_RUN_PER_SESSION,
                    output=None,
                    error=error,
                )

                print(f"    FAILED: {error}")

            results.append(result)

            # Persist every attempt, so a stop or crash doesn't lose
            # previous successful generations.
            save_results(results)

            if new_calls_this_session < MAX_NEW_CALLS_PER_SESSION:
                time.sleep(PAUSE_BETWEEN_CALLS_SECONDS)

    print()
    print("=" * 64)
    print("CURRENT EXPERIMENT STATUS")
    print("=" * 64)

    total_target_runs = len(images) * NUM_RUNS
    successful_count = len(completed)
    missing_count = total_target_runs - successful_count

    print(f"Target successful generations: {total_target_runs}")
    print(f"Successful generations saved : {successful_count}")
    print(f"Still missing                : {missing_count}")
    print(f"JSON                          : {JSON_OUTPUT}")
    print(f"CSV                           : {CSV_OUTPUT}")

    if missing_count == 0:
        print("All image/run pairs have successful captions.")
    else:
        print("Some runs remain missing or failed.")
        print("Run the script again to continue.")


if __name__ == "__main__":
    run_baseline()