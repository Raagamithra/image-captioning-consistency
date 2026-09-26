import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config_batch import CSV_OUTPUT


# ============================================================
# CSR OUTPUT DIRECTORY
# ============================================================

CSR_RESULT_DIR = Path("experiments/csr_results")


# ============================================================
# CSR CLASS
# ============================================================

class CSR:

    def __init__(self):

        print()
        print("=" * 60)
        print("LOADING CSR MODEL")
        print("=" * 60)

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("CSR model loaded successfully.")


    # ========================================================
    # SELECT ONE CAPTION FROM 5 CAPTIONS
    # ========================================================

    def select_consistent_caption(self, captions):

        if not captions:
            raise ValueError(
                "No captions provided."
            )

        if len(captions) == 1:
            return captions[0]

        # ----------------------------------------------------
        # Convert captions into semantic embeddings
        # ----------------------------------------------------

        embeddings = self.model.encode(
            captions,
            convert_to_numpy=True
        )

        # ----------------------------------------------------
        # Calculate cosine similarity matrix
        # ----------------------------------------------------

        similarity_matrix = cosine_similarity(
            embeddings
        )

        # ----------------------------------------------------
        # Calculate average similarity
        # for every caption
        # ----------------------------------------------------

        average_scores = []

        for i in range(len(captions)):

            scores = []

            for j in range(len(captions)):

                if i != j:

                    scores.append(
                        similarity_matrix[i][j]
                    )

            average_score = (
                sum(scores) / len(scores)
            )

            average_scores.append(
                average_score
            )

        # ----------------------------------------------------
        # Select caption with highest average similarity
        # ----------------------------------------------------

        best_index = average_scores.index(
            max(average_scores)
        )

        best_caption = captions[best_index]

        return best_caption


# ============================================================
# LOAD BASELINE CSV
# ============================================================

def load_baseline_results():

    baseline_csv = Path(CSV_OUTPUT)

    if not baseline_csv.exists():

        raise FileNotFoundError(
            f"\nBaseline CSV not found:\n"
            f"{baseline_csv}\n\n"
            f"Please run baseline.py first."
        )

    print()
    print("Loading baseline results...")
    print(f"File: {baseline_csv}")

    dataframe = pd.read_csv(
        baseline_csv
    )

    return dataframe


# ============================================================
# CONVERT NUMPY/PANDAS VALUES TO PYTHON VALUES
# ============================================================

def convert_to_python(value):

    # Convert numpy integers/floats/bools
    # into normal Python types

    if hasattr(value, "item"):

        return value.item()

    return value


# ============================================================
# SAVE CSR RESULTS
# ============================================================

def save_csr_results(results):

    # --------------------------------------------------------
    # Create separate CSR folder
    # --------------------------------------------------------

    CSR_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Convert all values to JSON-compatible Python values
    # --------------------------------------------------------

    clean_results = []

    for result in results:

        clean_result = {}

        for key, value in result.items():

            clean_result[key] = (
                convert_to_python(value)
            )

        clean_results.append(
            clean_result
        )

    # --------------------------------------------------------
    # CSR JSON
    # --------------------------------------------------------

    json_path = (
        CSR_RESULT_DIR /
        "csr_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            clean_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # CSR CSV
    # --------------------------------------------------------

    csv_path = (
        CSR_RESULT_DIR /
        "csr_results.csv"
    )

    dataframe = pd.DataFrame(
        clean_results
    )

    # --------------------------------------------------------
    # Exact column order
    # --------------------------------------------------------

    columns = [
        "image_id",
        "image_file",
        "model",
        "prompt",
        "temperature",
        "run_number",
        "attempts",
        "timestamp_utc",
        "generation_time_seconds",
        "output",
        "status",
        "error",
    ]

    dataframe = dataframe[
        columns
    ]

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    return csv_path, json_path


# ============================================================
# RUN CSR
# ============================================================

def run_csr():

    print()
    print("=" * 60)
    print("CANONICAL SEMANTIC REPRESENTATION (CSR)")
    print("=" * 60)

    # --------------------------------------------------------
    # Load baseline CSV
    # --------------------------------------------------------

    dataframe = load_baseline_results()

    print()
    print(
        f"Total baseline rows: {len(dataframe)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "image_id",
        "image_file",
        "model",
        "prompt",
        "temperature",
        "run_number",
        "attempts",
        "timestamp_utc",
        "generation_time_seconds",
        "output",
        "status",
    ]

    for column in required_columns:

        if column not in dataframe.columns:

            raise ValueError(
                f"Missing column in baseline CSV: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Keep successful captions only
    # --------------------------------------------------------

    successful = dataframe[
        (dataframe["status"] == "success")
        &
        (dataframe["output"].notna())
        &
        (
            dataframe["output"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    print(
        f"Successful captions: "
        f"{len(successful)}"
    )

    if successful.empty:

        raise ValueError(
            "No successful captions found."
        )

    # --------------------------------------------------------
    # Load CSR model
    # --------------------------------------------------------

    csr = CSR()

    # --------------------------------------------------------
    # Group captions by image
    # --------------------------------------------------------

    grouped = successful.groupby(
        "image_id",
        sort=False
    )

    results = []

    total_images = len(grouped)

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for image_number, (
        image_id,
        group
    ) in enumerate(
        grouped,
        start=1
    ):

        print()
        print("=" * 60)
        print(
            f"IMAGE {image_number}/{total_images}: "
            f"{image_id}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Sort according to run number
        # ----------------------------------------------------

        group = group.sort_values(
            "run_number"
        )

        # ----------------------------------------------------
        # Select maximum 5 successful captions
        # ----------------------------------------------------

        group = group.head(5)

        captions = (
            group["output"]
            .astype(str)
            .tolist()
        )

        print()
        print(
            f"Captions used for CSR: "
            f"{len(captions)}"
        )

        # ----------------------------------------------------
        # Display captions
        # ----------------------------------------------------

        for number, caption in enumerate(
            captions,
            start=1
        ):

            print()
            print(
                f"Caption {number}:"
            )
            print(caption)

        # ----------------------------------------------------
        # Check caption count
        # ----------------------------------------------------

        if not captions:

            print(
                "No captions found. Skipping image."
            )

            continue

        if len(captions) < 5:

            print()
            print(
                f"WARNING: Only {len(captions)} "
                f"successful captions available."
            )

        # ----------------------------------------------------
        # CSR selection
        # ----------------------------------------------------

        selected_caption = (
            csr.select_consistent_caption(
                captions
            )
        )

        print()
        print("-" * 60)
        print("SELECTED CSR CAPTION")
        print("-" * 60)
        print(selected_caption)

        # ----------------------------------------------------
        # Find original row of selected caption
        # ----------------------------------------------------

        selected_rows = group[
            group["output"]
            .astype(str)
            == selected_caption
        ]

        if not selected_rows.empty:

            selected_row = (
                selected_rows.iloc[0]
            )

        else:

            selected_row = group.iloc[0]

        # ----------------------------------------------------
        # Create CSR result
        # ----------------------------------------------------

        result = {

            "image_id": str(
                image_id
            ),

            "image_file": str(
                selected_row["image_file"]
            ),

            "model": str(
                selected_row["model"]
            ),

            "prompt": str(
                selected_row["prompt"]
            ),

            "temperature": convert_to_python(
                selected_row["temperature"]
            ),

            "run_number": convert_to_python(
                selected_row["run_number"]
            ),

            "attempts": convert_to_python(
                selected_row["attempts"]
            ),

            "timestamp_utc": str(
                selected_row["timestamp_utc"]
            ),

            "generation_time_seconds":
                convert_to_python(
                    selected_row[
                        "generation_time_seconds"
                    ]
                ),

            "output": str(
                selected_caption
            ),

            "status": "success",

            "error": None,
        }

        results.append(
            result
        )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        raise ValueError(
            "No CSR results were generated."
        )

    # ========================================================
    # SAVE CSR RESULTS
    # ========================================================

    csv_path, json_path = (
        save_csr_results(results)
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("CSR EXPERIMENT COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Images processed : {len(results)}"
    )

    print(
        f"Captions selected: {len(results)}"
    )

    print()
    print(
        f"CSR CSV : {csv_path}"
    )

    print(
        f"CSR JSON: {json_path}"
    )

    print()
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_csr()