from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image
from lwcc import LWCC


DATASET_DIR = Path(r"D:\jhu_crowd_v2.0")
TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_GT_DIR = DATASET_DIR / "test" / "gt"
OUTPUT_CSV = Path("outputs/csrnet_test_results.csv")


def get_ground_truth_count(annotation_path):
    """
    Count valid annotation rows.

    This assumes each non-empty row represents one annotated object.
    Verify the dataset's annotation specification before treating
    this as the official person count.
    """
    with open(annotation_path, "r", encoding="utf-8") as file:
        rows = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return len(rows)


def main():
    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {TEST_IMAGE_DIR}"
        )

    if not TEST_GT_DIR.is_dir():
        raise FileNotFoundError(
            f"Annotation directory not found: {TEST_GT_DIR}"
        )

    image_paths = sorted(
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = []

    print(f"Found {len(image_paths)} test images.")
    print("Evaluating CSRNet on CPU...")

    for index, image_path in enumerate(image_paths, start=1):
        annotation_path = TEST_GT_DIR / f"{image_path.stem}.txt"

        if not annotation_path.is_file():
            skipped.append(image_path.name)
            continue

        try:
            ground_truth = get_ground_truth_count(annotation_path)

            start = time.perf_counter()

            predicted_raw = LWCC.get_count(
                str(image_path),
                model_name="CSRNet",
                model_weights="SHA",
            )

            inference_seconds = time.perf_counter() - start

            if isinstance(predicted_raw, dict):
                predicted_raw = next(iter(predicted_raw.values()))

            prediction = max(0, int(round(float(predicted_raw))))
            error = prediction - ground_truth

            with Image.open(image_path) as image:
                width, height = image.size

            results.append({
                "image": image_path.name,
                "width": width,
                "height": height,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "absolute_error": abs(error),
                "signed_error": error,
                "inference_seconds": round(inference_seconds, 3),
            })

            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"GT={ground_truth}, prediction={prediction}, "
                f"absolute error={abs(error)}"
            )

        except Exception as error:
            print(f"Skipping {image_path.name}: {error}")
            skipped.append(image_path.name)

    if not results:
        raise RuntimeError(
            "No images were evaluated successfully."
        )

    absolute_errors = np.array(
        [row["absolute_error"] for row in results],
        dtype=float,
    )

    signed_errors = np.array(
        [row["signed_error"] for row in results],
        dtype=float,
    )

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(signed_errors ** 2)))
    mean_bias = float(np.mean(signed_errors))
    mean_inference = float(
        np.mean([row["inference_seconds"] for row in results])
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 45)
    print("CSRNet TEST RESULTS")
    print("=" * 45)
    print(f"Images evaluated: {len(results)}")
    print(f"Images skipped: {len(skipped)}")
    print(f"MAE: {mae:.3f} people")
    print(f"RMSE: {rmse:.3f} people")
    print(f"Mean signed error: {mean_bias:.3f} people")
    print(f"Mean inference time: {mean_inference:.3f} seconds")
    print(f"CSV saved to: {OUTPUT_CSV.resolve()}")

    if skipped:
        print("\nSkipped images:")
        print(", ".join(skipped[:20]))


if __name__ == "__main__":
    main()