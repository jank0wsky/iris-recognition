import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import iris
import numpy as np


# Project root: parent directory of src/
PROJECT_DIR = Path(__file__).resolve().parent.parent

GALLERY_DIR = PROJECT_DIR / "storage" / "iris-gallery"
METADATA_FILE = GALLERY_DIR / "metadata.json"


def generate_template(pipeline, image_path: Path, eye_side: str):
    """Generate an iris template from the input image."""

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img_pixels = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img_pixels is None:
        raise ValueError(f"Could not read image: {image_path}")

    ir_image = iris.IRImage(
        img_data=img_pixels,
        image_id=image_path.stem,
        eye_side=eye_side
    )

    output = pipeline(ir_image)

    if output.get("error"):
        raise RuntimeError(
            f"Open-IRIS failed to process {image_path}: {output['error']}"
        )

    template = output.get("iris_template")

    if template is None:
        raise RuntimeError(
            f"Open-IRIS did not generate a template for {image_path}"
        )

    return template


def load_gallery(eye_side: str, query_template):
    """
    Load existing templates from the gallery.

    Supports older NPZ files without iris_code_version by using
    the version from the newly generated query template.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Gallery metadata not found: {METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    gallery = []

    for record in metadata:
        if record.get("eye_side") != eye_side:
            continue

        template_path = GALLERY_DIR / record["template_file"]

        if not template_path.exists():
            print(f"Warning: template file not found: {template_path}")
            continue

        try:
            with np.load(template_path, allow_pickle=False) as data:

                iris_codes = list(data["iris_codes"])
                mask_codes = list(data["mask_codes"])

                # Older gallery files may not contain this field.
                if "iris_code_version" in data.files:
                    version_value = data["iris_code_version"].item()
                    iris_code_version = str(version_value)
                else:
                    iris_code_version = query_template.iris_code_version

                template = iris.IrisTemplate(
                    iris_codes=iris_codes,
                    mask_codes=mask_codes,
                    iris_code_version=iris_code_version
                )

            gallery.append({
                "subject_id": record["subject_id"],
                "eye_side": record["eye_side"],
                "capture_id": record.get("capture_id"),
                "source_image": record.get("source_image"),
                "template": template
            })

        except Exception as error:
            print(
                f"Warning: could not load {template_path}: {error}"
            )

    return gallery


def identify(image_path: Path, eye_side: str, top_n: int = 5):
    """Compare the input iris against every enrolled template."""

    print("Initializing Open-IRIS...")

    pipeline = iris.IRISPipeline()
    matcher = iris.HammingDistanceMatcher()

    print("Generating template for input image...")

    query_template = generate_template(
        pipeline,
        image_path,
        eye_side
    )

    print("Loading gallery...")

    gallery = load_gallery(eye_side, query_template)

    if not gallery:
        print(f"No gallery templates found for eye side: {eye_side}")
        return

    print(f"Loaded {len(gallery)} templates.")
    print("Comparing templates...\n")

    results = []

    for entry in gallery:
        distance = matcher.run(
            query_template,
            entry["template"]
        )

        results.append({
            "subject_id": entry["subject_id"],
            "eye_side": entry["eye_side"],
            "capture_id": entry["capture_id"],
            "source_image": entry["source_image"],
            "distance": distance
        })

    # Sort all individual image comparisons by Hamming distance.
    ranked_results = sorted(
        results,
        key=lambda result: result["distance"]
    )

    print("=" * 75)
    print("IDENTIFICATION RESULTS")
    print("=" * 75)

    print(f"Input image: {image_path}")
    print(f"Eye side: {eye_side}")
    print(f"Total comparisons: {len(ranked_results)}")
    print()

    for index, result in enumerate(ranked_results[:top_n], start=1):
        print(f"Rank: {index}")
        print(f"Subject ID: {result['subject_id']}")
        print(f"Eye side: {result['eye_side']}")
        print(f"Capture ID: {result['capture_id']}")
        print(f"Source image: {result['source_image']}")
        print(f"Hamming distance: {result['distance']:.6f}")
        print("-" * 75)

    if ranked_results:
        best_match = ranked_results[0]

        print("\nBEST MATCHING IMAGE")
        print(f"Subject ID: {best_match['subject_id']}")
        print(f"Source image: {best_match['source_image']}")
        print(f"Hamming distance: {best_match['distance']:.6f}")

    print(
        "\nNote: Every enrolled image is evaluated independently. "
        "Multiple images from the same subject are displayed separately. "
        "Lower Hamming distance indicates greater similarity."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Identify an iris using an existing Open-IRIS gallery."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input iris image."
    )

    parser.add_argument(
        "--eye",
        choices=["left", "right"],
        required=True,
        help="Eye side of the input image."
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top candidates to display (default: 5)."
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    try:
        identify(
            image_path=args.image,
            eye_side=args.eye,
            top_n=args.top
        )
    except Exception as error:
        print(f"\nIdentification failed: {error}")
        raise


if __name__ == "__main__":
    main()
