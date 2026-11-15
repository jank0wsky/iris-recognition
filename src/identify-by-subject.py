import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import iris
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

GALLERY_DIR = PROJECT_DIR / "storage" / "iris-gallery"
METADATA_FILE = GALLERY_DIR / "metadata.json"


def generate_template(image_path: Path, eye_side: str, pipeline):
    """Generate an iris template from an input image."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_pixels = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img_pixels is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    ir_image = iris.IRImage(
        img_data=img_pixels,
        image_id=image_path.stem,
        eye_side=eye_side
    )

    output = pipeline(ir_image)

    if output.get("error") is not None:
        raise RuntimeError(
            f"Iris processing failed: {output['error']}"
        )

    template = output.get("iris_template")

    if template is None:
        raise RuntimeError(
            "Open-IRIS did not generate an iris template."
        )

    return template


def is_same_image(query_image: Path, gallery_image: str) -> bool:
    """
    Check whether a gallery image is the same physical file
    as the query image.

    Relative gallery paths are resolved against PROJECT_DIR.
    """

    if not gallery_image:
        return False

    gallery_path = Path(gallery_image)

    if not gallery_path.is_absolute():
        gallery_path = PROJECT_DIR / gallery_path

    try:
        return query_image.resolve() == gallery_path.resolve()
    except OSError:
        return False


def load_gallery(eye_side: str, query_template):
    """
    Load enrolled templates for the selected eye side.

    Supports existing NPZ files both with and without
    iris_code_version.
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
            print(f"WARNING: Missing template: {template_path}")
            continue

        try:
            with np.load(template_path, allow_pickle=False) as data:

                iris_codes = list(data["iris_codes"])
                mask_codes = list(data["mask_codes"])

                # Support older templates without iris_code_version.
                if "iris_code_version" in data.files:
                    version = str(
                        data["iris_code_version"].item()
                    )
                else:
                    version = query_template.iris_code_version

                template = iris.IrisTemplate(
                    iris_codes=iris_codes,
                    mask_codes=mask_codes,
                    iris_code_version=version
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
                f"WARNING: Could not load {template_path}: {error}"
            )

    return gallery


def group_results_by_identity(results):
    """
    Group individual capture results by subject and eye.

    Keep the capture with the lowest Hamming distance
    for each subject.
    """

    identities = defaultdict(list)

    for result in results:
        identity = (
            result["subject_id"],
            result["eye_side"]
        )

        identities[identity].append(result)

    ranked_identities = []

    for (subject_id, eye_side), captures in identities.items():

        best_capture = min(
            captures,
            key=lambda item: item["distance"]
        )

        ranked_identities.append({
            "subject_id": subject_id,
            "eye_side": eye_side,
            "distance": best_capture["distance"],
            "best_capture_id": best_capture["capture_id"],
            "best_source_image": best_capture["source_image"],
            "total_captures": len(captures)
        })

    ranked_identities.sort(
        key=lambda item: item["distance"]
    )

    return ranked_identities


def identify_by_subject(
    image_path: Path,
    eye_side: str,
    top_n: int
):
    """Identify the subject with the best matching enrolled iris."""

    image_path = image_path.resolve()

    print("Initializing Open-IRIS...")

    pipeline = iris.IRISPipeline()
    matcher = iris.HammingDistanceMatcher()

    print("Generating query template...")

    query_template = generate_template(
        image_path,
        eye_side,
        pipeline
    )

    print("Loading gallery...")

    gallery = load_gallery(
        eye_side,
        query_template
    )

    if not gallery:
        raise RuntimeError(
            f"No enrolled templates found for {eye_side} eyes."
        )

    print(f"Loaded {len(gallery)} templates.")

    results = []
    skipped_same_image = 0

    print("\nComparing uploaded image against gallery...\n")

    for index, entry in enumerate(gallery, start=1):

        # Do not compare the query image against itself.
        if is_same_image(
            image_path,
            entry["source_image"]
        ):
            skipped_same_image += 1
            continue

        distance = matcher.run(
            query_template,
            entry["template"]
        )

        results.append({
            "subject_id": entry["subject_id"],
            "eye_side": entry["eye_side"],
            "capture_id": entry["capture_id"],
            "source_image": entry["source_image"],
            "distance": float(distance)
        })

        if index % 100 == 0 or index == len(gallery):
            percentage = index / len(gallery) * 100

            print(
                f"Progress: {index}/{len(gallery)} "
                f"({percentage:.2f}%)"
            )

    if not results:
        raise RuntimeError(
            "No templates available for comparison after excluding "
            "the query image."
        )

    # Keep only the best capture for each subject.
    ranked_identities = group_results_by_identity(results)

    print("\n========== IDENTIFICATION RESULTS ==========")

    print(f"Query image:          {image_path}")
    print(f"Eye side:             {eye_side}")
    print(f"Gallery templates:    {len(gallery)}")
    print(f"Actual comparisons:   {len(results)}")
    print(f"Unique identities:    {len(ranked_identities)}")

    print(
        f"\nTop {min(top_n, len(ranked_identities))} subjects:\n"
    )

    for rank, identity in enumerate(
        ranked_identities[:top_n],
        start=1
    ):

        print(f"Rank:             {rank}")
        print(f"Subject ID:       {identity['subject_id']}")
        print(f"Eye side:         {identity['eye_side']}")
        print(f"Best capture:     {identity['best_capture_id']}")
        print(f"Distance:         {identity['distance']:.6f}")
        print(f"Total captures:   {identity['total_captures']}")
        print(f"Source image:     {identity['best_source_image']}")
        print("-" * 55)

    print("============================================\n")

    return ranked_identities


def main():

    parser = argparse.ArgumentParser(
        description="Identify an iris image by subject."
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
        default=10,
        help="Number of unique subjects to display."
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    if args.top < 1:
        parser.error("--top must be at least 1")

    identify_by_subject(
        image_path=args.image,
        eye_side=args.eye,
        top_n=args.top
    )


if __name__ == "__main__":
    main()
