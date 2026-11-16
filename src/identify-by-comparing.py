import argparse
from pathlib import Path

import cv2
import iris


def generate_template(image_path: Path, eye_side: str, pipeline):
    """Generate an iris template from an image."""

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
            f"Iris processing failed for {image_path}: {output['error']}"
        )

    template = output.get("iris_template")

    if template is None:
        raise RuntimeError(
            f"Open-IRIS did not generate a template for {image_path}"
        )

    return template


def compare_images(
    image1: Path,
    image2: Path,
    eye_side: str
):
    """Generate templates and compare two iris images."""

    print("Initializing Open-IRIS...")

    pipeline = iris.IRISPipeline()
    matcher = iris.HammingDistanceMatcher()

    print("\nGenerating template for first image...")

    template1 = generate_template(
        image1,
        eye_side,
        pipeline
    )

    print("First template generated successfully.")

    print("\nGenerating template for second image...")

    template2 = generate_template(
        image2,
        eye_side,
        pipeline
    )

    print("Second template generated successfully.")

    print("\nComparing templates...")

    distance = matcher.run(
        template1,
        template2
    )

    print("\n========== COMPARISON RESULTS ==========")
    print(f"Image 1:        {image1}")
    print(f"Image 2:        {image2}")
    print(f"Eye side:       {eye_side}")
    print(f"Hamming distance: {float(distance):.6f}")
    print("========================================")

    print(
        "\nInterpretation: A lower Hamming distance indicates "
        "greater similarity between the two iris templates."
    )

    return float(distance)


def main():

    parser = argparse.ArgumentParser(
        description="Compare two iris images using Open-IRIS."
    )

    parser.add_argument(
        "image1",
        type=Path,
        help="Path to the first iris image."
    )

    parser.add_argument(
        "image2",
        type=Path,
        help="Path to the second iris image."
    )

    parser.add_argument(
        "--eye",
        choices=["left", "right"],
        required=True,
        help="Eye side shared by both images."
    )

    args = parser.parse_args()

    if not args.image1.is_file():
        parser.error(f"First image does not exist: {args.image1}")

    if not args.image2.is_file():
        parser.error(f"Second image does not exist: {args.image2}")

    try:
        compare_images(
            image1=args.image1,
            image2=args.image2,
            eye_side=args.eye
        )
    except Exception as error:
        print(f"\nComparison failed: {error}")
        raise


if __name__ == "__main__":
    main()
