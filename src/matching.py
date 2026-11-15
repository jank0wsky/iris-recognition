from pathlib import Path

import cv2
import iris


# --------------------------------------------------
# Project paths
# --------------------------------------------------

DATASET_DIR = Path("dataset")


# --------------------------------------------------
# Generate iris template
# --------------------------------------------------

def extract_template(image_path: Path):
    """
    Generate an iris template from a CASIA image.
    """

    # Determine eye side from the parent folder.
    eye_folder = image_path.parent.name.upper()

    if eye_folder == "L":
        eye_side = "left"
    elif eye_folder == "R":
        eye_side = "right"
    else:
        raise ValueError(f"Unknown eye side: {image_path}")

    # Read image as grayscale.
    img_pixels = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img_pixels is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    # Create Open-IRIS image object.
    ir_image = iris.IRImage(
        img_data=img_pixels,
        image_id=image_path.stem,
        eye_side=eye_side
    )

    # Run Open-IRIS pipeline.
    output = pipeline(ir_image)

    if output["error"] is not None:
        raise RuntimeError(
            f"Open-IRIS failed for {image_path}: {output['error']}"
        )

    # Already an IrisTemplate object.
    return output["iris_template"]


# --------------------------------------------------
# Initialize pipeline and matcher
# --------------------------------------------------

print("Initializing Open-IRIS...")

pipeline = iris.IRISPipeline()
matcher = iris.HammingDistanceMatcher()

print("Open-IRIS initialized.")


# --------------------------------------------------
# Select test images
# --------------------------------------------------

# Two captures of the same anatomical eye.
same_eye_1 = DATASET_DIR / "015/R/S1015R02.jpg"
same_eye_2 = DATASET_DIR / "015/R/S1015R03.jpg"

# Image from another subject.
different_eye = DATASET_DIR / "090/R/S1090R01.jpg"


# --------------------------------------------------
# Generate templates
# --------------------------------------------------

print("\nGenerating templates...")

template_1 = extract_template(same_eye_1)
template_2 = extract_template(same_eye_2)
template_3 = extract_template(different_eye)


# --------------------------------------------------
# Inspect generated template
# --------------------------------------------------

# print("\nTemplate information:")

# print("Template type:", type(template_1))
# print("Iris codes:", len(template_1.iris_codes))
# print("Mask codes:", len(template_1.mask_codes))
# print("First iris code shape:", template_1.iris_codes[0].shape)


# --------------------------------------------------
# Perform matching
# --------------------------------------------------

print("\nMatching results:")

# Compare image against itself.
same_image_distance = matcher.run(
    template_1,
    template_1
)

# Compare two captures of the same eye.
same_eye_distance = matcher.run(
    template_1,
    template_2
)

# Compare against another subject.
different_eye_distance = matcher.run(
    template_1,
    template_3
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print(f"Image against itself: {same_image_distance:.6f}")

print(
    f"Same eye, different capture: "
    f"{same_eye_distance:.6f}"
)

print(
    f"Different subject: "
    f"{different_eye_distance:.6f}"
)
