import json
import re

from pathlib import Path

import cv2
import numpy as np
import iris


# --------------------------------------------------
# Project paths
# --------------------------------------------------

DATASET_DIR = Path("dataset")
GALLERY_DIR = Path("storage/iris-gallery")

METADATA_FILE = GALLERY_DIR / "metadata.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Set to None to process the entire dataset.
MAX_IMAGES = None


# Expected filename format:
# S1012R01
# S1230L05
FILENAME_PATTERN = re.compile(
    r"^S(?P<filename_subject_id>\d+)"
    r"(?P<eye>[LR])"
    r"(?P<capture_id>\d+)$",
    re.IGNORECASE
)


# --------------------------------------------------
# Extract image metadata
# --------------------------------------------------

def extract_metadata(image_path: Path):
    """
    Extract subject ID, eye side and capture ID.

    The subject folder is used as the canonical
    subject ID, for example:

        dataset/230/L/S1230L01.jpg

    Subject ID: 230
    Eye side: left
    Capture ID: 01
    """

    match = FILENAME_PATTERN.match(image_path.stem)

    if not match:
        raise ValueError(
            f"Unexpected filename format: {image_path.name}"
        )

    filename_eye = match.group("eye").upper()
    capture_id = match.group("capture_id")

    # Use the subject directory as the canonical ID.
    subject_id = image_path.parent.parent.name

    # Determine eye side from the filename.
    eye_side = "left" if filename_eye == "L" else "right"

    # Validate against parent directory.
    folder_eye = image_path.parent.name.upper()

    if folder_eye != filename_eye:
        raise ValueError(
            f"Eye mismatch in {image_path}: "
            f"filename says {filename_eye}, "
            f"folder says {folder_eye}"
        )

    return subject_id, eye_side, capture_id


# --------------------------------------------------
# Generate iris template
# --------------------------------------------------

def generate_template(pipeline, image_path: Path):
    """
    Process an image using Open-IRIS
    and return its iris template.
    """

    subject_id, eye_side, capture_id = extract_metadata(image_path)

    img_pixels = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img_pixels is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    # Open-IRIS expects an IRImage object.
    ir_image = iris.IRImage(
        img_data=img_pixels,
        image_id=image_path.stem,
        eye_side=eye_side
    )

    # Execute the pipeline.
    output = pipeline(ir_image)

    if output["error"] is not None:
        raise RuntimeError(
            f"Open-IRIS processing failed: {output['error']}"
        )

    raw_template = output["iris_template"]

    # IrisTemplate exposes iris_codes and mask_codes
    # as attributes.
    iris_codes = np.asarray(raw_template.iris_codes)
    mask_codes = np.asarray(raw_template.mask_codes)

    return {
        "subject_id": subject_id,
        "eye_side": eye_side,
        "capture_id": capture_id,
        "iris_codes": iris_codes,
        "mask_codes": mask_codes
    }


# --------------------------------------------------
# Enroll dataset
# --------------------------------------------------

def enroll_dataset():

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing Open-IRIS...")

    # Initialize the pipeline only once.
    pipeline = iris.IRISPipeline()

    print("Pipeline initialized successfully.")

    # Find all supported images.
    images = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Found {len(images)} images.")

    if not images:
        raise RuntimeError(
            f"No images found in {DATASET_DIR}"
        )

    # Limit images during initial testing.
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    print(f"Images to process: {len(images)}")

    enrolled = 0
    failed = 0
    metadata = []

    # Process each image.
    for index, image_path in enumerate(images, start=1):

        try:

            template = generate_template(
                pipeline,
                image_path
            )

            subject_id = template["subject_id"]
            eye_side = template["eye_side"]
            capture_id = template["capture_id"]

            template_filename = (
                f"subject_{subject_id}_"
                f"{eye_side}_"
                f"capture_{capture_id}.npz"
            )

            template_path = GALLERY_DIR / template_filename

            # Save iris codes and masks.
            np.savez_compressed(
                template_path,
                iris_codes=template["iris_codes"],
                mask_codes=template["mask_codes"]
            )

            # Store metadata.
            metadata.append({
                "subject_id": subject_id,
                "eye_side": eye_side,
                "capture_id": capture_id,
                "source_image": str(image_path),
                "template_file": template_filename
            })

            enrolled += 1

            print(
                f"[{index}/{len(images)}] Enrolled: "
                f"Subject {subject_id}, "
                f"Eye: {eye_side}, "
                f"Capture: {capture_id}"
            )

        except Exception as error:

            failed += 1

            print(
                f"\n[{index}/{len(images)}] Failed: {image_path}"
                f"\nReason: {error}"
            )

        if index % 50 == 0:
            print(
                f"\nProgress: {index}/{len(images)} "
                f"| Enrolled: {enrolled} "
                f"| Failed: {failed}\n"
            )

    # Save metadata.
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nEnrollment completed!")
    print(f"Total images: {len(images)}")
    print(f"Successfully enrolled: {enrolled}")
    print(f"Failed: {failed}")
    print(f"Gallery location: {GALLERY_DIR}")
    print(f"Metadata file: {METADATA_FILE}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    enroll_dataset()
