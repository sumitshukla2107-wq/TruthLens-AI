
from pathlib import Path
import sys

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "test_dataset" / "images"
BACKEND_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BACKEND_DIR))

from analyses.detector import predict_image

class_names = ["Real", "AI-generated"]
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

y_true = []
y_pred = []
errors = []

folders = [
    ("real", 0),
    ("ai_generated", 1),
]

print("\nTruthLens AI - Image Detector Evaluation")
print("=" * 45)

for folder_name, true_label in folders:
    folder = DATASET_DIR / folder_name

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    print(f"\n{folder_name}: {len(files)} images")

    for path in files:
        try:
            with Image.open(path) as image:
                result = predict_image(image.copy())

            predicted_label = (
                1 if result["label"] == "AI-generated" else 0
            )

            y_true.append(true_label)
            y_pred.append(predicted_label)

            status = "CORRECT" if true_label == predicted_label else "WRONG"

            print(
                f"{path.name}: {result['label']} "
                f"(Real {result['real_probability']}%, "
                f"AI {result['ai_probability']}%) - {status}"
            )

        except Exception as exc:
            errors.append((path.name, str(exc)))
            print(f"ERROR: {path.name}: {exc}")

if not y_true:
    print("\nNo images were evaluated. Check your dataset folders.")
    sys.exit(1)

print("\n" + "=" * 45)
print("BASELINE RESULTS")
print("=" * 45)

print(f"Successfully evaluated: {len(y_true)}")
print(f"Skipped due to errors: {len(errors)}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(
    f"Precision: "
    f"{precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%"
)
print(
    f"Recall:    "
    f"{recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%"
)

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    )
)

matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("                 Predicted Real  Predicted AI")
print(f"Actual Real      {matrix[0][0]:>13}  {matrix[0][1]:>12}")
print(f"Actual AI        {matrix[1][0]:>13}  {matrix[1][1]:>12}")

if errors:
    print("\nFiles with errors:")
    for filename, error in errors:
        print(f"- {filename}: {error}")
