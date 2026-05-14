from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAINING_EXAMPLES = [
    ("убери полосы на градиенте", "gradient", "glass", "medium"),
    ("smooth pink peach gradient remove banding", "gradient", "glass", "medium"),
    ("сделай тени чище", "shadow", "glass", "medium"),
    ("dramatic realistic cast shadow", "shadow", "glass", "high"),
    ("мокрый асфальт отражение", "reflection", "asphalt", "high"),
    ("water reflection natural ripple", "reflection", "water", "high"),
    ("sharp mirror reflection", "reflection", "mirror", "medium"),
    ("glass reflection clean highlights", "reflection", "glass", "medium"),
]


def train_classifier(
    output_path: str = "backend/models/problem_classifier.joblib",
    manifest_path: str = "backend/data/training/manifest.jsonl",
    report_path: str = "backend/models/problem_classifier_report.json",
) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import joblib

    rows = _load_manifest(manifest_path)
    if rows:
        prompts = [_example_text(row) for row in rows]
        y = [[str(row["problem"]), str(row["material"]), str(row["strength"])] for row in rows]
        dataset_source = manifest_path
    else:
        prompts = [item[0] for item in TRAINING_EXAMPLES]
        y = [[item[1], item[2], item[3]] for item in TRAINING_EXAMPLES]
        dataset_source = "built-in seed examples"

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(2, 5), analyzer="char_wb", lowercase=True)),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=800, C=4.0))),
        ]
    )
    report: dict[str, Any] = {
        "dataset_source": dataset_source,
        "examples": len(prompts),
    }
    labels_for_split = [item[0] for item in y]
    if len(set(labels_for_split)) >= 2 and len(y) >= 12:
        x_train, x_test, y_train, y_test = train_test_split(
            prompts,
            y,
            test_size=0.22,
            random_state=42,
            stratify=labels_for_split if min(labels_for_split.count(label) for label in set(labels_for_split)) >= 2 else None,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        y_test_tuples = [tuple(item) for item in y_test]
        prediction_tuples = [tuple(item) for item in predictions]
        exact_matches = [expected == predicted for expected, predicted in zip(y_test_tuples, prediction_tuples)]
        report["holdout_accuracy"] = round(float(sum(exact_matches) / max(1, len(exact_matches))), 4)
        report["problem_accuracy"] = round(float(accuracy_score([item[0] for item in y_test], [item[0] for item in predictions])), 4)
        report["material_accuracy"] = round(float(accuracy_score([item[1] for item in y_test], [item[1] for item in predictions])), 4)
        report["strength_accuracy"] = round(float(accuracy_score([item[2] for item in y_test], [item[2] for item in predictions])), 4)
        report["problem_report"] = classification_report(
            [item[0] for item in y_test],
            [item[0] for item in predictions],
            output_dict=True,
            zero_division=0,
        )
        model.fit(prompts, y)
    else:
        model.fit(prompts, y)
        report["holdout_accuracy"] = None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: str) -> list[dict[str, Any]]:
    manifest = Path(path)
    if not manifest.exists():
        return []
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if {"problem", "material", "strength"}.issubset(row.keys()):
                rows.append(row)
    return rows


def _example_text(row: dict[str, Any]) -> str:
    return (
        f"{row.get('prompt', '')} "
        f"problem:{row.get('problem', '')} "
        f"material:{row.get('material', '')} "
        f"strength:{row.get('strength', '')}"
    )


if __name__ == "__main__":
    train_classifier()
