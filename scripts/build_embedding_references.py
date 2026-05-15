import argparse
import csv
from pathlib import Path


AI_LABELS = {"ai", "generated", "1", "true", "machine", "llm"}
HUMAN_LABELS = {"human", "0", "false", "real", "student"}


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AI/human reference text files for the embedding ensemble member."
    )
    parser.add_argument("input_csv", help="CSV with text and label columns.")
    parser.add_argument("--output-dir", default="models/references")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_texts: list[str] = []
    human_texts: list[str] = []

    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if args.text_column not in (reader.fieldnames or []):
            raise SystemExit(f"Missing text column: {args.text_column}")
        if args.label_column not in (reader.fieldnames or []):
            raise SystemExit(f"Missing label column: {args.label_column}")

        for row in reader:
            text = (row.get(args.text_column) or "").strip()
            label = normalize_label(row.get(args.label_column) or "")
            if not text:
                continue
            if label in AI_LABELS:
                ai_texts.append(text)
            elif label in HUMAN_LABELS:
                human_texts.append(text)

    if not ai_texts:
        raise SystemExit("No AI reference texts found.")
    if not human_texts:
        raise SystemExit("No human reference texts found.")

    (output_dir / "ai.txt").write_text("\n".join(ai_texts) + "\n", encoding="utf-8")
    (output_dir / "human.txt").write_text("\n".join(human_texts) + "\n", encoding="utf-8")
    print(f"Wrote {len(ai_texts)} AI texts to {output_dir / 'ai.txt'}")
    print(f"Wrote {len(human_texts)} human texts to {output_dir / 'human.txt'}")


if __name__ == "__main__":
    main()
