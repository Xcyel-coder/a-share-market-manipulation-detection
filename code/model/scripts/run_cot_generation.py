"""
CoT generation entry point.

Usage:
    # Generate CoT for all cases
    GOOGLE_API_KEY=xxx python -m scripts.run_cot_generation --cases-dir /path/to/extracted_cases

    # Evaluate quality only (if CoTs already generated)
    GOOGLE_API_KEY=xxx python -m scripts.run_cot_generation --evaluate-only --output-dir outputs/cot
"""
import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher.cot_generator import CoTGenerator


def load_cases(cases_dir: str) -> list[dict]:
    """Load all extracted case JSON files."""
    cases_path = Path(cases_dir)
    cases = []
    for f in sorted(cases_path.glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            cases.append(case)
    logger.info(f"Loaded {len(cases)} cases from {cases_dir}")
    return cases


def main():
    parser = argparse.ArgumentParser(description="CoT Generation")
    parser.add_argument("--cases-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/cot")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--rpm-limit", type=int, default=10)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_file = Path(args.output_dir) / "cot_generation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="DEBUG")

    generator = CoTGenerator(
        model=args.model,
        output_dir=args.output_dir,
        rpm_limit=args.rpm_limit,
    )

    if args.evaluate_only:
        results_path = Path(args.output_dir) / "cot_results.json"
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing CoT entries for evaluation")
    else:
        cases = load_cases(args.cases_dir)
        results = generator.batch_generate(cases)
        logger.info(f"Generated {len(results)} CoT entries")

    if not args.skip_evaluation:
        logger.info("Evaluating CoT quality...")
        results = generator.batch_evaluate(results)

    summary = generator.get_summary(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"CoT Generation Summary:")
    logger.info(f"  Total generated: {summary['total_generated']}")
    logger.info(f"  Errors:          {summary['errors']}")
    logger.info(f"  Evaluated:       {summary['evaluated']}")
    logger.info(f"  Avg quality:     {summary['avg_quality']:.2f}")
    logger.info(f"  Above 3.5:       {summary['above_3_5']}")
    logger.info(f"{'='*60}")

    summary_path = Path(args.output_dir) / "cot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
