"""Run CoT generation locally (API calls only, no GPU needed)."""
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "model"))

from scripts.run_cot_generation import load_cases, CoTGenerator
from loguru import logger
import json

logger.remove()
logger.add(sys.stderr, level="INFO")

cases_dir = str(Path(__file__).parent / "data_pipeline" / "data" / "processed" / "extracted_cases")
output_dir = str(Path(__file__).parent / "model" / "outputs" / "cot")

cases = load_cases(cases_dir)
logger.info(f"Loaded {len(cases)} cases")

generator = CoTGenerator(
    model="gemini-2.5-flash",
    output_dir=output_dir,
    rpm_limit=10,
)

# Generate CoTs
results = generator.batch_generate(cases)
logger.info(f"Generated {len(results)} CoT entries")

# Evaluate quality
logger.info("Evaluating quality...")
results = generator.batch_evaluate(results)

# Summary
summary = generator.get_summary(results)
logger.info(f"\nSummary: {json.dumps(summary, indent=2)}")

summary_path = Path(output_dir) / "cot_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Done! Results in {output_dir}")
