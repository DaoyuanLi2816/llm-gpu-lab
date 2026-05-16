"""Optional bridge to EleutherAI's lm-evaluation-harness.

This file purposely defers the heavy `lm_eval` import. We only import on demand
so the rest of the package keeps working when the optional dependency is not
installed.

Usage:

```bash
pip install lm-eval
python -m llm_gpu_lab lm-eval --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --tasks arc_easy --limit 20 --out results/rtx4080/lm_eval_results.json
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)


def run_lm_eval(
    base_model: str,
    tasks: List[str],
    limit: Optional[int] = None,
    out_path: str | Path = "results/rtx4080/lm_eval_results.json",
    adapter_path: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Run lm-eval-harness with HuggingFace LM and persist results JSON."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError(
            "lm-evaluation-harness is not installed. Run: pip install lm-eval"
        ) from exc

    model_args: Dict[str, Any] = {"pretrained": base_model}
    if adapter_path:
        model_args["peft"] = str(ROOT / adapter_path)
    lm = HFLM(**model_args)

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )
    out = ROOT / out_path if not Path(out_path).is_absolute() else Path(out_path)
    ensure_dir(out.parent)
    # Convert numpy types so json.dump succeeds
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("lm-eval results written to %s", out)
    return results
