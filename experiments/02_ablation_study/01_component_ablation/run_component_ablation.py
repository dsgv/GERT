"""Generate the four GERT component variants for one benchmark dataset.

Outputs live in predictions_new so archived paper predictions stay untouched.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_DIR = HERE.parents[1] / "01_main_benchmark"
REPO_ROOT = HERE.parents[2]
for path in (str(REPO_ROOT), str(MAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_main_experiments import (  # noqa: E402
    BaseSchemaLinkingRetrieverPPR,
    CachedRetryEmbedder,
    DATASETS,
    build_knowledge_graph,
    driver,
    verify_graph,
)

VARIANTS = {
    "full": (0.7, 0.3),
    "without_graph": (0.0, 0.3),
    "without_semantic": (1.0, 0.3),
    "without_seed_decay": (0.7, 0.0),
}


def run_dataset(dataset: str, force: bool) -> None:
    config = DATASETS[dataset]
    output_dir = HERE / "predictions_new" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{name}.json" for name in VARIANTS]
    if not force and any(path.exists() for path in existing):
        raise FileExistsError(
            f"Predictions already exist in {output_dir}; choose another directory "
            "or pass --force to replace this new-run output."
        )

    build_knowledge_graph(str(config["schema"]), clear_before_build=True)
    verify_graph(config["schema"])
    retriever = BaseSchemaLinkingRetrieverPPR(embedder=CachedRetryEmbedder())
    for name, (fusion_weight, seed_rank_decay) in VARIANTS.items():
        target = output_dir / f"{name}.json"
        retriever.process_dataset(
            str(config["test"]),
            str(target),
            top_k=20,
            verbose=False,
            initial_top_k=20,
            alpha=0.85,
            seed_rank_decay=seed_rank_decay,
            fusion_weight=fusion_weight,
        )
        print(f"[{dataset}] {name}: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        run_dataset(args.dataset, args.force)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
