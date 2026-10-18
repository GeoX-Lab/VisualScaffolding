"""Run a model on NMR-Bench or score existing predictions."""

import argparse
import json
from pathlib import Path

from .benchmark import evaluate_predictions, load_adapter, load_samples, read_predictions, run_model
from .scoring import TASKS


def main():
    parser = argparse.ArgumentParser(prog="python -m visual_scaffolding")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--task", choices=TASKS, required=True)
        sub.add_argument("--benchmark", type=Path, default=Path("benchmark"))
        sub.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Prediction JSONL for evaluate, summary JSON for score",
        )
        if name == "evaluate":
            sub.add_argument("--checkpoint", type=Path)
            sub.add_argument("--adapter", help="Custom model factory as module:function")
            sub.add_argument(
                "--model-config", type=Path, help="JSON object passed to the adapter factory"
            )
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--max-new-tokens", type=int, default=16384)
            sub.add_argument("--num-shards", type=int, default=1)
            sub.add_argument("--shard-index", type=int, default=0)
        else:
            sub.add_argument("--predictions", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    samples = list(load_samples(args.benchmark, args.task))
    if args.command == "score":
        summary, _ = evaluate_predictions(samples, read_predictions(args.predictions), args.task)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        print(json.dumps(summary, indent=2))
        return
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.num_shards == 1 and args.output.with_suffix(".summary.json").exists():
        parser.error("The output summary already exists")
    if not 1 <= args.num_shards <= len(samples) or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard count or index")
    if args.max_new_tokens < 0:
        parser.error("max-new-tokens must be nonnegative")
    if args.adapter:
        if args.checkpoint:
            parser.error("Use model-config to pass custom checkpoint settings to an adapter")
        config = json.loads(args.model_config.read_text()) if args.model_config else {}
        if not isinstance(config, dict):
            parser.error("model-config must contain a JSON object")
        config.setdefault("device", args.device)
        config.setdefault("max_new_tokens", args.max_new_tokens)
        model = load_adapter(args.adapter, config)
    else:
        if not args.checkpoint:
            parser.error("Supply --checkpoint or --adapter")
        if args.model_config:
            parser.error("model-config is used with custom adapters")
        info = json.loads((args.checkpoint / "model_info.json").read_text())
        if info["task"] != args.task:
            parser.error("The checkpoint was trained for a different benchmark task")
        from .inference import VisualScaffolding

        model = VisualScaffolding(args.checkpoint, args.device, args.max_new_tokens)
    owned = samples[args.shard_index :: args.num_shards]
    run_model(model, owned, args.output)
    if args.num_shards == 1:
        summary, _ = evaluate_predictions(samples, read_predictions([args.output]), args.task)
        target = args.output.with_suffix(".summary.json")
        with target.open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
