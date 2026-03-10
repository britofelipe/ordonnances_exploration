"""
Plot training loss + key metrics from HuggingFace Trainer 'trainer_state.json'.

Usage:
  python plot_trainer_state.py \
    --state ./toobib-ordo-bert2bert-prod-100000/trainer_state.json \
    --outdir ./plots \
    --ema 0.95 \
    --rolling 200

What it plots (if present in log_history):
  - train loss (raw + EMA + rolling mean)
  - eval loss
  - your main eval metric (auto-detect: eval_exact_match, eval_f1, etc.)
  - learning rate
  - grad_norm
  - steps/sec and samples/sec (if logged)
  - best checkpoint marker + best global step marker
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_state(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ema(values: List[float], alpha: float) -> List[float]:
    """EMA with decay alpha in [0,1). alpha closer to 1 => smoother."""
    if not values:
        return []
    out = []
    m = values[0]
    for v in values:
        m = alpha * m + (1 - alpha) * v
        out.append(m)
    return out


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def collect_series(log_history: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Return dict: key -> list of (step, epoch, value) sorted by step.
    """
    series = defaultdict(list)
    for rec in log_history:
        step = rec.get("step")
        epoch = rec.get("epoch")
        if step is None:
            continue
        # store all numeric fields except step/epoch
        for k, v in rec.items():
            if k in ("step", "epoch"):
                continue
            if isinstance(v, (int, float)):
                series[k].append((float(step), float(epoch) if epoch is not None else float("nan"), float(v)))

    # sort each series
    for k in list(series.keys()):
        series[k].sort(key=lambda t: t[0])
    return dict(series)


def pick_main_eval_metric(series: Dict[str, List[Tuple[float, float, float]]]) -> Optional[str]:
    """
    Heuristic to pick the most relevant eval metric:
      - prefer 'eval_exact_match'
      - else any 'eval_*' excluding eval_loss
      - else None
    """
    if "eval_exact_match" in series:
        return "eval_exact_match"
    eval_keys = [k for k in series.keys() if k.startswith("eval_") and k != "eval_loss"]
    # Prefer common names
    preferred = ["eval_accuracy", "eval_f1", "eval_bleu", "eval_rougeL", "eval_rouge", "eval_precision", "eval_recall"]
    for p in preferred:
        if p in eval_keys:
            return p
    return eval_keys[0] if eval_keys else None


def plot_xy(
    x: List[float],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    vlines: Optional[List[Tuple[float, str]]] = None,
    y2: Optional[List[float]] = None,
    y2label: Optional[str] = None,
):
    plt.figure()
    plt.plot(x, y, linewidth=1.2)
    if y2 is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(x, y2, linewidth=1.0, linestyle="--")
        ax2.set_ylabel(y2label if y2label else "y2")

    if vlines:
        for xv, lab in vlines:
            plt.axvline(x=xv, linestyle=":", linewidth=1.0)
            plt.text(xv, plt.ylim()[1], f" {lab}", rotation=90, va="top")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def unzip(triples: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float], List[float]]:
    steps, epochs, vals = [], [], []
    for s, e, v in triples:
        steps.append(s)
        epochs.append(e)
        vals.append(v)
    return steps, epochs, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="Path to trainer_state.json")
    ap.add_argument("--outdir", default="plots", help="Directory to save plots")
    ap.add_argument("--ema", type=float, default=0.95, help="EMA smoothing factor (0.0-0.999).")
    ap.add_argument("--rolling", type=int, default=0, help="Rolling mean window over steps (0 disables).")
    ap.add_argument("--xaxis", choices=["step", "epoch"], default="step", help="X axis for plots.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    state = load_state(args.state)
    log_history = state.get("log_history", [])
    if not log_history:
        raise SystemExit("No log_history found in trainer_state.json")

    series = collect_series(log_history)

    best_step = state.get("best_global_step", None)
    best_ckpt = state.get("best_model_checkpoint", None)
    vlines = []
    if best_step is not None:
        vlines.append((float(best_step), "best_step"))
    if best_ckpt:
        # annotate checkpoint number if present
        vlines.append((float(best_step) if best_step is not None else 0.0, "best_ckpt"))

    # --- TRAIN LOSS ---
    if "loss" in series:
        steps, epochs, losses = unzip(series["loss"])
        x = steps if args.xaxis == "step" else epochs

        loss_ema = ema(losses, alpha=args.ema) if 0.0 <= args.ema < 1.0 else losses[:]
        loss_roll = rolling_mean(losses, args.rolling) if args.rolling and args.rolling > 1 else None

        # plot raw + EMA (and rolling if requested)
        plt.figure()
        plt.plot(x, losses, linewidth=0.8, alpha=0.5, label="train_loss (raw)")
        plt.plot(x, loss_ema, linewidth=1.3, label=f"train_loss (EMA α={args.ema})")
        if loss_roll is not None:
            plt.plot(x, loss_roll, linewidth=1.1, linestyle="--", label=f"train_loss (rolling {args.rolling})")

        for xv, lab in vlines:
            plt.axvline(x=(xv if args.xaxis == "step" else (epochs[steps.index(xv)] if xv in steps else xv)),
                        linestyle=":", linewidth=1.0)
        plt.title("Training Loss over Fine-tuning")
        plt.xlabel(args.xaxis)
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"train_loss_{args.xaxis}.png"), dpi=180)
        plt.close()

    # --- EVAL LOSS ---
    if "eval_loss" in series:
        steps, epochs, vals = unzip(series["eval_loss"])
        x = steps if args.xaxis == "step" else epochs
        plot_xy(
            x=x,
            y=vals,
            title="Eval Loss over Fine-tuning",
            xlabel=args.xaxis,
            ylabel="eval_loss",
            outpath=os.path.join(args.outdir, f"eval_loss_{args.xaxis}.png"),
            vlines=vlines if args.xaxis == "step" else None,
        )

    # --- MAIN EVAL METRIC ---
    main_eval = pick_main_eval_metric(series)
    if main_eval:
        steps, epochs, vals = unzip(series[main_eval])
        x = steps if args.xaxis == "step" else epochs
        plot_xy(
            x=x,
            y=vals,
            title=f"{main_eval} over Fine-tuning",
            xlabel=args.xaxis,
            ylabel=main_eval,
            outpath=os.path.join(args.outdir, f"{main_eval}_{args.xaxis}.png"),
            vlines=vlines if args.xaxis == "step" else None,
        )

    # --- LR + grad_norm (good sanity plots) ---
    if "learning_rate" in series:
        steps, epochs, lr = unzip(series["learning_rate"])
        x = steps if args.xaxis == "step" else epochs
        plot_xy(
            x=x,
            y=lr,
            title="Learning Rate Schedule",
            xlabel=args.xaxis,
            ylabel="learning_rate",
            outpath=os.path.join(args.outdir, f"learning_rate_{args.xaxis}.png"),
        )

    if "grad_norm" in series:
        steps, epochs, gn = unzip(series["grad_norm"])
        x = steps if args.xaxis == "step" else epochs
        plot_xy(
            x=x,
            y=gn,
            title="Gradient Norm (sanity check)",
            xlabel=args.xaxis,
            ylabel="grad_norm",
            outpath=os.path.join(args.outdir, f"grad_norm_{args.xaxis}.png"),
        )

    # --- Throughput metrics if present ---
    for k in ["train_runtime", "train_steps_per_second", "train_samples_per_second",
              "eval_runtime", "eval_steps_per_second", "eval_samples_per_second"]:
        if k in series:
            steps, epochs, vals = unzip(series[k])
            x = steps if args.xaxis == "step" else epochs
            plot_xy(
                x=x,
                y=vals,
                title=f"{k} over time",
                xlabel=args.xaxis,
                ylabel=k,
                outpath=os.path.join(args.outdir, f"{k}_{args.xaxis}.png"),
            )

    # --- Quick terminal summary ---
    print("Saved plots to:", os.path.abspath(args.outdir))
    print("Detected keys:", ", ".join(sorted(series.keys())))
    if main_eval:
        print("Main eval metric chosen:", main_eval)
    if best_ckpt:
        print("Best checkpoint:", best_ckpt)
    if best_step is not None:
        print("Best global step:", best_step, "best_metric:", state.get("best_metric"))


if __name__ == "__main__":
    main()