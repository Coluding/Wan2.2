"""Precompute per-task T5 text embeddings once, so training and inference never
load T5.

Reads a prompts YAML mapping each MetaWorld ``task_name`` to a positive prompt,
encodes every prompt (plus a ``__default__`` fallback, the negative prompt, and
the empty-string unconditional) with T5 a single time, and saves a lookup table:

    {
      "positive": {task_name: Tensor[L, C], "__default__": Tensor[L, C]},
      "negative": Tensor[L, C],   # sample_neg_prompt (or the file's `negative`)
      "uncond":   Tensor[L, C],   # empty string (== uncond_context.pt)
      "prompts":  {task_name: str, "__default__": str, "__negative__": str},
      "text_len": cfg.text_len,
    }

The Wan training preprocessor (``PromptContextProvider``) maps a clip's
``task_name`` to ``positive[task_name]`` and feeds it as the base's ``context``;
inference uses ``positive`` / ``negative`` for real CFG via a cached-embedding
stub. Neither loads T5.

Prompts YAML format:

    default: "A robot arm performs a manipulation task, sharp realistic video."
    negative: null            # null -> use the model's sample_neg_prompt
    tasks:
      pick-place-v2: "A robot arm picks up the object and places it at the goal."
      door-open-v2:  "A robot arm opens the door by pulling the handle."

Usage:
    python precompute_prompt_contexts.py --task ti2v-5B \
        --ckpt_dir ../../ckpts/Wan2.2-TI2V-5B \
        --prompts_file ../../configs/prompts/metaworld_tasks.yaml
"""
import argparse
import os

import torch
import yaml

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def _default_out(prompts_file):
    root, _ = os.path.splitext(prompts_file)
    return root + ".contexts.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ti2v-5B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", required=True, help="Model checkpoint directory (holds the T5 weights).")
    p.add_argument("--prompts_file", required=True, help="YAML: {default, negative, tasks: {task_name: prompt}}.")
    p.add_argument("--out", default=None,
                   help="Output path (default: <prompts_file>.contexts.pt).")
    args = p.parse_args()

    cfg = WAN_CONFIGS[args.task]
    out = args.out or _default_out(args.prompts_file)

    with open(args.prompts_file) as f:
        spec = yaml.safe_load(f) or {}
    tasks = spec.get("tasks") or {}
    if not tasks:
        raise ValueError(f"{args.prompts_file}: no 'tasks' mapping found.")
    default_prompt = spec.get("default") or "A robot arm performs a manipulation task."
    negative_prompt = spec.get("negative") or cfg.sample_neg_prompt  # None/empty -> model default

    te = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device="cpu",
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )

    def encode(text):
        return te([text], torch.device("cpu"))[0]  # [L, C]

    positive = {name: encode(prompt) for name, prompt in tasks.items()}
    positive["__default__"] = encode(default_prompt)

    table = {
        "positive": positive,
        "negative": encode(negative_prompt),
        "uncond": encode(""),
        "prompts": {**{name: prompt for name, prompt in tasks.items()},
                    "__default__": default_prompt, "__negative__": negative_prompt},
        "text_len": cfg.text_len,
    }
    torch.save(table, out)

    a_key = next(iter(tasks))
    print(f"Encoded {len(tasks)} task prompts + default + negative + uncond.")
    print(f"  embedding shape (e.g. {a_key!r}): {tuple(positive[a_key].shape)} dtype={positive[a_key].dtype}")
    print(f"  negative: {negative_prompt[:60]!r}{'...' if len(negative_prompt) > 60 else ''}")
    print(f"Saved prompt-context table -> {out}")


if __name__ == "__main__":
    main()
