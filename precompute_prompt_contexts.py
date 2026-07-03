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
    pool_prompts = spec.get("prompts")  # flat, task-independent set (sampled per clip)
    if not tasks and not pool_prompts:
        raise ValueError(
            f"{args.prompts_file}: need a 'prompts' list (task-independent pool) or a "
            "'tasks' mapping (per-task prompts)."
        )
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

    def _as_prompt_list(value):
        """A task/default value may be one prompt (str) or a set of paraphrases
        (list of str). Normalise to a non-empty list of strings."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)) and value:
            return [str(v) for v in value]
        raise ValueError(f"prompt value must be a string or a non-empty list, got {value!r}")

    def encode_prompts(value):
        """Single str -> one [L,C] tensor; list of str -> list of [Lᵢ,C] tensors
        (the provider samples one per clip during training)."""
        prompts = _as_prompt_list(value)
        embs = [encode(p) for p in prompts]
        return embs[0] if len(embs) == 1 else embs

    positive = {name: encode_prompts(prompt) for name, prompt in tasks.items()}
    positive["__default__"] = encode_prompts(default_prompt)

    table = {
        "positive": positive,
        "negative": encode(negative_prompt),
        "uncond": encode(""),
        "prompts": {**{name: _as_prompt_list(prompt) for name, prompt in tasks.items()},
                    "__default__": _as_prompt_list(default_prompt), "__negative__": negative_prompt},
        "text_len": cfg.text_len,
    }

    # Task-independent pool: `default` first (deterministic eval), then the `prompts`
    # set. Training samples one per clip; pool takes precedence over `positive`.
    if pool_prompts:
        pool_texts = [default_prompt] + _as_prompt_list(pool_prompts) if spec.get("default") \
            else _as_prompt_list(pool_prompts)
        table["pool"] = [encode(p) for p in pool_texts]
        table["prompts"]["__pool__"] = pool_texts

    torch.save(table, out)

    if pool_prompts:
        pool_texts = table["prompts"]["__pool__"]
        print(f"Encoded a task-independent pool of {len(pool_texts)} prompts (eval uses #0) "
              f"+ negative + uncond. Shape {tuple(table['pool'][0].shape)}.")
        print(f"  #0 (eval): {pool_texts[0][:70]!r}{'...' if len(pool_texts[0]) > 70 else ''}")
    else:
        a_key = next(iter(tasks))
        n_prompts = sum(len(_as_prompt_list(p)) for p in tasks.values())
        a_emb = positive[a_key]
        a_shape = tuple(a_emb[0].shape) if isinstance(a_emb, list) else tuple(a_emb.shape)
        a_count = len(a_emb) if isinstance(a_emb, list) else 1
        print(f"Encoded {n_prompts} task prompts across {len(tasks)} tasks + default + negative + uncond.")
        print(f"  {a_key!r}: {a_count} prompt(s), embedding shape {a_shape} dtype={(a_emb[0] if isinstance(a_emb, list) else a_emb).dtype}")
    print(f"  negative: {negative_prompt[:60]!r}{'...' if len(negative_prompt) > 60 else ''}")
    print(f"Saved prompt-context table -> {out}")


if __name__ == "__main__":
    main()
