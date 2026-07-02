"""Precompute the unconditional (empty-prompt) T5 text embedding once.

Run this a single time while the T5 checkpoint is available. It saves
`uncond_context.pt` into the checkpoint dir. After that, running generate.py
for ti2v-5B *without* `--prompt` skips loading T5 entirely and reuses this
cached embedding for both CFG branches -> pure frame-only conditioning.
(Passing a `--prompt` still loads T5 and works as normal.)

Usage:
    python precompute_uncond_context.py --task ti2v-5B --ckpt_dir ../../ckpts/Wan2.2-TI2V-5B
"""
import argparse
import os

import torch

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ti2v-5B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", required=True,
                   help="Path to the model checkpoint directory.")
    p.add_argument("--prompt", default="",
                   help="Prompt to encode as the 'unconditional' embedding "
                        "(default: empty string = true null).")
    p.add_argument("--out", default=None,
                   help="Output path (default: <ckpt_dir>/uncond_context.pt).")
    args = p.parse_args()

    cfg = WAN_CONFIGS[args.task]
    out = args.out or os.path.join(args.ckpt_dir, "uncond_context.pt")

    te = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device="cpu",
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )

    uncond = te([args.prompt], torch.device("cpu"))[0]
    torch.save(uncond, out)
    print(f"Saved unconditional context: shape={tuple(uncond.shape)} "
          f"dtype={uncond.dtype} -> {out}")


if __name__ == "__main__":
    main()
