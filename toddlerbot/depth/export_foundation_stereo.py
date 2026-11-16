"""Export FoundationStereo to an ExecuTorch program (.pte) with a TensorRT engine inside.

The TensorRT engine in the .pte is built for the GPU and TensorRT version used here,
so run this script on the machine that will run the depth estimator.

Example:
    XFORMERS_DISABLED=1 python toddlerbot/depth/export_foundation_stereo.py \
        --fs-dir ~/FoundationStereo \
        --ckpt ~/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth \
        --height 96 --width 128 --valid-iters 16 \
        --output toddlerbot/depth/models/foundation_stereo_vits_96x128_16.pte
"""

import argparse
import os
import sys

import torch


class FoundationStereoForward(torch.nn.Module):
    """Takes left and right images and returns the disparity, like make_onnx.py."""

    def __init__(self, model: torch.nn.Module, valid_iters: int):
        super().__init__()
        self.model = model
        self.valid_iters = valid_iters

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model(left, right, iters=self.valid_iters, test_mode=True)


def main():
    """Export FoundationStereo to a .pte file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fs-dir",
        required=True,
        help="path to a FoundationStereo checkout with the changes named in README.md",
    )
    parser.add_argument(
        "--ckpt", required=True, help="path to model_best_bp2.pth (cfg.yaml beside it)"
    )
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--valid-iters", type=int, default=16)
    parser.add_argument("--output", required=True, help="path of the .pte to write")
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        parser.error("--height and --width must be multiples of 32")

    # FoundationStereo is a source checkout, not an installed package.
    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.fs_dir)))
    from core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf

    import torch_tensorrt

    ckpt_path = os.path.expanduser(args.ckpt)
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "cfg.yaml"))
    # The 23-51-11 checkpoint has no vit_size. FoundationStereo's own scripts use vitl then.
    if "vit_size" not in cfg:
        cfg.vit_size = "vitl"
    # Export in FP32, which matched PyTorch more closely than FP16.
    cfg.mixed_precision = False

    model = FoundationStereo(cfg)
    # The checkpoint also holds training state, so weights_only loading fails.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    module = FoundationStereoForward(model.cuda().eval(), args.valid_iters)

    example = tuple(
        torch.rand(1, 3, args.height, args.width, device="cuda") * 255 for _ in range(2)
    )
    with torch.no_grad():
        exported = torch.export.export(module, example)
        trt_module = torch_tensorrt.dynamo.compile(
            exported, inputs=list(example), min_block_size=1
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch_tensorrt.save(trt_module, args.output, output_format="executorch")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
