"""Convert Meta's DINOv3 ViT-L checkpoint (.pth, from the download links Meta emails) to
the Hugging Face transformers format TRELLIS.2 loads, and optionally upload it to a
private Hugging Face repo. Then set `dino_model = "<that repo>"` in a kami.toml profile.

Use it when Hugging Face's gated facebook/dinov3-vitl16-pretrain-lvd1689m is not open to
your account but Meta's direct download is. Runs on the CPU; needs torch, transformers
(4.56+) and huggingface_hub, not kami:

    python -m venv .venv-dino
    .venv-dino/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    .venv-dino/Scripts/python -m pip install "transformers>=4.56" safetensors huggingface_hub
    .venv-dino/Scripts/python tools/convert_dinov3.py dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ^
        --out converted --repo <you>/dinov3-vitl16 --license LICENSE.md

transformers' own converter does the work and checks the converted model's output
against Meta's reference values, so a bad file fails here, not on a billed pod.
The upload needs a Hugging Face token with write access (HF_TOKEN or `hf auth login`).
The repo is created private: keep it so (the DINOv3 license governs redistribution).
"""
import argparse
import shutil
import sys
from pathlib import Path

MODEL = "vitl16_lvd1689m"     # the one TRELLIS.2's pipeline.json names


# transformers' converter at a fixed commit of its main branch: releases (and their tags)
# leave conversion scripts out. Checked with transformers 5.23.
CONVERTER_URL = ("https://raw.githubusercontent.com/huggingface/transformers/"
                 "f262c71703eb138381d65ced86b3aaa496be0196/"
                 "src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py")


def load_converter(path=None):
    """transformers' DINOv3 converter: `path` (a copy you checked), or fetched from
    CONVERTER_URL. It also needs torchvision."""
    import importlib.util
    import tempfile
    import urllib.request
    if path is None:
        print(f"fetching the converter: {CONVERTER_URL}")
        path = Path(tempfile.mkdtemp()) / "convert_dinov3_vit_to_hf.py"
        with urllib.request.urlopen(CONVERTER_URL, timeout=60) as response:
            path.write_bytes(response.read())
    spec = importlib.util.spec_from_file_location("convert_dinov3_vit_to_hf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="Meta's dinov3_vitl16_pretrain_lvd1689m-*.pth")
    parser.add_argument("--out", type=Path, default=Path("converted"), help="output directory")
    parser.add_argument("--repo", help="upload to this Hugging Face repo (created private)")
    parser.add_argument("--license", type=Path, help="DINOv3 license file to store with the weights")
    parser.add_argument("--converter", type=Path,
                        help="local copy of transformers' convert_dinov3_vit_to_hf.py "
                             "(default: fetched from a pinned commit)")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        sys.exit(f"not found: {args.checkpoint}")

    import huggingface_hub
    convert = load_converter(args.converter)

    # the converter downloads the .pth from the gated repo: hand it the local file instead
    local = str(args.checkpoint.resolve())
    convert.hf_hub_download = lambda repo_id, filename, **_: local
    # Meta saved the checkpoint from CUDA tensors: load them on the CPU, GPU or not
    torch_load = convert.torch.load
    convert.torch.load = lambda *a, **kw: torch_load(*a, **{**kw, "map_location": "cpu"})
    convert.convert_and_test_dinov3_checkpoint(
        argparse.Namespace(model_name=MODEL, save_dir=str(args.out), push_to_hub=False))
    folder = args.out / MODEL
    if args.license:
        shutil.copy(args.license, folder / "LICENSE.md")

    if args.repo:
        api = huggingface_hub.HfApi()
        api.create_repo(args.repo, private=True, exist_ok=True)
        if not api.repo_info(args.repo).private:
            sys.exit(f"{args.repo} exists and is public: not uploading DINOv3 weights there")
        api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                          commit_message="DINOv3 ViT-L/16 LVD-1689M, converted from Meta's checkpoint")
        print(f"uploaded to https://huggingface.co/{args.repo} (private)")
        print(f'in kami.toml: dino_model = "{args.repo}"')
    else:
        print(f"converted model in {folder}")


if __name__ == "__main__":
    main()
