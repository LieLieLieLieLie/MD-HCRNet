"""Dataset and MANO model setup helpers.

Usage from the project code directory:
    python -m data.download freihand --root data
    python -m data.download mano --root data
    python -m data.download all --root data
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


FREIHAND_URLS = {
    "FreiHAND_pub_v2.zip": (
        "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
    ),
    "FreiHAND_pub_v2_eval.zip": (
        "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip"
    ),
}


def _run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _download(url: str, dst: Path):
    if dst.exists():
        print(f"  Found {dst}, skipping download.")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  Downloading {url}")
    print(f"  -> {dst}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_mb = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            mb = done // (100 * 1024 * 1024)
            if mb != last_mb:
                last_mb = mb
                if total:
                    print(f"    {done / total:.1%} ({done / 1024**3:.2f} GiB)")
                else:
                    print(f"    {done / 1024**3:.2f} GiB")
    tmp.replace(dst)


def _extract(archive: Path, dst: Path):
    print(f"  Extracting {archive} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dst)
    else:
        _run(["tar", "xf", str(archive), "-C", str(dst)])


def _copy_if_exists(src: Path, dst: Path):
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _find_file(root: Path, name: str):
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def setup_freihand(root: str, download: bool = True):
    """Set up FreiHAND under <root>/freihand.

    The loader expects:
        <root>/freihand/training/rgb/*.jpg
        <root>/freihand/evaluation/rgb/*.jpg
        <root>/freihand/training_K.json
        <root>/freihand/training_xyz.json
        <root>/freihand/training_mano.json

    Current official archives:
        FreiHAND_pub_v2.zip
        FreiHAND_pub_v2_eval.zip

    Legacy archive names are also supported if they have already been placed
    in <root>/freihand.
    """
    base = Path(root)
    freihand = base / "freihand"
    freihand.mkdir(parents=True, exist_ok=True)

    if download:
        for fname, url in FREIHAND_URLS.items():
            _download(url, freihand / fname)

    extracted_any = False
    v2_train = freihand / "FreiHAND_pub_v2.zip"
    v2_eval = freihand / "FreiHAND_pub_v2_eval.zip"

    if v2_train.exists():
        _extract(v2_train, freihand)
        extracted_any = True
    if v2_eval.exists():
        _extract(v2_eval, freihand)
        extracted_any = True

    legacy_zips = {
        "freihand_training_rgb.zip": freihand / "training" / "rgb",
        "freihand_training_mano.zip": freihand,
        "freihand_evaluation_rgb.zip": freihand / "evaluation" / "rgb",
    }
    for fname, dst in legacy_zips.items():
        archive = freihand / fname
        if archive.exists():
            _extract(archive, dst)
            extracted_any = True

    required = [
        freihand / "training" / "rgb",
        freihand / "evaluation" / "rgb",
        freihand / "training_K.json",
        freihand / "training_xyz.json",
        freihand / "training_mano.json",
    ]
    missing = [p for p in required if not p.exists()]

    if missing:
        print("[FreiHAND] Setup is incomplete. Missing:")
        for path in missing:
            print(f"  - {path}")
        print("[FreiHAND] Download page:")
        print("  https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html")
        if not extracted_any:
            print("[FreiHAND] Place the zip archives in:", freihand)
        return False

    print("[FreiHAND] Setup complete:", freihand)
    return True



def setup_mano(root: str):
    """Set up MANO under <root>/mano after downloading from the MANO website."""
    mano = Path(root) / "mano"
    mano.mkdir(parents=True, exist_ok=True)

    dst = mano / "MANO_RIGHT.pkl"
    if dst.exists():
        print("[MANO] Setup complete:", dst)
        return True

    for archive_name in ("mano_v1_2.zip", "MANO.zip"):
        archive = mano / archive_name
        if archive.exists():
            _extract(archive, mano)
            src = _find_file(mano, "MANO_RIGHT.pkl")
            if src:
                _copy_if_exists(src, dst)
                print("[MANO] Setup complete:", dst)
                return True

    src = _find_file(mano, "MANO_RIGHT.pkl")
    if src:
        _copy_if_exists(src, dst)
        print("[MANO] Setup complete:", dst)
        return True

    print("[MANO] MANO_RIGHT.pkl not found in:", mano)
    print("[MANO] Please register/login and download from:")
    print("  https://mano.is.tue.mpg.de/")
    print("[MANO] Then place either MANO_RIGHT.pkl or mano_v1_2.zip in:", mano)
    return False


def main():
    parser = argparse.ArgumentParser(description="Dataset / model setup utility")
    parser.add_argument("dataset", choices=["freihand", "mano", "all"])
    parser.add_argument("--root", default="data", help="Base data directory")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="For FreiHAND, only extract local zip files and do not download.",
    )
    args = parser.parse_args()

    handlers = {
        "freihand": lambda: setup_freihand(args.root, download=not args.no_download),
        "mano": lambda: setup_mano(args.root),
    }

    if args.dataset == "all":
        ok = True
        for name, fn in handlers.items():
            print(f"\n== {name.upper()} ==")
            ok = fn() and ok
    else:
        ok = handlers[args.dataset]()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
