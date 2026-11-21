#!/usr/bin/env python3
"""Download the paper's networks (and TinyImageNet) and check them against networks.csv.

    python reproduce/download.py --nets nets                  # every downloadable network
    python reproduce/download.py --nets nets --resnet50       # + N40, exported from the MadryLab checkpoint
    python reproduce/download.py --nets nets --vgg16 --mxnet-python /path/to/python   # + N54's VNN-LIB specs
    python reproduce/download.py --tinyimagenet               # TinyImageNet into ./tinyimagenet (for N38)

MNIST, CIFAR-10 and CIFAR-100 are downloaded by torchvision into the working directory on first use.
Files already present with the right checksum are skipped.
"""
import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# N40: MadryLab robustness CIFAR10 ResNet50, Linf eps 8/255 (https://github.com/MadryLab/robustness)
RESNET50_CKPT = "https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=1"
RESNET50_CKPT_MD5 = "04df96d97be7bd86e605f988369a7234"
ROBUSTNESS = ("https://github.com/MadryLab/robustness.git", "a954124")
# N54: VNN-COMP's vggnet16_2022 generator, patched to perturb the whole image by +-7.5e-5 (vggnet16_linf.patch)
VGG_GENERATOR = ("https://github.com/stanleybak/vggnet16_benchmark2022.git", "c1eece7")
VGG_SEED, VGG_EPS = "157191271", "7.5e-5"
TINYIMAGENET = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(path + ".part", path)


def check(path, expected):
    got = md5(path)
    if got != expected:
        sys.exit(f"checksum mismatch for {path}: {got}, expected {expected}")


def clone(repo, commit, dest):
    subprocess.run(["git", "clone", "-q", repo, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "-q", commit], check=True)


def networks():
    with open(os.path.join(HERE, "networks.csv")) as f:
        return list(csv.DictReader(f))


def download_networks(nets, only):
    for row in networks():
        if only and row["config"] not in only or row["source"] != "download":
            continue
        path = os.path.join(nets, row["path"])
        if not (os.path.exists(path) and md5(path) == row["md5"]):
            fetch(row["url"], path)
            check(path, row["md5"])
        print(f"ok  {row['config']:4s} {row['path']}")


def resnet50(nets):
    (row,) = [r for r in networks() if r["source"] == "resnet50"]
    out = os.path.join(nets, row["path"])
    if os.path.exists(out) and md5(out) == row["md5"]:
        print(f"ok  {row['config']:4s} {row['path']}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "cifar_linf_8.pt")
        fetch(RESNET50_CKPT, ckpt)
        check(ckpt, RESNET50_CKPT_MD5)
        clone(*ROBUSTNESS, os.path.join(tmp, "robustness"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([sys.executable, os.path.join(HERE, "export_resnet50.py"), os.path.join(tmp, "robustness"),
                        ckpt, out], check=True)
    check(out, row["md5"])
    print(f"ok  {row['config']:4s} {row['path']}")


def vgg16(nets, mxnet_python):
    """N54's specs: run the patched generator (needs mxnet 1.9.1, numpy 1.23.5, onnx, onnxruntime)."""
    (row,) = [r for r in networks() if r["config"] == "N54"]
    folder = os.path.join(nets, os.path.dirname(os.path.dirname(row["path"])))
    with open(os.path.join(HERE, "vggnet16_linf_specs.md5")) as f:
        expected = dict(line.split()[::-1] for line in f if line.strip())
    if all(os.path.exists(os.path.join(folder, p)) and md5(os.path.join(folder, p)) == h for p, h in expected.items()):
        print(f"ok  N54  {folder} specs")
        return
    onnx = os.path.join(nets, row["path"])  # downloaded by download_networks
    check(onnx, row["md5"])
    with tempfile.TemporaryDirectory() as tmp:
        gen = os.path.join(tmp, "generator")
        clone(*VGG_GENERATOR, gen)
        subprocess.run(["git", "-C", gen, "apply", os.path.join(HERE, "vggnet16_linf.patch")], check=True)
        os.makedirs(os.path.join(gen, "onnx"), exist_ok=True)
        shutil.copy(onnx, os.path.join(gen, "onnx", "vgg16-7.onnx"))
        subprocess.run([mxnet_python, "generate_properties.py", VGG_SEED, VGG_EPS], cwd=gen, check=True)
        for p in expected:
            os.makedirs(os.path.dirname(os.path.join(folder, p)), exist_ok=True)
            shutil.copy(os.path.join(gen, p), os.path.join(folder, p))
    for p, h in expected.items():
        check(os.path.join(folder, p), h)
    print(f"ok  N54  {folder} specs")


def tinyimagenet(root):
    if os.path.exists(os.path.join(root, "tiny-imagenet-200", "val", "val_annotations.txt")):
        print(f"ok  TinyImageNet in {root}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "tiny-imagenet-200.zip")
        fetch(TINYIMAGENET, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
    print(f"ok  TinyImageNet in {root}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", default="nets", help="folder for the networks (default: ./nets)")
    p.add_argument("--only", default="", help="comma list of network IDs, e.g. N1,N16 (default: all)")
    p.add_argument("--resnet50", action="store_true", help="also build N40 from the MadryLab checkpoint")
    p.add_argument("--vgg16", action="store_true", help="also generate N54's VNN-LIB specs")
    p.add_argument("--mxnet-python", help="python with mxnet 1.9.1, numpy 1.23.5, onnx and onnxruntime (for --vgg16)")
    p.add_argument("--tinyimagenet", action="store_true", help="also download TinyImageNet into ./tinyimagenet")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the subprocesses' output
    only = set(filter(None, args.only.split(",")))
    download_networks(args.nets, only)
    if args.resnet50:
        resnet50(args.nets)
    if args.vgg16:
        if not args.mxnet_python:
            p.error("--vgg16 needs --mxnet-python")
        vgg16(args.nets, args.mxnet_python)
    if args.tinyimagenet:
        tinyimagenet("tinyimagenet")


if __name__ == "__main__":
    main()
