"""Remove totally black images and duplicates from dataset."""

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


def remove_black_images(path):
    path = Path(path)
    for p in path.glob("**/*.jpg"):
        img = cv2.imread(str(p))
        if np.alltrue(img == 0):
            print(f"{p} is totally black, deleting")
            os.remove(p)


def remove_duplicates(path):
    path = Path(path)
    hash_dict = {}
    for p in path.glob("**/*.jpg"):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            buffer = f.read()
            h.update(buffer)
        digest = h.hexdigest()
        if digest in hash_dict:
            other_path = hash_dict[digest]
            print(f"duplicate hash: {digest} {p} {other_path}")
            if p.parent.name == other_path.parent.name:
                # duplicate is in the same directory (same breed)
                # it can safely be deleted
                print(f"deleting file {p}")
                os.remove(p)
        else:
            hash_dict[digest] = p


def main():
    remove_black_images(Path("dogs/train"))
    remove_duplicates(Path("dogs/train"))


if __name__ == "__main__":
    main()
