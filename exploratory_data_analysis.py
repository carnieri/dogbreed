from collections import Counter
from pathlib import Path
from pprint import pprint

import cv2
import matplotlib.pyplot as plt


def summarize_resolutions(path):
    path = Path(path)
    sizes = Counter()
    aspect_ratios = Counter()
    aspect_ratio_list = []
    for p in path.glob("**/*.jpg"):
        img = cv2.imread(str(p))
        size = img.shape
        sizes[size] += 1

        h, w = size[:2]
        aspect_ratio = h / w
        aspect_ratios[aspect_ratio] += 1
        aspect_ratio_list.append(aspect_ratio)
    print("")
    print(f"resolutions seen in {path}:")
    pprint(sizes)
    print("")
    print(f"aspect ratios seen in {path}:")
    pprint(aspect_ratios)

    plt.hist(aspect_ratio_list, bins=100)
    plt.xlabel("aspect ratio (h/w)")
    plt.ylabel("frequency")
    plt.title("Aspect ratio histogram")
    plt.savefig("aspect_ratio_histogram.png")
    plt.show()


if __name__ == "__main__":
    summarize_resolutions("dogs/train")
    summarize_resolutions("dogs/recognition/enroll")
    summarize_resolutions("dogs/recognition/test")
