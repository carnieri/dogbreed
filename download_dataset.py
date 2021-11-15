import shutil
import gdown


def main():
    gdown.download(
        "https://drive.google.com/u/0/uc?id=1Xrr_C0ho9UpOarWBluK4pTY1ps92EqxR",
        "dogs.zip",
    )

    shutil.unpack_archive("dogs.zip")


if __name__ == "__main__":
    main()
