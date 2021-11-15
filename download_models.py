import gdown
from pathlib import Path

path = Path("models")
path.mkdir(exist_ok=True)

# download model trained for Part 1
gdown.download(
    "https://drive.google.com/uc?id=1UyLNp68kYoZglzfyDOduEPzXcS6sJwR4",
    str(path / "exported_resnext50_32x4d.pickle"),
)
