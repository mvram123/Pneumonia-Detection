import os

dirs = [
    os.path.join("data", "train"),
    os.path.join("data", "test"),
    "notebooks",
    "models",
    "reports",
    "stages",

]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = [
    "params.yaml",
    ".gitignore",
    "app.py",
    "requirements.txt",
]

for file_ in files:
    with open(file_, "w") as f:
        pass
