import os

def tree(dir_path, prefix="", ignore={".git", ".venv", "__pycache__", ".idea", "*.pyc"}):
    entries = sorted(os.listdir(dir_path))
    entries = [e for e in entries if not any(e.endswith(i.strip("*")) for i in ignore)]
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = i == len(entries) - 1
        current = "└── " if is_last else "├── "
        print(f"{prefix}{current}{entry}")
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            tree(path, prefix + extension, ignore)

if __name__ == "__main__":
    tree(".")