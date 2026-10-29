from pathlib import Path


def get_project_root_path() -> Path:
    """
    Get path to project root directory.
    """
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")
