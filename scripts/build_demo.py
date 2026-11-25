"""Build GitHub Pages from the same template and assets used by Django."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build():
    destination = ROOT / "docs"
    destination.mkdir(exist_ok=True)
    source = (ROOT / "templates/PigPredict_app/studio.html").read_text()
    source = source.replace("{% load static %}\n", "")
    source = re.sub(r"\{% static 'studio/(.*?)' %\}", r"./assets/\1", source)
    if "{%" in source or "{{" in source:
        raise ValueError("Unresolved template directives in static demo")
    (destination / "index.html").write_text(source)
    shutil.copytree(ROOT / "static/studio", destination / "assets", dirs_exist_ok=True)
    (destination / ".nojekyll").touch()
    print(f"Static demo built: {destination / 'index.html'}")


if __name__ == "__main__":
    build()
