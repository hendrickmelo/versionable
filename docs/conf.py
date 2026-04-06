from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

project = "versionable"
author = "Hendrick Melo"
try:
    from importlib.metadata import version as _getVersion

    release = _getVersion("versionable")
except Exception:
    release = "dev"

extensions = ["myst_parser"]

exclude_patterns = ["_build", "plans/**"]

html_favicon = "images/favicon-192.png"
html_theme = "furo"
html_static_path = ["_static", "images"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-stacked-above.svg",
    "dark_logo": "logo-stacked-above.svg",
}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

_PROJECT_SPAN = '<span class="project-name">versionable</span>'


def _replaceProjectName(*args: object) -> None:
    """Replace **`versionable`** with a styled HTML span before parsing."""
    source: list[str] = args[-1]  # type: ignore[assignment]
    source[0] = source[0].replace("**`versionable`**", _PROJECT_SPAN)


def setup(app: Sphinx) -> None:
    app.connect("source-read", _replaceProjectName)
    app.connect("include-read", _replaceProjectName)
