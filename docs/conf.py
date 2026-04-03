from __future__ import annotations

project = "versionable"
author = "Hendrick Melo"
release = "0.0.1"

extensions = ["myst_parser"]

exclude_patterns = ["_build", "plans/**"]

html_theme = "furo"
html_static_path = ["_static", "images"]
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
