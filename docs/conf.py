"""Sphinx configuration for the anuga_drainage documentation."""
import os
import sys

# Make the src-layout package importable for autodoc (also pip-installed on RTD).
sys.path.insert(0, os.path.abspath("../src"))

project = "anuga_drainage"
author = "Stephen Roberts"
copyright = "2026, Stephen Roberts"
release = "0.0.1"

extensions = [
    "myst_nb",          # MyST Markdown + Jupyter notebook rendering (includes myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Notebooks (docs/tutorial.ipynb) ship pre-executed with their outputs; don't
# re-run them on build (ANUGA/pipedream aren't installed on Read the Docs).
nb_execution_mode = "off"

# anuga / pyswmm / pipedream are heavy and optional (the package imports them
# lazily), and aren't installed on Read the Docs — mock them so autodoc imports.
autodoc_mock_imports = ["anuga", "pyswmm", "pipedream_solver"]
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_title = "anuga_drainage"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
