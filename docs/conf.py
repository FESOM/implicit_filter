# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

project = "Implicit Filter"
copyright = "2025, Kacper Nowak"
author = "Kacper Nowak"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autodoc_mock_imports = ["jax", "cupy"]

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = ".rst"
master_doc = "index"
language = "en"
pygments_style = "sphinx"

apidoc_module_dir = "./src/implicit_filter"
apidoc_output_dir = "./_build/"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/FESOM/implicit_filter",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_static_path = ["_static"]
