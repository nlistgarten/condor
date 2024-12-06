# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import condor

project = "Condor"
copyright = "2023, Benjamin W. L. Margolis"
author = "Benjamin W. L. Margolis"
version = condor.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
]

# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/README.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["overrides.css"]


sphinx_gallery_conf = {
    "examples_dirs": "guide_src",
    "gallery_dirs": "guide",
    "filename_pattern": ".*.py",
    "within_subsection_order": "FileNameSortKey",
}
