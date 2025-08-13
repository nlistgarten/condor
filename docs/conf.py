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
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/README.rst"]

gallery_src_dirs = ["tutorial_src", "howto_src", "examples_src"]
gallery_out_dirs = [name.split("_")[0] for name in gallery_src_dirs]
exclude_patterns.extend(gallery_src_dirs)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["overrides.css"]


import os
import re

sep = re.escape(os.sep)

sphinx_gallery_conf = {
    "examples_dirs": gallery_src_dirs,
    "gallery_dirs": gallery_out_dirs,
    # include any .py starting with a lowercase letter
    "filename_pattern": sep + r"[a-z].*\.py",
    # files with leading underscore ignored completely by sphinx-gallery
    "ignore_pattern": "^_",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "copyfile_regex": r".*\.rst|.*_lti.py",
    "reference_url": {
        "sphinx_gallery": None,
    },
}
print(sphinx_gallery_conf["filename_pattern"])

napoleon_custom_sections = [("Options", "params_style")]

napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": ":class:`numpy.ndarray`",
    "array-like": ":term:`array-like <array_like>`",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
