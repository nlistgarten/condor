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
]

# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/README.rst"]

gallery_src_dirs = ["tutorial_src", "howto_src"]
exclude_patterns.extend(gallery_src_dirs)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["overrides.css"]


sphinx_gallery_conf = {
    "examples_dirs": gallery_src_dirs,
    "gallery_dirs": [name.split("_")[0] for name in gallery_src_dirs],
    "filename_pattern": ".*.py",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "copyfile_regex": r".*\.rst",
    "reference_url": {
        "sphinx_gallery": None,
    },
}

napoleon_custom_sections = [("Options", "params_style")]
