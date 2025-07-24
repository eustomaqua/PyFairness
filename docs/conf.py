# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import solar_theme


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyEnsemble'
copyright = '2025, eustomadew'
author = 'eustomadew'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'conestack'
# html_theme = 'piccolo_theme'
html_theme = 'solar_theme'
html_theme_path = [solar_theme.theme_path]
# html_theme = "sphinx_rtd_theme"
# extensions = ['recommonmark', 'sphinx_markdown_tables']
# html_theme = 'alabaster'
html_static_path = ['_static']
