# Sphinx configuration for Torch-Candle documentation

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Project Information
project = 'Torch-Candle'
copyright = '2026, Deepmind Advanced Agentic Coding'
author = 'Antigravity'
release = '0.1.0'

# General Configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML Options
html_theme = 'pytorch_sphinx_theme'
html_theme_options = {
    'pytorch_project': 'docs',
    'canonical_url': 'https://torch-candle.org/docs',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

# Intersphinx mapping to PyTorch docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# LaTeX/PDF Generation configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}
