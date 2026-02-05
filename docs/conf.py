# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup ---------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "leonardo_toolset")),
)


# -- Project information ------------------------------------------
project = "Leonardo"
copyright = "2024, Yu Liu"
author = "Yu Liu"

# -- General configuration ----------------------------------------
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx.ext.napoleon",
    "sphinx_panels",
    "sphinx_design",
]

# templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc & napoleon settings
autoclass_content = "both"
autodoc_typehints = "description"
add_module_names = False
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# -- HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
pygments_style = "sphinx"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links.html", "search-field.html"],
    "navbar_persistent": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/peng-lab/leonardo_toolset",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/leonardo-toolset/",
            "icon": "fa-solid fa-box-open",
        },
        {
            "name": "Image.sc Forum",
            "url": "https://forum.image.sc/t/software-release-leonardo-a-toolset-to-remove-sample-induced-aberrations-in-light-sheet-microscopy-images/107622/2",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "Issues",
            "url": "https://github.com/peng-lab/leonardo_toolset/issues",
            "icon": "fa-solid fa-bug",
        },
    ],
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    },
}


# html_sidebars = {"pagename": []}
html_sidebars = {"**": []}

html_context = {
    "display_github": True,
    "github_user": "peng-lab",
    "github_repo": "leonardo_toolset",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


html_show_sphinx = False
epub_show_urls = "footnote"

autodoc_mock_imports = [
    "torch",
    "open3d",
    "bioio",
    "tifffile",
    "pandas",
    "h5py",
    "skimage",
    "cv2",
    "matplotlib",
    "dask",
    "pyntcloud",
    "natsort",
    "ptwt",
    "SimpleITK",
    "ants",
    "jax",
    "jaxwt",
    "haiku",
    "bioio_ome_tiff",
    "bioio_tifffile",
    "asciitree",
    "numpy",
    "tqdm",
    "pywt",
    "scipy",
    "copy",
    "gc",
    "shutil",
    "traceback",
    "yaml",
    "jinja2",
    "pathlib",
    "re",
    "colour",
    "haiku",
    "torchvision",
]

rst_prolog = """
.. |Docs| replace:: `Docs <https://github.com/peng-lab/leonardo_toolset>`__
"""
