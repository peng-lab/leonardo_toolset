Installation
============

Leonardo requires Python<=3.9 and numpy<2.0. We highly recommend using :mod:`conda` 
virtual environment to install and operate Leonardo.
The jax version has been tested on Windows, a solution for macOS and Linux is being worked on.
Please refer to the PyTorch version on those operating systems.

PyPI
-----

Install Leonardo from PyPI using:

.. code-block:: bash

    conda create -n leonardo python=3.9 numpy=1.26.4
    conda activate leonardo
    pip install leonardo_toolset

or full software including Napari plugins:

.. code-block:: bash

    pip install leonardo_toolset[napari]


.. toggle::
   :show:

    Core components in :mod:`Leonardo` are installable separately:

    .. code-block:: bash

        # Leonardo-DeStripe
        pip install lsfm-destripe
        
        # Leonardo-Fuse
        pip install lsfm-fuse

        # Leonardo-DeStripe in Napari
        pip install lsfm_destripe_napari

        # Leonardo-Fuse in Napari
        pip install lsfm_fusion_napari

Development Version
--------------------

To work with the latest development version, install from GitHub using:

.. code-block:: bash

    pip install git+https://github.com/peng-lab/leonardo_toolset.git
