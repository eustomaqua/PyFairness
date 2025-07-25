.. quickstart.rst


================
Getting started
================


Requirements
-------------

We may test the code on different environments, and in that case, please choose the corresponding Python packages.

.. code-block:: console

  $ # Install anaconda/miniconda if you didn't
  $
  $ # To create a virtual environment
  $ conda create -n ensem python=3.8
  $ conda env list
  $ source activate ensem
  $
  $ # To install packages
  $ pip list && cd ~/FairML
  $ pip install -U pip
  $ pip install -r requirements.txt
  $ python -m pytest
  $
  $ # To delete the virtual environment
  $ conda deactivate && cd ..
  $ yes | rm -r FairML
  $ conda remove -n ensem --all




