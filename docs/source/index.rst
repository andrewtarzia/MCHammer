.. toctree::
   :hidden:
   :caption: MCHammer
   :maxdepth: 2

   Analysis <analysis>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>


Introduction
------------

| GitHub: https://www.github.com/andrewtarzia/MCHammer


.. important::

  **Warning**: DOCS underdevelopment.

:mod:`.CGExplore` is a Python library for for working with
coarse-grained models.

The library is built off of `stk <https://stk.readthedocs.io/en/stable/>`_,
which comes with the pip install.

.. important::

  **Warning**: This package is still very much underdevelopment and many changes
  are expected.

Installation
------------

To install :mod:`.CGExplore`, you need to follow these steps:

Create a `conda` or `mamba` environment::

  mamba create -n NAME python=3.11

Activate the environment::

  conda activate NAME


Install :mod:`.CGExplore` with pip::

  pip install cgexplore


Install `OpenMM` `docs <https://openmm.org/>`_::

  mamba install openmm

or::

  conda install -c conda-forge openmm


Install `openmmtools` `docs <https://openmmtools.readthedocs.io/en/stable/gettingstarted.html>`_::

  mamba install openmmtools

or::

  conda config --add channels omnia --add channels conda-forge
  conda install openmmtools


Then, update directory structure in `env_set.py` if using example code.


The library implements some analysis that uses `Shape 2.1`. Follow the
instructions to download and installed at
`Shape <https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/>`_


Developer Setup
---------------

To develop with `CGExplore`, you should create a new environment as above, then:

Clone :mod:`.CGExplore` from `here <https://github.com/andrewtarzia/CGExplore>`_

From :mod:`.CGExplore` directory use `just <https://github.com/casey/just>`_ to
install a dev environment with::

  just dev

And then follow the previous steps.


Examples
--------


The main series of examples are in `First Paper Example`_. In that page you
will find all the information necessary to reproduce the work in
`10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_

With each pull request a test is run as a GitHub Action connected to this
`repository <https://github.com/andrewtarzia/cg_model_test>`_.
This ensures that the results obtained for a subset of the original data set do
not change with changes to this library.

.. note::

  `cg_model_test <https://github.com/andrewtarzia/cg_model_test>`_ is a good
  example of usage too!


New works done with :mod:`.CGExplore`:

* TBC.


Acknowledgements
----------------

This work was completed during my time as a postdoc, and then research fellow
in the Pavan group at PoliTO (https://www.gmpavanlab.com/).

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`First Paper Example`: first_paper_example.html
