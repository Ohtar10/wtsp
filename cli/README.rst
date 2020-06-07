WTSP ML — CLI to Where to Sell Products ML training
===================================================

.. contents:: **Table of Contents**
  :depth: 3

Introduction
------------

This repository contains the code for the project under the codename *wtsp*,
an automated workflow to describe, train and transform data from twitter and
product review data in order to characterize geographic areas in terms of their
relationship with products.

The tool is a CLI (Command Line Interface) that executes the necessary tasks in
the above mentioned stages, saving metadata on each execution in the local file
system that can be used by subsequent stages or by the user as output or intermediate
results.

This tool is part of the MsC project of Luis Eduardo Ferro Diez <luisedof10@gmail.com>
and it has only demonstrative purpose.

The project is maintained by Luis Eduardo Ferro Diez.

Deliverables
------------

The tool is a python installable CLI tool from sources, not all the dependencies
will be part of the final requirements file since there are some that are really
hard to make it work out of the box and they will be advertised to the users
to install them manually.

There will be a dockerized version of the tool to speed up the setup process, however,
the ANN portion of the project won't be able to take advantage of the GPU if available.

.. code-block:: console

    docker container run ohtar10/wtsp:0.1.1

Formally released packages will follow later.

Directory layout
----------------

This is a rough overview of the top-level directory structure inside the
repository:

::

    +- wtsp
    |  |
    |  +- cli           # Command Line Interface artifacts
    |  |
    |  +- core          # Core artifacts
    |  |
    |  +- describe      # Artifacts related to 'describe' command
    |  |
    |  +- train         # Artifacts related to 'train' command
    |  |
    |  +- transform     # Artifacts related to 'predict' command
    |  |
    |  +- view          # Visualization and export utilities
    |
    +- tests
       |
       +- assets        # assets used in the tests
       |
       +- integration   # integration tests
       |
       +- unit          # unit tests


Development setup for end users
-------------------------------

Requirements
.............

Make sure you have a working installation of Python 3.7+.

    Software packages:

    - nvidia-cuda-toolkit
    - cuda 10.1


Install instructions
....................

It is recommended to use Conda_ to create an isolated environment, first.
Then, you can install the package using ``conda`` from the unpacked archive's
root directory like this:

.. code-block:: console

    $ conda env create --name wtsp-cli-dev -f environment.yml
    $ conda activate wtsp-cli-dev
    $ make all

To update the conda environment.yml file and export without fixed versions:

.. code-block:: console

    $ conda update conda
    $ conda update --all
    $ conda env export | cut -f 1 -d '=' > environment.yml


Test instructions
.................

To run the entire test suite you only need to execute the following command:

.. code-block:: console

    $ make test

Usage instructions
..................
Once the tool is installed you can navigate through the commands normally:

.. code-block:: console

    $ wtsp --help
    Using TensorFlow backend.
    Usage: wtsp [OPTIONS] COMMAND [ARGS]...

      Where To Sell Products (wtsp) 0.1.1.

    Options:
      --debug / --no-debug  Enable debug output.
      -wd, --work-dir TEXT  Which folder to use as working directory. Default to
                            ~/wtsp

      --help                Show this message and exit.

    Commands:
      describe  Describe module.
      predict   Predict module.
      train     Train module.

For more detailed usage please visit the Usage_Note_.

License
-------

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

See the LICENSE_ file in the root of this project for license details.


.. _Anaconda: https://www.anaconda.com/distribution/
.. _Conda: https://docs.conda.io/
.. _LICENSE: ./LICENSE
.. _Usage_Note: ../notebooks/jupyter/machine-learning/all-cli-process.ipynb
