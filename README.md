# HM Scripts

Personal scripts for the Higher Mathematics course at the Zurich University of Applied Sciences.

## Structure

There are two main folders:

* `numpy/` contains scripts that use the `numpy` library and simply calculates the result of the given operation. It
  does not provide intermediate steps nor does it work with symbolic variables.

* `sympy/` contains scripts that use the `sympy` library and provides intermediate steps as well as support for symbolic
  variables. It provides a more detailed output than the `numpy` scripts and can be used to simply copy its output as
  the solution for the given task.

## Usage

### Numpy

To use the `numpy` scripts, simply open them in your favorite text editor or IDE. You can then alter the values in the
*"Definitions"* section to your liking and run the script. The result will be printed to the console.

### Sympy

The `sympy` scripts use jupyter notebooks in order to provide a more detailed output. To use them, you need to have
`jupyter` installed. You can then open the notebook in your browser and run the cells. The result will be output as
mathematical expressions in the notebook.

Within the folder you can find multiple files for each problem. The `*.ipynb` files are the notebooks that you can open
in your browser. The `*.py` files are the python scripts that are used to provide the implementation details for the
notebooks. You can open them in your favorite text editor or IDE. The `_test.py` files are used to test the 
implementation of the `*.py` files. They are not required to use the notebooks.

### Virtual environment

Pipenv users may install all dependencies with `pipenv install`. Remember to select the proper runtime for scripts and notebooks via `pipenv shell` or your IDE.