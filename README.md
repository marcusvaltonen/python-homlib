# python-homlib
Python homography library. Classic and state-of-the-art methods for homography estimation.

Wrapps the C++/Eigen library HomLib.

## Solvers available
List will be updated.

## Development
The Python wrapper uses eigency, which can be downloaded using pip. The requirements.txt
file in the `python` subdirectory also installs the necessary dependencies (numpy and cython)
```bash
    $ pip install -r requirements.txt
```
In order to compile and wrap the C++ code
```bash
    $ python setup.py bdist_wheel
```
To install the local changes use
```bash
    $ pip install -e .[dev]
```
