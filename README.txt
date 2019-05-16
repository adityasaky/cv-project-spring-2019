### README

A narrated video describing the project: https://youtu.be/0bKNLO7Pb2A

This project requires Python 3. Python 2 will cause issues, and outright failures as far as some outputs go.

The dependencies for this project are:
- NumPy
- OpenCV

We recommend setting this up within a Python virtual environment. After creating a new virtual environment
(and sourcing it), run:

pip install -r requirements.txt

This should install both dependencies. Please note: this has only been tested on UNIX-like systems.

To run the project, use:

python main.py

This runs the project in its default configuration, specifically using a monocle clipart. This can be modified
using the --clipart flag:

python main.py --clipart {monocle, hat, glasses, pin}

Please be careful to maintain the directory structure. The folder data contains all the input used in the project,
actively as well as files used previously during our experimentation process. All outputs are generated in a folder
called output, which is created during execution if it doesn't already exist. The files landmarking.py,
media_functions.py, and constants.py contain important methods and definitions used in main.py. The file experimental.py
isn't actively used in main.py, but contains some methods we wrote as part of the experimental process.

Lastly, this software is strictly Bring-Your-Own-Bird (BYOB).