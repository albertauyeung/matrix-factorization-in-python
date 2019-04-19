Prerequisite libraries:


	----> #math

		import math


	----> numpy

		pip: pip install numpy

		anaconda: conda install -c anaconda numpy


	----> scikit-cuda (skcuda)

		pip: pip install scikit-cuda

		anaconda: conda install -c lukepfister scikits.cuda

		Dependecies:
	
			- Python 2.7 or 3.4.
			
			- Setuptools 0.6c10 or later.
			
			- Mako 1.0.1 or later.
			
			- NumPy 1.2.0 or later.
			
			- PyCUDA 2016.1 or later (some parts of scikit-cuda might not work properly with earlier versions).
			
			- NVIDIA CUDA Toolkit 5.0 or later.


	----> **pycuda**

		pip: pip install pycuda

		anaconda:  conda install -c lukepfister pycuda

		Dependecies:

			- Nvidia's CUDA toolkit. PyCUDA was developed against version 2.0 beta. It may work with other versions, too.

			- A C++ compiler, preferably a Version 4.x gcc.

			- A working Python installation, Version 2.4 or newer. 

			additional information can be found: https://wiki.tiker.net/PyCuda/Installation

		
