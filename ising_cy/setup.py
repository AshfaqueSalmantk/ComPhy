import numpy
from Cython.Build import cythonize
from distutils.core import Extension,setup


ext = Extension("ising",
                         sources=["ising.pyx"],
                         include_dirs=[numpy.get_include()])


setup(ext_modules=cythonize(ext))
