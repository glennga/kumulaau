from distutils.core import setup, Extension

k = [Extension('pop', ['src/_pop.c'], libraries=["m", "gsl", "gslcblas"])]
setup(name='pop', version='1.0', ext_modules=k)

