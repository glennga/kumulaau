from distutils.core import setup, Extension

setup(name='kumulaau', 
      version='1.0',
      description='Backwards microsatellite simulation and demographic model inference.',
      url='https://github.com/glennga/kumulaau',
      author='Glenn Galvizo',
      packages=['kumulaau'],
      ext_modules=[Extension('pop', ['kumulaau/_pop.c'], libraries=["m", "gsl", "gslcblas"])],
      requires=['numpy', 'numba', 'matplotlib'],
      scripts=['data/alfred/alfred.sh'])
