"""
Build script for fast C++ algorithms module.

This module provides optimized C++ implementations of performance-critical
algorithms using pybind11 for Python bindings.

Build Instructions:
    pip install pybind11 numpy
    python setup_cpp.py build_ext --inplace

Or install as a package:
    pip install .
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get pybind11 include path
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("pybind11 not found. Installing...")
    os.system(f"{sys.executable} -m pip install pybind11")
    import pybind11
    pybind11_include = pybind11.get_include()

# Get numpy include path
try:
    import numpy as np
    numpy_include = np.get_include()
except ImportError:
    print("numpy not found. Installing...")
    os.system(f"{sys.executable} -m pip install numpy")
    import numpy as np
    numpy_include = np.get_include()


class BuildExt(build_ext):
    """Custom build extension for adding compiler-specific options."""
    
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/std:c++17'],
        'unix': ['-O3', '-std=c++17', '-fPIC'],
    }
    
    link_opts = {
        'msvc': [],
        'unix': [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.link_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            # Enable OpenMP if available
            opts.append('-fopenmp')
            link_opts.append('-fopenmp')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            # Enable OpenMP for MSVC
            opts.append('/openmp')
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)


# Define the extension module
ext_modules = [
    Extension(
        'fast_algorithms',
        sources=['cpp/fast_algorithms.cpp'],
        include_dirs=[
            pybind11_include,
            numpy_include,
        ],
        language='c++',
    ),
]

setup(
    name='fast_algorithms',
    version='1.0.0',
    author='VcMoldCreator',
    description='Fast C++ implementations for mesh algorithms',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
)
