"""
Copyright (c) 2014, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""
from setuptools import setup

setup(
    name='fbpca',
    version='1.0',
    author='Facebook Inc',
    author_email='opensource@fb.com',
    maintainer='tulloch@fb.com',
    maintainer_email='tulloch@fb.com',
    url='https://www.facebook.com',
    description='Fast computations of PCA/SVD/eigendecompositions via randomized methods',
    py_modules=['fbpca'],
    license='BSD License',
    platforms='Any',
    long_description=open('README.rst').read(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ]
)
