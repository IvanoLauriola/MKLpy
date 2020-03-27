#from distutils.core import setup, find_packages
from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'MKLpy',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.4.1',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt",
        "scikit-learn"
  ],
  license = "GNU General Public License v3.0",
  description = 'A package for Multiple Kernel Learning scikit-compliant',
  author = 'Lauriola Ivano',
  author_email = 'ivano.lauriola@phd.unipd.it',
  url = 'https://github.com/IvanoLauriola/MKLpy',
  download_url = 'https://github.com/IvanoLauriola/MKLpy',
  keywords = ['kernel', 'MKL', 'learning', 'multiple kernel learning', 'EasyMKL','SVM','boolean kernels'],
  classifiers = [
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Mathematics',
                ],
  long_description = long_description,
)
