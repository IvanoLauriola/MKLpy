from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'MKLpy',
  packages = find_packages(exclude=['build', 'docs', 'templates']),
  version = '0.5.2',
  install_requires=[
        "numpy",
        "torch",
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
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Mathematics',
                ],
  long_description = long_description,
  long_description_content_type='text/markdown'
)
