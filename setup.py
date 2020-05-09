from setuptools import setup, Command
import os
import sys


setup(name='datagene',
      version='0.0.1',
      description='Data Comparison Toolbox with Transformation and Similarity Analysis',
      url='https://github.com/firmai/datagene',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['datagene'],
      install_requires=[
            "fbprophet",
            "pandas",
            "pykalman",
            "tsaug",
            "gplearn",
            "ta",
            "tensorflow",
            "scikit-learn",
            "scipy",
            "sklearn",
            "statsmodels",
            "numpy",
            "seasonal"],
      zip_safe=False)
