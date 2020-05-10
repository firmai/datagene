from setuptools import setup, Command
import os
import sys


setup(name='datagene',
      version='0.0.3',
      description='Data Comparison Toolbox with Transformation and Similarity Analysis',
      url='https://github.com/firmai/datagene',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['datagene'],
      install_requires=[
            "pandasvault",
            "scikit-image==0.15",
            "scikit-image",
            "deltapy",
            "esig",
            "ImageHash",
            "porespy",
            "shap",
            "pyts",
            "tensorly",
            "pandas",
            "scikit-learn",
            "scipy",
            "statsmodels",
            "numpy",
            "google",
            "tensorflow",
            "keras",
            "opencv-python"
            
            
            ],
      zip_safe=False)
