# Use setuptools in preference to distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os

DESCRIPTION = "A package for computing shock elasticities."

setup(name='shockelas',
      packages=['shockelas',
                'shockelas.util',
                'shockelas.tests'],
      version=0.1,
      description=DESCRIPTION,
      author='Quentin Batista',
      author_email='batista.quent@gmail.com',
      url='https://github.com/QBatista/KnightianInnovationModel.py',  # URL to the repo
      keywords=['quantitative', 'economics', 'shock', 'elasticities'],
      install_requires=[
          'numpy',
          'scipy',
          'plotly'
          ]
      )
