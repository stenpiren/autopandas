"Setup.py"

from setuptools import setup

setup(name='autopandas',
      version='0.1',
      description='automatic pre-processing of pandas',
      url='http://github.com/tvaroska/autopandas',
      author='Boris Tvaroska',
      author_email='boris@tvaroska.sk',
      license='MIT',
      packages=['autopandas'],
      install_requires=['numpy', 'scipy', 'sklearn', 'pandas'],
      zip_safe=False)
