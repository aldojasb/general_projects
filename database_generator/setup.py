from setuptools import setup, find_packages

setup(
    name='database_generator',
    version='0.1',
    packages=find_packages('src'),  # Adjust if your source code is under 'src'
    package_dir={'': 'src'},  # Use 'src' if your code is under a 'src' directory

)
