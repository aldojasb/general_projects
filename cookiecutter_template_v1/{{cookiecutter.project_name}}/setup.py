from setuptools import setup, find_packages

setup(
    name='{{ cookiecutter.project_name }}',
    version='0.1',
    packages=find_packages(),
    package_dir={'': 'src'},  # Adjust if your source code is under 'src'
)
