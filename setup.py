from setuptools import setup, find_packages
from os import path, environ

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()



setup(
    name='PyCSR3D',
    version = '0.0.1',
    packages=find_packages(),  
    package_dir={'csr3d':'csr3d'},
    url='https://github.com/ChristopherMayes/PyCSR3D',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
