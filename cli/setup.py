
"""This is the setup file as used by ``setuptools`` and ``pip``."""

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    lic = f.read()

packages = find_packages(exclude=('tests', 'docs'))
print(packages)

version = {}
with open(f'{packages[0]}/__version__.py') as f:
    exec(f.read(), version)

setup(
    name='wtsp',
    version=version['__version__'],
    description='Where to Sell Products ML Pipeline',
    long_description=readme,
    license=lic,
    author='Luis Eduardo Ferro Diez',
    authod_email='luisedof10@gmail.com',
    packages=packages,
    include_package_data=True,
    install_requires=['click'],
    entry_points='''
        [console_scripts]
        wtsp=wtsp.cli.cli:wtsp
    '''
)
