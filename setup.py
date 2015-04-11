import os
from setuptools import find_packages
from setuptools import setup

version = '0.1dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = ''
except IOError:
    README = CHANGES = ''

install_requires = [
    'numpy',
    'Theano',
    ]

docs_require = [
    'Sphinx',
    ]

setup(
    name="Theanet",
    version=version,
    description="Convolutional Neural Networks in Theano",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
        ],
    keywords="Convolutional Neural Networks, Dropout, ImageNet, Deep Learning",
    author="rakeshvar",
    author_email="rakeshvaram@gmail.com",
    url="https://github.com/rakeshvar/theanet",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
        },
    )
