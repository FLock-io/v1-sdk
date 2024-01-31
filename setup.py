from setuptools import setup, find_packages

requirements = [
        'flask==2.3.2',  # dependency list
        'loguru==0.7.0',
    ]

setup(
    name='flock-sdk',  # package name
    version='0.0.2',  # package version
    author='FLock.io',  # author name
    author_email='info@flock.io',  # author email
    description='An SDK for building applications on top of FLock V1',  # short description
    long_description=open('README.md').read(),  # long description, usually is README
    long_description_content_type='text/markdown',  # long description content type, e.g. text/markdown or text/plain
    url='https://github.com/FLock-io/v1-sdk',  # project home page link
    packages=find_packages(),  # automatically find directories containing '__init__.py'
    keywords=[
        "blockchain",
        "federated learning",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",  # supported Python versions
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  # license
        'Operating System :: OS Independent',  # operating system
    ],
    install_requires=requirements,
    python_requires='>=3.7',  # Python version requirement
)
