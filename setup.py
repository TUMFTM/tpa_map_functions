import setuptools

"""
Package on pypi.org can be updated with the following commands:
python3 setup.py sdist bdist_wheel
sudo python3 -m twine upload dist/*
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tpa_map_functions',
    version='0.8',
    url='https://github.com/TUMFTM',
    author="Leonhard Hermansdorfer",
    author_email="leo.hermansdorfer@tum.de",
    description="Functions to process local acceleration limits for trajectory planning within the TUM Autonomous Motorsports project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["*inputs*", "*outputs*", "*tests*", "*resources*", "*venv*", "*tpa_map_gui*"]),
    install_requires=[
        'numpy>=1.18.1',
        'ad-interface-functions>=0.21',
        'trajectory-planning-helpers>=0.74',
        'pyzmq>=19.0.2',
        'matplotlib>=3.3.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ])
