from distutils.core import setup

description = """
# Dynamically Restricted Action Spaces for Multi-Agent Reinforcement Learning Frameworks

## Links

Repository:
https://github.com/michoest/hicss-2024

Documentation:
https://drama-wrapper.readthedocs.io/en/master/
"""

setup(
    name='drama-wrapper',
    packages=['drama-wrapper'],
    package_dir={'drama-wrapper': 'drama'},
    version='v1.0.0',
    license='MIT',
    description=description,
    author='Michael Oesterle, Tim Grams',
    author_email='michael.oesterle@uni-mannheim.de, tim.nico.grams@uni-mannheim.de',
    url='https://github.com/michoest/hicss-2024',
    download_url='https://github.com/michoest/hicss-2024/archive/refs/tags/v0.1.tar.gz',
    install_requires=[
        'pettingzoo',
        'torch',
        'gymnasium',
        'shapely',
        'numpy',
        'pygame',
    ],
    python_requires=">=3.6",
)
