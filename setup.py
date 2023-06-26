from distutils.core import setup


setup(
    name='drama-wrapper',
    packages=['drama'],
    version='v1.0',
    license='MIT',
    description='Dynamically Restricted Action Spaces for Multi-Agent Reinforcement Learning Frameworks',
    author='Michael Oesterle, Tim Grams',
    author_email='michael.oesterle@uni-mannheim.de, tim.nico.grams@uni-mannheim.de',
    url='https://github.com/michoest/hicss-2024',
    download_url='https://github.com/michoest/hicss-2024/archive/refs/tags/v1.0.tar.gz',
    install_requires=[
        'pettingzoo',
        'gymnasium',
        'numpy'
    ],
    python_requires=">=3.6",
)
