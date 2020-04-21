from setuptools import setup

setup(
    name='LycorisQ',
    version='1.2.18',
    description="A neat reinforcement learning framework based on LycorisNet.",
    author="RootHarold",
    author_email="rootharold@163.com",
    url="https://github.com/RootHarold/LycorisQ",
    py_modules=['LycorisQ'],
    zip_safe=False,
    install_requires=['LycorisNet>=2.6', 'numpy>=1.18']
)
