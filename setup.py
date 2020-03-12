from setuptools import setup

setup(
    name='LycorisQ',
    version='1.0.18',
    description="A neat reinforcement learning framework based on LycorisNet.",
    author="RootHarold",
    author_email="rootharold@163.com",
    url="https://github.com/RootHarold/LycorisQ",
    py_modules=['LycorisQ'],
    zip_safe=False,
    install_requires=['LycorisNet>=2.5', 'numpy>=1.18']
)
