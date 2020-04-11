import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize
from distutils.extension import Extension


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="fanok",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.5.0",
    description="Knockoffs in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qrebjock/fanok",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(
        [
            Extension("fanok.sdp._low_rank", ["fanok/sdp/_low_rank.pyx"]),
            Extension("fanok.sdp._full_rank", ["fanok/sdp/_full_rank.pyx"]),
            Extension(
                "fanok.factor_model._shrinkage",
                ["fanok/factor_model/_shrinkage.pyx"],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
            ),
            Extension("fanok.utils._cholesky", ["fanok/utils/_cholesky.pyx"]),
        ],
        compiler_directives={"language_level": "3"},
        annotate=False,
    ),
    include_dirs=[np.get_include()],
)
