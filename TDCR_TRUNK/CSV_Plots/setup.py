from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "spiral_optimizer",
        ["spiral_optimizer.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="spiral_optimizer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
