import os
from glob import glob

from setuptools import find_packages, setup

package_name = "bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Install all launch files under share/bringup/launch so
        # `ros2 launch bringup <name>.launch.py` resolves them.
        (os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py"))),
        # The nodes' default parameters, loaded by launch_common.node_parameters().
        (os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Raul Moran",
    maintainer_email="raul@isc-fs.com",
    description=(
        "Top-level launch files for the IFSSIM autonomy pipeline "
        "(full / sim / car) plus shared launch helpers."
    ),
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
