from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cone_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name+'/meshes', glob('meshes/*.dae')),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaime',
    maintainer_email='jaimeperezgil21@gmail.com',
    description='LiDAR cone detection: ground removal (RANSAC), DBSCAN clustering, and per-cluster cone fitting that publishes raw cone observations on /Conos_raw for downstream factor-graph SLAM in cone_slam.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Lifecycle entry. Node name "cone_detection_node" matches the
            # AUTONOMY_LIFECYCLE_NODES list in mode_manager.
            'cone_detection_node = cone_detection.node:main',
        ],
    },
)

