from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'aruco_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share/launch', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PrathamKiran',
    maintainer_email='prathamkiran18@gmail.com',
    description='Gives the location of aruco tags using camera stream',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'pose_publisher = aruco_pose.pose_publisher:main',
        ],
    },
)
