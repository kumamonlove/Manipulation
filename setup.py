from setuptools import find_packages, setup

package_name = 'manipulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andy',
    maintainer_email='andy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'Manipulation = manipulation.Manipulation:main',
        'record = manipulation.record:main',
        'gen3lite_pymoveit2 = manipulation.gen3lite_pymoveit2:main',
        'safe = manipulation.safe:main',
        'monitor = manipulation.monitor:main',
        'help = manipulation.help:main',
        'simulated_gripper = manipulation.simulated_gripper_publisher:main'
        ],
    },
)
