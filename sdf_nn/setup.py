from setuptools import setup

setup(
    name='sdf_nn',
    version='0.1.0',
    packages=['sdf_nn'],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'rospy',
        'std_msgs',
        'sensor_msgs',
        'message_filters',
    ],
    author='Guillermo Gil Garcia',
    author_email='ggilgar@upo.es',
    description='SDF Neural Network Implementation',
    keywords='ROS Python',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

