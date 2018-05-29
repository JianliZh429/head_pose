# from distutils.core import setup

from setuptools import setup

setup(
    name='head_pose',
    version='0.0.4',
    packages=['head_pose'],
    package_dir={'head_pose': 'head_pose'},
    package_data={
        'head_pose': ['models/*.txt']
    },
    url='https://github.com/DewMaple/head_pose',
    description='Use opencv solvePnP to do head pose estimation',
    author='dew.maple',
    author_email='dew.maple@gmail.com',
    license='MIT',
    keywords=['computer vision', 'image processing', 'head pose', 'opencv-python', 'numpy'],
    classifiers=['Programming Language :: Python :: 3.6'],
    project_urls={
        'Bug Reports': 'https://github.com/DewMaple/head_pose/issues',
        'Source': 'https://github.com/DewMaple/head_pose',
    },
    zip_safe=True
)
