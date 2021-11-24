from setuptools import setup

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='ct-data-annotation',
    version='0.0.1',
    description='Semi-Automatically generate masks and annotations for 3D CT volume using CAD files',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='annotation python CT CAD semi-automated stl',
    author='Md. jamiul alam Khan And Abderrazak Chahid ',
    url='https://github.com/jamiulalam/Semi-Automation-Labelling',
    download_url='https://github.com/jamiulalam/Semi-Automation-Labelling/releases',
    install_requires=['numpy', 'Pillow', 'matplotlib', 'vtk'],
    packages=['SALCT'],
    python_requires='>=3',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'SALCT = SALCT.gui:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)