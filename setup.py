from distutils.core import setup

setup(
    name='NDim_ForceAtlas2',
    version='0.1',
    description="Gephi's ForceAtlas2 algorithm for Python, adapted for n dimensions.",
    long_description=open('README.md').read(),
    license='GNU General Public License v3.0',
    author='Igor Sapijaszko',
    author_email='igor.sapijaszko@gmail.com',
    packages=['ndforceatlas'],
    requires=['numpy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords='forceatlas forceatlas2 visualization'
)
