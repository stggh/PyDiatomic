from setuptools import setup, find_packages

setup(
    name = "PyDiatomic",
    version = "0.3",
    description='A Python package to solve the time-independent coupled-channel Schr&ouml;inger equation using the Johnson renormalized Numberov method',
    packages=find_packages(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Chemistry',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Programming Language :: Python :: 3.7',
      ],
)
