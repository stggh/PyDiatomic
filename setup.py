import setuptools

with open("README.rst", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name = "PyDiatomic",
    version = "0.3.0",
    author = "Stephen Gibson",
    author_email = "Stephen.Gibson@anu.edu.au",
    description='A Python package to solve the time-independent coupled-channel Schr&ouml;inger equation using the Johnson renormalized Numberov method',
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/stggh/PyDiatomic",
    packages = setuptools.find_packages(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Chemistry',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Programming Language :: Python :: 3.10',
      ],
)
