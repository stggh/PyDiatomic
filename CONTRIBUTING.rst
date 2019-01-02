Contributing to PyDiatomic
==========================


PyDiatomic is an open source project and improvements are welcome! Please let us know about any issues with the software, even if is just a typo. The easiest way to get started is to open a `new issue <https://github.com/stggh/PyDiatomic/issues>`_.

If you would like to make a Pull Request, the following information may be useful.



Unit tests
----------

Before submitting at Pull Request, be sure to run the unit tests. The test suite can be run from within the PyDiatomic package with ::

    pytest

For more detailed information, the following can be used:  ::

    pytest cse/  -v  --cov=cse

Note that this requires that you have `pytest <https://docs.pytest.org/en/latest/>`_ and (optionally) `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ installed. You can install these with ::

    pip install pytest pytest-cov


Documentation
-------------

PyDiatomic uses Sphinx and `Napoleon <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html>`_ to process Numpy style docstrings, and is synchronized to `pyabel.readthedocs.io <http://pyabel.readthedocs.io>`_. To build the documentation locally, you will need `Sphinx <http://www.sphinx-doc.org/>`_, the `recommonmark <https://github.com/rtfd/recommonmark>`_ package, and the `sphinx_rtd_theme <https://github.com/snide/sphinx_rtd_theme/>`_. You can install all this this using ::

    pip install sphinx
    pip install recommonmark
    pip install sphinx_rtd_theme

Once you have that installed, then you can build the documentation using ::

    cd PyDiatomic/doc/
     make html

Then you can open ``doc/_build/hmtl/index.html`` to look at the documentation. Sometimes you need to use ::

    make clean
    make html

to clear out the old documentation and get things to re-build properly.

When you get tired of typing ``make html`` every time you make a change to the documentation, it's nice to use use `sphix-autobuild <https://pypi.python.org/pypi/sphinx-autobuild>`_ to automatically update the documentation in your browser for you. So, install sphinx-autobuild using ::

    pip install sphinx-autobuild

Now you should be able to ::

    cd PyDiatomic/doc/
    make livehtml

which should launch a browser window displaying the docs. When you save a change to any of the docs, the re-build should happen automatically and the docs should update in a matter of a few seconds.

Alternatively, `restview <https://pypi.python.org/pypi/restview>`_ is a nice way to preview the ``.rst`` files.
