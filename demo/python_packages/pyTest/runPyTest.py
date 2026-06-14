"""Pytest
=========

Execute lines from this file from demo folder to run all subsequent pytest scripts.

.. hint::
    from pytest verion ``5.x`` onwards make sure to use -p no:faulthandler

Run pytest using line below. run it from within the pytestfolder in the demo scripts itom folder
modules that are import ed once dont get reimportet, so restart itom frequently
capturing stdin/out/err is not possible since os.openfd is not supported by itom
run from within itom console...
"""

import pytest
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPyTest.png'


# this one collects ALL the scripts from the topLevelfolder(..)
# omit the collect-only switch to run them all
pytest.main(["../", "-v", "--capture=no", "--collect-only", "-c=./pytest.ini"])
# this one runs only tests from certain file
pytest.main(["../demoDataObject.py", "--capture=no", "-c=./pytest.ini"])
# and there are many more mnipulations possible, documented on http://pytest.org/en/latest/
# Configuration is more easily done in the *.ini and conftest.py files.
# So here two more helpful examples:
# this one runs all the demo functions
pytest.main(
    ["../", "-v", "--capture=no", "--collect-only", "-c=./pytest_demo.ini"]
)
# this one runs all the user demo functions, which require interaction from the user.
pytest.main(["../", "-v", "--capture=no", "-c=./pytest_userdemo.ini"])
