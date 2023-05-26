This folder is supposed to contain python scripts or packages, which should be used together with the software itom.

The folder itom-packages is already appended to the variable "sys.path" at startup of itom.

You have the following possibilities:

1. Copy any file "yourFileName.py" directly to the folder itom-packages. Then this file can be imported in itom by

	>> import yourFileName

		or

	>> from yourFileName import *

		...

2. Create a folder "myPackage" in folder itom-packages. This folder should contain any structure like the following one:

	- file __init__.py (you can keep this file empty)
	- file yourFileName1.py
	- file yourFileName2.py
	- ...

   Then the content of the files is imported by one of the following commands:

	>> import myPackage.yourFileName1 (then call any method in yourFileName1 by myPackage.yourFileName1.method(params))

		or

	>> from myPackage import yourFileName1 (then call any method in yourFileName1 by yourFileName1.method(params))

		...



The folder mpl_itom should be kept in itom-packages, since it provides a backend for matplotlib together with itom.
This can be used by firstly call the following commands:

>>> import matplotlib
>>> matplotlib.use('module://mpl_itom.backend_itomagg',False) #do not warn if already loaded
