import matplotlib

if matplotlib.__version__ < '2.0.0':
    from .backend_itomagg_v1 import *
if matplotlib.__version__ < '3.0.0':
    from .backend_itomagg_v2 import *
else:
    raise RuntimeError("unsupported matplotlib version: %s" % matplotlib.__version__)