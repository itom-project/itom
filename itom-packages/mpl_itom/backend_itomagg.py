import matplotlib

if matplotlib.__version__ < "2.0.0":
    from .backend_itomagg_v1 import *
elif matplotlib.__version__ < "3.0.0":
    from .backend_itomagg_v2 import *
elif matplotlib.__version__ < "3.7.0":
    from .backend_itomagg_v3 import *
else:
    from .backend_itomagg_v3_7 import *
