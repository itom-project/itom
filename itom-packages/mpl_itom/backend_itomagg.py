import matplotlib
from mpl_itom import versiontuple

matplotlibversion = versiontuple(matplotlib.__version__)

if matplotlibversion < versiontuple("2.0.0"):
    from .backend_itomagg_v1 import *
elif matplotlibversion < versiontuple("3.0.0"):
    from .backend_itomagg_v2 import *
elif matplotlibversion < versiontuple("3.6.0"):
    from .backend_itomagg_v3 import *
else:
    from .backend_itomagg_v3_6 import *
