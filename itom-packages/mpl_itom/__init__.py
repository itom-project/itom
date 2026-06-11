def versiontuple(versionstring):
    """Returns a tuple from a given version string.
    E.g. "3.10.0" returns (3,10,0).

    This tuple can then be compared.
    """
    return tuple(map(int, (versionstring.split("."))))
