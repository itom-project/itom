import collections


class ItomEnumException(Exception):
    pass


class ItomEnum:
    """
    provides a c-like enumeration class

    Examples:
    eEx1 = ItomEnum("enumAutoValue",["VALUE1","VALUE2","VALUE3",...])
    eEx2 = ItomEnum("enumUserValue",["VALUE0", ("VALUE100",100), "VALUE101"])

    See: http://code.activestate.com/recipes/67107-enums-for-python/
    """

    def __init__(self, name, enumList):
        """Arguments:
        1. name -- the name of the enumeration, safed in __doc__
        2. enumList -- the list with all elements.

        For the enumList you can either provide a string-list.
        Then every element is the key, the corresponding value is
        an auto-incremented integer, starting with 0. You can
        interrupt the auto-incrementing, by providing elements,
        which are again a list with (key [string], value [integer]).
        """
        self.__doc__ = name
        lookup = collections.OrderedDict()
        reverseLookup = collections.OrderedDict()
        i = 0
        uniqueNames = []
        uniqueValues = []

        for x in enumList:
            if isinstance(x, tuple) or isinstance(x, list):
                x, i = x
            if type(x) != str:
                raise ItomEnumException("enum name is not a string: " + x)
            if type(i) != int:
                raise ItomEnumException("enum value is not an integer: " + i)
            if x in uniqueNames:
                raise ItomEnumException("enum name is not unique: " + x)
            if i in uniqueValues:
                raise ItomEnumException("enum value is not unique for " + x)
            uniqueNames.append(x)
            uniqueValues.append(i)
            lookup[x] = i
            reverseLookup[i] = x
            i = i + 1

        self.table = lookup
        self.__reverseLookup = reverseLookup

    def __repr__(self):
        return "ItomEnum '" + str(self.__doc__) + "'"

    def __getattr__(self, attr):
        """returns value for enum-key or raises AttributeError if not found"""
        if attr not in self.table:
            raise AttributeError
        return self.table[attr]

    def whatis(self, value):
        """returns enum-key for certain value (integer) or raises KeyError if not found"""
        return self.__reverseLookup[value]

    def keys(self):
        """returns tuple with all keys"""
        return tuple(self.table.keys())

    def values(self):
        """returns tuple with all values"""
        return tuple(self.table.values())

    def key_exists(self, key):
        """returns TRUE if key exists in Enum-keys, else FALSE"""
        return key in self.table

    def value_exists(self, value):
        """returns TRUE if value exists in Enum-values, else FALSE"""
        return value in self.__reverseLookup
