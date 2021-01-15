"""This is a demo script for the itom_jedilib tests without type hints."""

import itom
import sys

__version__ = "2.0.0"


def meth1_nodocstr():
    var = 2 + 3
    var2 = 3
    return float(var)


def meth1_docstr():
    """Returns the float result of 2 + 3."""
    var = 2 + 3
    return float(var)


def meth2_nodocstr(arg1, arg2=4.0):
    print(arg1, arg2)
    
    dobj = itom.dataObject([2, 3])
    dobj2 = dobj - dobj
    dobj3 = dobj.reshape([3, 2])
    print(dobj3.shape)
    dobj3.axisScales = (1.0, 1.5)
    dobj4 = itom.dataObject.zeros([5, 7], dtype="uint8")
    versiondict = itom.version(dictionary=True)


def meth2_docstr(arg1, arg2=4.0):
    """Prints the values of both arguments."""
    print(arg1, arg2)


def meth3_docstr(arg1, arg2, *args):
    """Returns a list of all unwrapped arguments."""
    return [arg1, arg2, *args]


class MyClass:
    
    def __init__(self, arg1, arg2):
        self._arg1 = arg1
        self.__arg2 = arg2
    
    def doit(self):
        return [self._arg1, self.__arg2]
    
    def _doit2(self, ret1, ret2):
        if self._arg1:
            return ret1
        return ret2
    
    @property
    def prop1(self):
        if self._arg1:
            return 5
        return None
    
    @property
    def prop2(self):
        return self._arg1
    
    @prop2.setter
    def prop2(self, arg):
        self._arg1 = arg
    
    @staticmethod
    def fromNothing():
        obj = MyClass(arg1=5, arg2=7)
        return obj
    
    @classmethod
    def fromAnything(cls):
        obj = cls(5, arg2=7)
        return obj


class MyClassDocStr:
    """MyClass with docstrings."""
    
    def __init__(self, arg1, arg2):
        """Initializes MyClassDocStr."""
        self._arg1 = arg1
        self.__arg2 = arg2
    
    def doit(self):
        """Returns a list of arguments."""
        return [self._arg1, self.__arg2]
    
    def _doit2(self, ret1, ret2):
        """Do something fancy.
        
        Returns ret1 or ret2."""
        if self._arg1:
            return ret1
        return ret2
    
    @property
    def prop1(self):
        """First property."""
        if self._arg1:
            return 5
        return None
    
    @property
    def prop2(self):
        """2nd property. Can be set."""
        return self._arg1
    
    @prop2.setter
    def prop2(self, arg):
        self._arg1 = arg
    
    @staticmethod
    def fromNothing():
        """Returns an initialized class object."""
        obj = MyClassDocStr(arg1=5, arg2=7)
        return obj
    
    @classmethod
    def fromAnything(cls):
        """Returns an initialized class object (2)."""
        obj = cls(5, arg2=7)
        return obj


if __name__ == "__main__":
    # main part
    pathes = sys.path
    
    if "C:/temp" not in pathes:
        print("not contained")
    
    result1 = meth1_nodocstr()
    result2 = meth1_docstr()
    result3 = meth2_nodocstr(-3)
    result4 = meth2_docstr(7)
    result5 = meth3_docstr(2, 3.0, "hello", [b"test", 3])
    
    mycls1 = MyClass(6, 7)
    p1 = mycls1.prop1
    p2 = mycls1.prop2
    mycls1.prop2 = -5
    mycls1.doit()
    mycls1._doit2("ret1", None)
    mycls2 = MyClass.fromNothing()
    mycls3 = MyClass.fromAnything()
    
    mycls1 = MyClassDocStr(6, 7)
    p1 = mycls1.prop1
    p2 = mycls1.prop2
    mycls1.prop2 = -5
    mycls1.doit()
    mycls1._doit2(2, "string")
    mycls2 = MyClassDocStr.fromNothing()
    mycls3 = MyClassDocStr.fromAnything()

