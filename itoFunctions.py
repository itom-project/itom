# ***********************************************************************
#    itom software
#    URL: http://www.uni-stuttgart.de/ito
#    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
#    Universitaet Stuttgart, Germany#
#
#    This file is part of itom.
#
#    itom is free software; you can redistribute it and/or modify it
#    under the terms of the GNU Library General Public Licence as published by
#    the Free Software Foundation; either version 2 of the Licence, or (at
#    your option) any later version.
#
#    itom is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
#    General Public Licence for more details.
#
#    You should have received a copy of the GNU Library General Public License
#    along with itom. If not, see <http://www.gnu.org/licenses/>.
# ***********************************************************************

import sys
import inspect
import gc
import __main__
from types import ModuleType, FunctionType, MethodType, BuiltinMethodType, BuiltinFunctionType

def getModules():
    mods = sys.modules
    result = [ [key] + getModuleFile(value) for key,value in mods.items() ]
    result = sorted(result, key=lambda item: item[0])
    return result

def getModuleFile(mod):
    try:
        #print(mod)
        p = inspect.getfile(mod)
        #print(p)
        if(p.startswith((sys.prefix,sys.exec_prefix))):
            return [p,2]
        else:
            return [p,0]
    except Exception as e:
        #print("Error:", e)
        return ["<build-in>",1]

def reloadModules(modNames):
    import imp
    res = []
    for i in modNames:
        if(sys.modules[i] != None):
            try:
                imp.reload(sys.modules[i])
            except SyntaxError as err:
                s = "module %s could not be reloaded. Invalid syntax in file %s, line %i: %s (character %i)" % (str(sys.modules[i]), err.filename, err.lineno, err.text, err.offset)
                print(s)
            except Exception as err:
                print("error while reloading module", str(sys.modules[i]), ":", str(err))
        else:
            res.append(i)
    return res

def at(addr):
    """Return an object at a given memory address.

    The reverse of id(obj):

        >>> at(id(obj)) is obj
        True

    Note that this function does not work on objects that are not tracked by
    the GC (e.g. ints or strings).
    """
    for o in gc.get_objects():
        if id(o) == addr:
            return o
    return None

def importMatlabMatAsDataObject(value):
    """
    This method is called by loadMatlabMat if the containing element is a numpy-array with a field itomMetaInformation

    Then the fields are analyzed and assumed to be tags and additional information for the dataObject.
    """
    import numpy as np
    import itom

    if(type(value) is np.ndarray):
        fields = value.dtype.fields
        itomMetaInformation = value["itomMetaInformation"] #str( value.getfield( *(fields["itomMetaInformation"]) ) )
        if(itomMetaInformation == "dataObject"):
            res = itom.dataObject( value["dataObject"].flat[0] )

            #res.valueUnit = float(value["valueUnit"])

        elif(itomMetaInformation == "npDataObject"):
            res = itom.npDataObject( value["dataObject"].flat[0] )
        else:
            raise RuntimeError('itomMetaInformation unknown')
    else:
        raise RuntimeError('value must be a numpy ndarray')

    return res

def clearAll():
    '''
    Clears all the global variables from the workspace except for all function, modules, classes, itom variables and items stored in clearAllState...
    '''
    if not clearAllState:
        raise RuntimeError('No initial state found')
        return
    else:
        deleted_keywords = []

        for var in __main__.__dict__:
            if var[0] == '_':
                continue
            #ignore the three constants defined by the itom module
            if var in ['BUTTON', 'MENU', 'SEPARATOR'] or var in clearAllState:
                continue

            item = __main__.__dict__[var]
            if isinstance(item, ModuleType) or \
            isinstance(item, FunctionType) or \
            isinstance(item, MethodType) or \
            isinstance(item, BuiltinMethodType) or \
            isinstance(item, BuiltinFunctionType) or \
            isinstance(item, type):
                continue

            deleted_keywords.append(var)

        for key in deleted_keywords:
            del __main__.__dict__[key]
        #call garbage collector to really and immediately remove all flaged variables
        gc.collect()

clearAllState = None
def getClearAllValues():
    global clearAllState
    clearAllState = []
    for var in __main__.__dict__:
        clearAllState.append(var)
