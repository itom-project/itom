import sys
import inspect
import gc

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
            except:
                print("error while reloading module", str(sys.modules[i]))
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
    