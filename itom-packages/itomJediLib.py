import jedi
import sys
import itom

if jedi.__version__ >= '0.12.0':
    jedienv = jedi.api.environment.InterpreterEnvironment()
    
ICON_CLASS = ('code-class', ':/classNavigator/icons/class.png') #':/pyqode_python_icons/rc/class.png')
ICON_FUNC = ('code-function', ':/classNavigator/icons/method.png') #':/pyqode_python_icons/rc/func.png')
ICON_FUNC_PRIVATE = ('code-function', ':/classNavigator/icons/method_private.png') #':/pyqode_python_icons/rc/func_priv.png')
ICON_FUNC_PROTECTED = ('code-function',
                       ':/classNavigator/icons/method_protected.png') #':/pyqode_python_icons/rc/func_prot.png')
ICON_NAMESPACE = ('code-context', ':/classNavigator/icons/namespace.png') #':/pyqode_python_icons/rc/namespace.png')
ICON_VAR = ('code-variable', ':/classNavigator/icons/var.png') #':/pyqode_python_icons/rc/var.png')
ICON_KEYWORD = ('quickopen', ':/classNavigator/icons/keyword.png') #':/pyqode_python_icons/rc/keyword.png')

def icon_from_typename(name, icon_type):
    """
    Returns the icon resource filename that corresponds to the given typename.
    :param name: name of the completion. Use to make the distinction between
        public and private completions (using the count of starting '_')
    :pram typename: the typename reported by jedi
    :returns: The associate icon resource filename or None.
    """
    ICONS = {
        'CLASS': ICON_CLASS,
        'IMPORT': ICON_NAMESPACE,
        'STATEMENT': ICON_VAR,
        'FORFLOW': ICON_VAR,
        'FORSTMT': ICON_VAR,
        'WITHSTMT': ICON_VAR,
        'GLOBALSTMT': ICON_VAR,
        'MODULE': ICON_NAMESPACE,
        'KEYWORD': ICON_KEYWORD,
        'PARAM': ICON_VAR,
        'ARRAY': ICON_VAR,
        'INSTANCEELEMENT': ICON_VAR,
        'INSTANCE': ICON_VAR,
        'PARAM-PRIV': ICON_VAR,
        'PARAM-PROT': ICON_VAR,
        'FUNCTION': ICON_FUNC,
        'DEF': ICON_FUNC,
        'FUNCTION-PRIV': ICON_FUNC_PRIVATE,
        'FUNCTION-PROT': ICON_FUNC_PROTECTED
    }
    ret_val = ""
    icon_type = icon_type.upper()
    # jedi 0.8 introduced NamedPart class, which have a string instead of being
    # one
    if hasattr(name, "string"):
        name = name.string
    if icon_type == "FORFLOW" or icon_type == "STATEMENT":
        icon_type = "PARAM"
    if icon_type == "PARAM" or icon_type == "FUNCTION":
        if name.startswith("__"):
            icon_type += "-PRIV"
        elif name.startswith("_"):
            icon_type += "-PROT"
    if icon_type in ICONS:
        ret_val = ICONS[icon_type][1]
    elif icon_type:
        _logger().warning("Unimplemented completion icon_type: %s", icon_type)
    return ret_val

def calltipModuleItomModification(sig, params):
    '''mod that returns the call signature for all methods and classes in the itom module
    based on their special docstrings
    '''
    try:
        doc = sig.docstring(raw = False, fast = True)
    except:
        return None
    
    arrow_idx = doc.find("->")
    
    if arrow_idx == -1 or not doc.startswith(sig.name):
        return None
    
    signature = doc[len(sig.name):arrow_idx].strip()
    #remove ( and )
    signature = signature[1:-1]
    parts = signature.split(",")
    
    if len(parts) == len(params):
        return parts
    else:
        return None

calltipsModificationList = { 'itom': calltipModuleItomModification }

def calltips(code, line, column, path = None, encoding = "utf-8"):
    '''
    '''
    
    if jedi.__version__ >= '0.12.0':
        script = jedi.Script(code, line + 1, column, path, encoding, environment = jedienv)
    else:
        script = jedi.Script(code, line + 1, column, path, encoding)
    signatures = script.call_signatures()
    result = []
    for sig in signatures:
        index = sig.index
        if index is None:
            index = -1
        #create a formatted calltip (current index appear in bold)
        module_name = str(sig.module_name)
        call_name = str(sig.name)
        
        paramlist = None
        if module_name in calltipsModificationList:
            paramlist = calltipsModificationList[module_name](sig, sig.params)
        
        if paramlist is None:
            paramlist = [p.description for p in sig.params]
        
        if index >= 0 and index < len(paramlist):
            paramlist[index] = "<b>%s</b>" % paramlist[index]
        params = ", ".join(paramlist)
        if module_name != "":
            calltip = "<p style='white-space:pre'>%s.%s(%s)</p>" % (module_name, call_name, params)
        else:
            calltip = "<p style='white-space:pre'>%s(%s)</p>" % (call_name, params)
        result.append( \
            (calltip, \
            column, \
            sig.bracket_start[0], \
            sig.bracket_start[1]) \
            )
    return result


    
def completions(code, line, column, path, prefix, encoding = "utf-8"):
    '''
    '''
    if jedi.__version__ >= '0.12.0':
        script = jedi.Script(code, line + 1, column, path, encoding, environment = jedienv)
    else:
        script = jedi.Script(code, line + 1, column, path, encoding)
    completions = script.completions()
    result = []
    
    #the following pairs of [name, type] will not be returned as possible completion
    blacklist = [['and', 'keyword'], ['if', 'keyword'], ['in', 'keyword'], ['is', 'keyword'], ['not', 'keyword'], ['or', 'keyword']]
    
    for completion in completions:
        if [completion.name, completion.type] in blacklist:
            continue
        try:
            desc = completion.description
            result.append( \
                (completion.name, \
                desc, \
                icon_from_typename(completion.name, completion.type), \
                completion.docstring()) \
                )
        except:
            break #todo, check this further
    return result

def goto_assignments(code, line, column, path, mode=0, encoding = "utf-8"):
    '''
    mode: 0: goto definition, 1: goto assignment (no follow imports), 2: goto assignment (follow imports)
    '''
    if jedi.__version__ >= '0.12.0':
        script = jedi.Script(code, line + 1, column, path, encoding, environment = jedienv)
    else:
        script = jedi.Script(code, line + 1, column, path, encoding)
    
    try:
        if mode == 0:
            assignments = script.goto_definitions()
        elif mode == 1:
            assignments = script.goto_assignments(False)
        else:
            assignments = script.goto_assignments(True)
    except Exception as ex:
        #print(str(ex))
        assignments = []
    result = []
    for assignment in assignments:
        #print("Assignment::", str(assignment), assignment.module_path, assignment.line, assignment.column, assignment.full_name)
        result.append( \
            (assignment.module_path if assignment.module_path is not None else "", \
            assignment.line - 1 if assignment.line else -1, \
            assignment.column if assignment.column else -1, \
            assignment.full_name, \
            ) \
            )
    return result
    
if __name__ == "__main__":
    
    print(calltips("from itom import dataObject\ndataObject.zeros(", 1, 17, "utf-8"))
    result = completions("Pdm[:,i] = m[02,i]*P[:,i]", 0, 15, "", "", "utf-8")
    print(calltips("from itom import dataObject\ndataObject([4,5], 'u",1,17, "utf-8"))
    print(calltips("def test(a, b=2):\n    pass\ntest(", 2, 5, "utf-8"))
    
    print(completions("from itom import dataObject\ndataO, 'u",1,5, "", "", "utf-8"))
    print(completions("1", 0, 1, "", "", "utf-8"))
    print(completions("import numpy as np\nnp.arr", 1, 6, "", "", "utf-8"))
    print(goto_assignments("import numpy as np\nnp.ones([1,2])", 1, 5, ""))
    print(goto_assignments("def test(a,b):\n    pass\n\ntest(2,3)", 3, 2, ""))
    

    

#script = jedi.Script("from itom import dataObject\ndataObject([4,5], 'u",2,17,None, "Utf-8")
#script = jedi.Script(text,4,5,None, "Utf-8")
#script = jedi.Script(text2, 3, 12,None,"Utf-8")
#sigs = script.call_signatures()
#for sig in sigs:
    #print("Signature (%s)\n-----------------" % str(sig))
    #print(sig.full_name)
    #print("Current param:", sig.index)
    #print("brackets starts in line %i, column %i" % (sig.bracket_start))
    #for p in sig.params:
        #print(p.description, p.is_keyword)
    ##sig.docstring()
                    

