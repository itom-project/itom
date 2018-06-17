import jedi

def calltips(code, line, column, encoding = "uft-8"):
    '''
    '''
    script = jedi.Script(code, line, column, None, encoding)
    signatures = script.call_signatures()
    result = []
    for sig in signatures:
        index = sig.index
        #create a formatted calltip (current index appear in bold)
        module_name = str(sig.module_name)
        call_name = str(sig.name)
        paramlist = [p.description for p in sig.params]
        if sig.index and sig.index >= 0:
            paramlist[sig.index] = "<b>%s</b>" % paramlist[sig.index]
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
    
if __name__ == "__main__":
    print(calltips("from itom import dataObject\ndataObject([4,5], 'u",2,17, "utf-8"))
    print(calltips("def test(a, b=2):\n    pass\ntest(", 3, 5, "utf-8"))

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
                    

