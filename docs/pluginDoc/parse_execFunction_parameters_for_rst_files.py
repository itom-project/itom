import itom

#choose instance
varnames = [name for name in dir() if (type(globals()[name]) is itom.actuator or type(globals()[name]) is itom.dataIO)]
[name,success] = ui.getItem("Choose instance", "Choose actuator or dataIO instance", varnames, editable=False)

def parse_parameters(instance):
    result = []
    funcdict = instance.getExecFuncsInfo(detailLevel = 1) #get execfunctions names
    
    for funcdictName, funcdictValue in funcdict.items(): #get mandatory/ optional/ output parameters
        paramtypedict = instance.getExecFuncsInfo(funcdictName, detailLevel = 1)
        item = ".. py:function:: instance.exec('%s'  [, mandatoryParameters, optionalParameter , outputParameter])\n" %(funcdictName)
        result.append(item)
        
        item = "    %s\n" %(funcdictValue)
        result.append(item)
        
        
        # separate parameter for parameter type
        for typeName, typeValue in paramtypedict.items():
            item = "%s:\n" %(typeName)
            result.append(item)
            
            for paramtypename in typeValue:
                item = "**%s**: {%s}\n    %s\n" % (paramtypename["name"], paramtypename["type"], paramtypename["info"])
                result.append(item)
        
    print("\n".join(result))
        

if success:
    parse_parameters(globals()[name])
