import itom

#choose instance
varnames = [name for name in dir() if (type(globals()[name]) is itom.actuator or type(globals()[name]) is itom.dataIO)]
[name,success] = ui.getItem("Choose instance", "Choose actuator or dataIO instance", varnames, editable=False)

def parse_parameters(instance):
    result = []
    funcdict = instance.getExecFuncsInfo(detailLevel = 1) #get execfunctions names
    
    for key, value in funcdict.items(): #get mandatory/ optional/ output parameters
        paramtypeinfo = instance.getExecFuncsInfo(key, detailLevel = 1)
        item = "'%s'\n" %(key)
        

        #item = "Parameters for the execFunction **%s** are:\n" %(key)
        result.append(item)
        
        item = value + "\n"
        result.append(item)
        
        for key, value in paramtypeinfo.items():
            
            item = "%s\n" %(key)
            result.append(item)
            for p in value:
                item = "**%s**: {%s}\n    %s\n" % (p["name"], p["type"], p["info"])
                result.append(item)
    print("\n".join(result))

if success:
    parse_parameters(globals()[name])
