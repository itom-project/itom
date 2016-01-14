import itom

#choose instance
varnames = [name for name in dir() if (type(globals()[name]) is itom.actuator or type(globals()[name]) is itom.dataIO)]
[name,success] = ui.getItem("Choose instance", "Choose actuator or dataIO instance", varnames, editable=False)

def parse_parameters(instance):
    result = []
    funcdict = instance.getExecFuncsInfo(detailLevel = 1) #get execfunctions names
    
    for funcdictName, funcdictValue in funcdict.items(): #get mandatory/ optional/ output parameters
        paramtypedict = instance.getExecFuncsInfo(funcdictName, detailLevel = 1)
        
        mandParamsString = ""
        optParamsString = ""
        outParamsString = ""
        
        if "Mandatory Parameters" in paramtypedict.keys():
            mandParamsString = ", ".join([d["name"] for d in paramtypedict["Mandatory Parameters"]]) 
        if "Optional Parameters" in paramtypedict.keys():
            optParamsString = " [," + ", ".join([d["name"] for d in paramtypedict["Optional Parameters"]]) +"]"
        #if "Output Parameters" in paramtypedict.keys():
            #outParamsString = " ".join([d["name"] for d in paramtypedict["Output Parameters"]]) +" = "
            
        item = "\n.. py:function:: " + outParamsString + " instance.exec('%s', "%(funcdictName) + mandParamsString + optParamsString + ")\n\n"
        result.append(item)
        
        item = "    %s\n\n" %(funcdictValue)
        result.append(item)
        mandParamsString = ""
        optParamsString = ""
        outParamsString = ""
        
        mandParams = []
        optParams = []
        outParams = []
        
        if "Mandatory Parameters" in paramtypedict.keys():
            mandParams = paramtypedict["Mandatory Parameters"]
        if "Optional Parameters" in paramtypedict.keys():
            optParams = paramtypedict["Optional Parameters"]
        if "Output Parameters" in paramtypedict.keys():
            outParams = paramtypedict["Output Parameters"]
        
        mandSignature = ", ".join([i["name"] for i in mandParams])
        optSignature = ", ".join([i["name"] for i in optParams])
        signature = mandSignature
        if signature != "" and optSignature != "":
            signature += "[, " + optSignature + "]"
        elif optSignature != "":
            signature = "[" + optSignature + "]"
        
        if signature != "":
            for i in mandParams:
                result.append("    :param %s: %s\n    :type %s: %s\n" % (i["name"], i["info"], i["name"], i["type"]))
            for i in optParams:
                result.append("    :param %s: %s\n    :type %s: %s - optional\n" % (i["name"], i["info"], i["name"], i["type"]))
            
        for i in outParams:
            result.append("    :return: %s - %s\n    :rtype: %s\n" % (i["name"], i["info"], i["type"]))
        
        
    #print("\n".join(result))
    print("".join(result))
if success:
    parse_parameters(globals()[name])
