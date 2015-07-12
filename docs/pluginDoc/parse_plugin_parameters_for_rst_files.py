import itom

#choose instance
varnames = [name for name in dir() if (type(globals()[name]) is itom.actuator or type(globals()[name]) is itom.dataIO)]
[name,success] = ui.getItem("Choose instance", "Choose actuator or dataIO instance", varnames, editable=False)

def parse_parameters(instance):
    result = []
    info = instance.getParamListInfo(1)
    for p in info:
        if p["readonly"]:
            item = "**%s**: {%s}, read-only\n    %s" % (p["name"], p["type"], p["info"])
        else:
            item = "**%s**: {%s}\n    %s" % (p["name"], p["type"], p["info"])
        result.append(item)
    print("\n".join(result))

if success:
    parse_parameters(globals()[name])
