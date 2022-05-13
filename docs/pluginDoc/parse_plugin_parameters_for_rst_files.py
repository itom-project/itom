# coding=iso-8859-15

"""This script print a reStructuredText representation of all parameters
of an opened plugin instance to the console.

The printed text can be pasted into the rst-documentation file of
a plugin.

Before calling this script, open an instance of a plugin and
make sure that a python variable in the global workspace is assigned
to this instance. Then call this script and choose the variable from
the combobox. The parameters of this plugin will then be parsed and printed.
"""

import itom
import textwrap

clc()

def parse_parameters(instance):
    results = {}
    info = instance.getParamListInfo(1)
    
    for p in info:

        info = textwrap.wrap(p["info"], width=88)
        info = "\n    ".join(info)

        meta = p.get("metaReadableStr", "")
        if meta != "":
            meta = textwrap.wrap(meta, width=88)
            meta = "\n    ".join(meta)

        if p["readonly"]:
            item = "**%s**: %s, read-only\n    %s" % (p["name"], p["type"], info)
        else:
            item = "**%s**: %s\n    %s" % (p["name"], p["type"], info)

        if meta != "":
            item = item + "\n    \n    *%s*" % meta

        results[p["name"]] = item

    results = dict(sorted(results.items()))
    text = "\n".join(list(results.values()))
    itom.copyStringToClipboard(text)
    print(text)


# scan the global workspace for all itom.actuator and itom.dataIO objects
varnames = [
    name
    for name in dir()
    if (type(globals()[name]) is itom.actuator or type(globals()[name]) is itom.dataIO)
]

# the user can now select for which of these objects, the parameter docs should
# be printed to the command line
[name, success] = ui.getItem(
    "Choose instance", "Choose actuator or dataIO instance", varnames, editable=False
)

if success:
    parse_parameters(globals()[name])
