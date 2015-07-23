import fnmatch
import os
import create_plugin_doc

try:
    if (defaultDir is None):
        defaultDir = ""
except:
    defaultDir = ""

defaultDir = itom.ui.getExistingDirectory("plugin build folder", defaultDir)

matches = []
for root, dirnames, filenames in os.walk(defaultDir):
    for filename in fnmatch.filter(filenames, '*.cfg'):
        matches.append(os.path.join(root, filename))

buildernames = ["qthelp"] #["qthelp", "htmlhelp", "latex", "html"]

for cfg in matches:
    print("create plugin documentation for", cfg)
    try:
        create_plugin_doc.createPluginDoc(cfg, buildernames)
    except:
        print("Error", cfg)
