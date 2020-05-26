# coding=iso-8859-15 

# -*- coding: utf-8 -*-

import os
import sys
from sphinx.util.console import nocolor

from sphinx.application import Sphinx
import itom
import shutil
 

buildernames = ["qthelp"]  # ["qthelp", "htmlhelp", "latex", "html"]


def pathConv(p: str):
    """Converts all backslashes in p to slashes and returns the new string."""
    if os.sep == "\\":
        return p.replace("/", "\\")
    return p


def createPluginDoc(confFile: str, buildernames):
    """Main method."""
    with(open(confFile, "r")) as infile:
        pluginConfiguration = infile.readlines()
        pluginConfiguration = "".join(pluginConfiguration)
        
        cfgDict = {}
        
        if "pluginDocInstallDir" not in pluginConfiguration:
            raise RuntimeError(
                "config file " + confFile +
                "seems not to be a plugin documentation config file")
        exec("".join(pluginConfiguration), globals(), cfgDict)

    all_files = True
    filenames = False
    confoverrides = {}
    freshenv = True  # fresh environment variable, else False
    
    srcdir = cfgDict["pluginDocSourceDir"]  # from pluginConfiguration
    confdir = pathConv(
        os.path.join(itom.getAppPath(),
                     "SDK" + os.sep + "docs" + os.sep + "pluginDoc"))
    
    for buildername in buildernames:
        outdir = pathConv(os.path.join(cfgDict["pluginDocBuildDir"],buildername))
        doctreedir = os.path.join(cfgDict["pluginDocBuildDir"], "doctrees")
        
        if (itom.pluginLoaded(cfgDict["pluginName"])):
            helpDict = itom.pluginHelp(cfgDict["pluginName"], True)
            
            confoverrides = {
                "copyright": helpDict["author"],
                "project": "itom plugin '" + helpDict["name"] + "'",
                "version": helpDict["version"],
                "release": "",
                "html_title": helpDict["name"] + " (" + helpDict["type"] + ")",
                "html_short_title": helpDict["name"],
                "master_doc": cfgDict["pluginDocMainDocument"]}
        
            app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername,
                         confoverrides, sys.stdout, sys.stderr, freshenv)
            
            try:
                os.mkdir(outdir)
            except Exception:
                pass

            nocolor()

            if not app.builder:
                raise RuntimeError

            if all_files:
                app.builder.build_all()
            elif filenames:
                app.builder.build_specific(filenames)
            else:
                app.builder.build_update()
            
            if (buildername == "qthelp"):
                # copy important files from qthelp subfolder to pluginDocInstallDir
                pluginDocInstallDir = pathConv(cfgDict["pluginDocInstallDir"])
                
                if (os.path.exists(pluginDocInstallDir)):
                    shutil.rmtree(pluginDocInstallDir)
                        
                shutil.copytree(
                    outdir,
                    pluginDocInstallDir,
                    ignore=shutil.ignore_patterns("*.js","search.html",".buildinfo"))
        
        else:
            print("Plugin documentation for", cfgDict["pluginName"],
                  "could not be build since not available on this computer")
            

if (__name__ == "__main__"):
    try:
        if defaultConfFile is None:
            defaultConfFile = ""
    except Exception:
        defaultConfFile = ""

    confFile = itom.ui.getOpenFileName("plugin_doc_config.cfg file", 
                                       defaultConfFile, 
                                       "plugin doc config (*.cfg)")
    if confFile is not None:
        defaultConfFile = confFile
        
        createPluginDoc(confFile, buildernames)
