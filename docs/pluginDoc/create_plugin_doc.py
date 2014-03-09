# coding=iso-8859-15 

# -*- coding: utf-8 -*-

import os
import sys
from os import path
from sphinx.util import format_exception_cut_frames, save_traceback
from sphinx.util.console import darkred, nocolor
import distutils.dir_util
import time

from sphinx.application import Sphinx
from docutils.utils import SystemMessage
import subprocess
import itom
import __main__
import shutil
import glob

def process_docstring(app, what, name, obj, options, lines):
    pass
    
def process_signature(app, what, name, obj, options, signature, return_annotation):
    pass


try:
    defaultFile = confFile
except:
    defaultFile = ""

if (defaultFile is None):
    defaultFile = ""
    
confFile = itom.ui.getOpenFileName("plugin_doc_config.py file", defaultFile, "python file (*.py)")

if (not confFile is None):
    with(open(confFile, "r")) as infile:
        pluginConfiguration = infile.readlines()
        exec("\n".join(pluginConfiguration))

    all_files = True
    filenames = False
    confoverrides = {}
    freshenv = False

    basedir = getCurrentPath()
    srcdir = pluginDocSourceDir #from pluginConfiguration
    confdir = os.path.join(itom.getAppPath(), "SDK/docs/pluginDoc")

    buildernames = ["qthelp"] #["qthelp", "htmlhelp", "latex", "html"]

    for buildername in buildernames:
        outdir = os.path.join(pluginDocBuildDir,buildername)
        doctreedir = os.path.join(pluginDocBuildDir,"doctrees")
        
        helpDict = itom.pluginHelp(pluginDocTarget,True)
        
        
        confoverrides = {"project": pluginDocTarget,
            "copyright": helpDict["author"],
            "project": helpDict["name"],
            "version": str(helpDict["version"]),
            "release": "",
            "master_doc": pluginDocMainDocument }
        
        try:
            os.mkdir(outdir)
        except:
            pass


        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername,
                     confoverrides, sys.stdout, sys.stderr, freshenv)

        app.connect('autodoc-process-docstring',process_docstring)
        app.connect('autodoc-process-signature',process_signature)

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
            [code,text] = itom.ui.msgQuestion("publish","Should the generated source files be copied to the source directory of the plugin", itom.ui.MsgBoxYes | itom.ui.MsgBoxNo)
            if (code == ui.MsgBoxYes):
                #remove old files
                pluginDocGeneratedDir = pluginDocGeneratedDir.replace("/","\\")
                pluginDocInstallDir = pluginDocInstallDir.replace("/","\\")
                
                if (os.path.exists(pluginDocGeneratedDir)):
                    distutils.dir_util.remove_tree(pluginDocGeneratedDir)
                
                if (os.path.exists(pluginDocInstallDir)):
                    distutils.dir_util.remove_tree(pluginDocInstallDir)
                
                #copy files
                fromDir = outdir
                toDir = pluginDocGeneratedDir
                shutil.copytree(fromDir,toDir, ignore=shutil.ignore_patterns("*.js","search.html",".buildinfo"))
                
                #copy content of qthelp subfolder to pluginDocInstallDir
                fromDir = pluginDocGeneratedDir
                toDir = pluginDocInstallDir
                shutil.copytree(fromDir,toDir, ignore=shutil.ignore_patterns("*.js","search.html",".buildinfo"))
                