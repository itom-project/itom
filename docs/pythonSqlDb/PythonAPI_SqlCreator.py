# coding=iso-8859-15
'''Python API Sql Creator
---------------------------

This script creates database files for the online script reference of itom.
The final db-file must be placed in the help folder of the itom installation
and can then be selected in the property editor of itom.

Use the following variables to select the part of python, itom or other packages
that should be created
'''

#  Systemstuff
import itom
import pprint

import inspect, time, sys, os, pkgutil, re, types, sqlite3, docutils.core, keyword, html
from sphinx import environment
from sphinx.application import Sphinx
import docutils
from sphinx import addnodes
from sphinx import directives
from sphinx.writers.html import HTMLWriter
from docutils import languages
from sphinxext import numpydoc
from sphinx import environment

# remove this import if numpy is not used, else leave it here, because of ufunc in getPyType()
#import numpy
import time

from sphinx import locale
locale.init([], "en")

docutils.nodes.NodeVisitor.optional = ('only', 'displaymath', 'tabular_col_spec', 'autosummary_table', 'autosummary_toc', 'toctree') #'pending_xref', 'tabular_col_spec', 'autosummary_table', 'autosummary_toc', 'displaymath', 'only', 'toctree')

blacklist = ['this','__future__','argparse','ast','bdb','tkinter','turtle','turtledemo','win32traceutil', 
                     'win32pdh', 'perfmondata', 'tzparse', '__next__',
                     'libqtcmodule-2.2', 'libqtc', 'win32com', 'GDK', 'GTK', 'GdkImlib', 'GtkExtra', 
                     'Gtkinter', 'gtk', 'GTKconst', 'zip_it', 'wx.core', 'wx.lib.analogclock', 
                     'wx.lib.analogclockopts', 'wx.lib.ErrorDialogs', 'wx.lib.ErrorDialogs_wdr',
                     'wx.lib.wxPlotCanvas', 'antigravity', 'idle', 'test_capi', 'autotest', 
                     'itoDebugger', 'itoStream', 'itomDbgWrapper', 'mpl_itom', 'itoDebuger', 'itoFunctions',
                     'itomDbgWrapper', 'pythonStream', 'proxy', 'pkg_resources', '__class__', '__base__',
                     '__add__', '__contains__', '__delitem__', '__call__', '__eq__', '__ge__', '__getattribute__', 
                     '__getitem__', '__gt__', '__iadd__', '__imul__', '__iter__', '__le__', 
                     '__len__', '__lt__', '__mul__', '__delattr__', '__ne__', '__setitem__', '__doc__', '__dict__',
                     '__new__', '__init__', '__name__', '__cached__', '__ior__', '__isub__', '__and__',
                     '__sub__', '__xor__', '__main__','__repr__','itoDebugger','__path__','__file__']

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#-------------------------------------------All Settings start here-------------------------------------------
#------------------------------------------------------------------------------------------------------------------

# if your module is already in here, the id is set automatically. If not, add it:  {...., "myNewModule":1009}
# Make shure that your new ID is not already used by another module!!!
idDict = {"builtins":1000, "itom":1001, "numpy":1002, "scipy":1003, "matplotlib":1004}

#------------------------------------------------------------------------------------------------------------------
#-------------------------------DonŽt make any changes below this Line-------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

# This is the path, where the file will be copied after it is created
autocopy = 1
destination = os.path.abspath(os.path.join( itom.getAppPath(), "help" ))

language = languages.get_language("en")
language.labels.update(locale.admonitionlabels)
if not "versionmodified" in locale.versionlabels:
    locale.versionlabels["versionmodified"] = ''
language.labels.update(locale.versionlabels)


def printPercent(value, maxim, config):
    percentage = value/maxim*100
    if round(percentage) > round(config["oldPercentage"]):
        print("Writing to DB progress: %d%%   \r" % (percentage))
        config["oldPercentage"] = percentage
    return

def openReport(config, name, openReport = True):
    if openReport:
        config["reportFile"] = open("HelpReport_%s.txt" % name,"w")
        config["numWarnings"] = 0
        config["numErrors"] = 0
    else:
        config["reportFile"] = None
        config["numWarnings"] = 0
        config["numErrors"] = 0

def closeReport(config):
    if config["reportFile"]:
        reportFile = config["reportFile"]
        reportFile.write('Timestamp: '+time.asctime(time.localtime())+'\n')
        reportFile.write('Warnings:  '+format(config["numWarnings"])+'\n')
        reportFile.write('Errors:    '+format(config["numErrors"])+'\n')
        reportFile.close()

def reportMessage(message, typ, config):
    if config["reportFile"]:
        reportFile = config["reportFile"]
        if typ == 'w':
            config["numWarnings"] += 1
            reportFile.write('Warning'+format(config["numWarnings"])+': '+message+'\n')
        else:
            config["numErrors"] += 1
            reportFile.write('Error'+format(config["numErrors"])+': '+message+'\n')

def ispackage(obj):
    """Guess whether a path refers to a package directory."""
    if hasattr(obj, '__path__'):
        try:
            for path in obj.__path__:
                if os.path.isdir(path):
                    for ext in ('.py', '.pyc', '.pyo'):
                        if os.path.isfile(os.path.join(path, '__init__' + ext)):
                            return True
        except:
            return False
    return False


def getPyType(path, ns):
    exec('testFunc = '+path,ns)
    if ispackage(ns['testFunc']):
        return('2')
    if inspect.ismodule(ns['testFunc']):
        return('3')
    if inspect.isclass(ns['testFunc']):
        return('4')
    if inspect.ismethod(ns['testFunc']):
        return('5')
        #return('method')
    if inspect.isfunction(ns['testFunc']):
        return('5')
        #return('function')
    if inspect.isgeneratorfunction(ns['testFunc']):
        return('5')
        #return('generatorfunction')
    if inspect.isgenerator(ns['testFunc']):
        return('5')
        #return('generator')
    if inspect.istraceback(ns['testFunc']):
        return('5')
        #return('traceback')
    if inspect.isframe(ns['testFunc']):
        return('5')
        #return('frame')
    if inspect.iscode(ns['testFunc']):
        return('5')
        #return('code')
    if inspect.isbuiltin(ns['testFunc']):
        return('5')
        #return('builtin')
    if inspect.isroutine(ns['testFunc']):
        return('5')
        #return('routine')
    if inspect.isabstract(ns['testFunc']):
        return('5')
        #return('abstract')
    if inspect.ismethoddescriptor(ns['testFunc']):
        return('5')
        #return('methoddescriptor')
    if inspect.isdatadescriptor(ns['testFunc']):
        return('5')
        #return('datadescripor')
    if inspect.isgetsetdescriptor(ns['testFunc']):
        return('5')
        #return('getsetdescriptor')
    if inspect.ismemberdescriptor(ns['testFunc']):
        return('5')
        #return('memberdescriptor')
    if type(ns['testFunc']).__name__ == 'ufunc':
        #if isinstance(ns['testFunc'], numpy.ufunc):
        # This only exists in Numpy
        return('5')
    return '6'


def getAllModules(ns, config, stackList, builtinList):
    # add Python Modules
    exec('import inspect, pkgutil', ns)
    # add built in Modules
    if config["add_builtins"]:
        for name in dir(__builtins__):
            if name not in config["blacklist"]:
                stackList.append(name)
                builtinList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w', config)
    if config["add_builtin_modules"]:
        for name in sys.builtin_module_names:
            if name not in config["blacklist"]:
                stackList.append(name)
                builtinList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul ',name,' existiert nicht','w', config)
    if config["add_manual_modules"]:
        for name in config["manualList"] :
            if name not in config["blacklist"]:
                stackList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w', config)
    if config["add_package_modules"]:
        # remove the current directory from the list
        selPath = sys.path
        dirsToRemove = ['',os.getcwd().lower(), os.getcwd()]
        for dtr in dirsToRemove:
            if (dtr in selPath):
                selPath.remove(dtr)
        for module_loader, name, ispkg in pkgutil.iter_modules(selPath):
            if name not in config["blacklist"]:
                stackList.append(name)
                builtinList.append(name) # this line adds the module to buildin list to set python infront afterwards
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w', config)
    return len(stackList)

def processModules(ns, config, stackList, builtinList, idList, docList, imgList):
    start_time = time.time()
    while len(stackList)>0:
        
        if time.time() - start_time > 5:
            print("%i items on stack" % len(stackList))
            start_time = time.time()
        
        if stackList[0].split('.')[-1:][0] not in config["blacklist"] and stackList[0] not in config["blacklist"]:
            if not inspect.ismodule(stackList[0]):
                if not stackList[0].startswith('__') or stackList[0][0] != '_':
                    processName(stackList[0], ns, config, stackList, builtinList, idList, docList, imgList)
                    del stackList[0]
                else:
                    del stackList[0]
            else:
                del stackList[0]
        else:
            del stackList[0]


def processName(moduleP, ns, config, stackList, builtinList, idList, docList, imgList, recLevel = 0):
    print("process module '%s'" % moduleP)
    # extract prefix from name
    prefixP = moduleP.split('.')[:-1]
    module = moduleP.split('.')[-1:][0]
    if len(prefixP) > 0:
        prefix = '.'.join(prefixP)
        prefix = prefix +'.'
    else:
        prefix = ''
     
    if module[0] == '_' and not module.startswith('__'):
        return
    if config["remove_all_double_underscore"] and module.startswith('__'):
        return
    if recLevel > 10 or module in keyword.kwlist:
        return
    try:
        exec('mID = id('+prefix+module+')', ns)
    except:
        reportMessage('Module unknown: '+module,'e', config)
        return
    
    if prefix+module in config["blacklist"]:
        return
    nametype = getPyType(prefix+module, ns)
    
    if (ns['mID'] not in idList) or nametype == '6':
        idList[ns['mID']] = prefix+module
        try:
            exec('hasdoc = hasattr('+prefix+module+', "__doc__")', ns)
        except:
            reportMessage(prefix+module, 'e', config)
        
        if ns['hasdoc']:
            exec('doc = '+prefix+module+'.__doc__', ns)
            createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'], config, builtinList, docList, imgList, ns)
        elif nametype == '6': #const
            exec('doc = str('+prefix+module+')', ns)
            createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'], config, builtinList, docList, imgList, ns)
        
        try:
            #this object can only have 'childs' if it is either a class, module or package
            if (nametype == '2' or nametype == '3' or nametype == '4'):
                exec('sublist = inspect.getmembers('+prefix+module+')', ns)
                newlist = [prefix+module+'.'+ s[0] for s in ns['sublist']]
                exec('hasdir = hasattr('+prefix+module+', "__dir__")', ns)
                if (ns["hasdir"]):
                    exec('sublist = dir('+prefix+module+')', ns)
                    newlist2 = [prefix + module + '.' + s for s in ns['sublist']]
                    for n in newlist2:
                        if not n in newlist:
                            newlist.append(n)
                types = [getPyType(i, ns) for i in newlist]
                
                newlist = [name for name, _ in sorted(zip(newlist, types), key=lambda pair: pair[1], reverse = True)]
                
                print("add %i members from %s" % (len(newlist), prefix + module))
                stackList += newlist
                # remove the processed items
                return
        except:
            print('Sublist of : '+prefix+module,'e')
    else:
        # Hier werden die Links eingefuegt!!!
        # da diese Objekte im original bereits bestehen!
        try:
            prefixP = idList[ns['mID']].split('.')[:-1]
            moduleL = idList[ns['mID']].split('.')[-1:][0]
            if len(prefixP) > 0:
                prefixL = '.'.join(prefixP)
                prefixL = prefixL +'.'
            else:
                prefixL = ''
            if (prefix[:prefix.find('.')] in builtinList) or (prefix in builtinList) or (module in builtinList):
                doclink = '<a id=\"HiLink\" href=\"itom://python.'+prefixL+moduleL+'\">python.'+prefixL+moduleL+'</a>'
            else:
                doclink = '<a id=\"HiLink\" href=\"itom://'+prefixL+moduleL+'\">'+prefixL+moduleL+'</a>'
            createSQLEntry(doclink, prefix, module, '1'+nametype, 0, config, builtinList, docList, imgList, ns)
        except:
            reportMessage('Error in: '+prefix+module,'e', config)
        return
    #exec('class test:\n    class config:\n        numpydoc_edit_link = False', ns)

def createSQLEntry(docstrIn, prefix, name, nametype, id, config, builtinList, docList, imgList, ns):
    #print(prefix+name)
    # Nametype can be a type or a string... nothing else!!!!
    # create one new Row in Database for every function
    line = [0, 0, 0, 0, 0, 0, 0]
    time.sleep(0.001) #TODO: why
    
    if type(docstrIn) == str:
        docstr = docstrIn
    else:
        docstr = ''
        
    # 1. Typ bestimmen und eintragen
    line[0] = nametype
    
    # 2. prefix
    if (prefix[:prefix.find('.')] in builtinList) or (prefix in builtinList) or (name in builtinList):
        line[1] = 'python.' + prefix + name
    else:
        line[1] = prefix+name
    
    # 3. Name
    if (name != ''):
        line[2] = name
    else:
        line[2] = ''
        
    # 4. Parameter
    
    # Falls ein Befehl laenger als 20 Zeichen ist, klappt die erkennung der Parameter nicht mehr
    m = re.search(r'^.{0:20}[(].*?[)]',docstr[0:min(20000, len(docstr))],re.DOTALL)
    if (m != None):
        s = docstr[m.start():m.end()]
        s2 = s[s.find('('):]
        line[3] = s2
    else:
        line[3] = ''
        
    remainingDocStr = docstr
    
    # 5. Shortdescription
    if (id != 0):
        m = re.search(r'->.*?\n',docstr,re.DOTALL)
        if (m != None):
            s = docstr[m.start()+2:m.end()].strip()
            line[4] = s + "\n"
            remainingDocStr = docstr[m.end():]
        else:
            line[4] = ''
    else:
        line[4] = 'The documentation for this item can be found under: \n'
        
    # 6. Doc
    if (id != 0):
        if (nametype != '06'):
            
            # String in lines aufsplitten
            lines = remainingDocStr.split('\n')
            ns["lines"] = lines
            # numpy docstring korrigieren
            
            types = {2 : 'module', 3 : 'module', 4 : 'class', 5 : 'method', 6 : 'attribute'}
            
            try:
                exec('numpydoc.mangle_docstrings(SphinxApp,\''+types[int(nametype)]+'\', '+ prefix + name +'.__name__,'+ prefix + name +', None, lines)', ns)
            except Exception as ex:
                reportMessage('Error in createSQLEntry \'numpydoc.mangle_docstrings\' (%s%s): %s' % (prefix, name, str(ex)),'e', config)
                return
            lines = ns['lines']
            # Linien wieder zusamensetzen
            cor = "\n".join(lines)
            
            if ".. only" in cor:
                cor = cor.replace(".. only:: latex", "::")
            try:
                doctree = docutils.core.publish_doctree(cor, settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'strict_visitor':False, 'stylesheet_path':'', 'stylesheet':'', 'env':ns["SphinxApp"].env})
                ns["SphinxBuildEnvironment"].resolve_references(doctree, "", ns["SphinxApp"].builder)
                sout = docutils.core.publish_from_doctree(doctree, writer=config["SphinxAppWriter"], writer_name = 'html', settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'strict_visitor':False, 'stylesheet_path':'', 'stylesheet':'', 'env':ns["SphinxApp"].env})
                #sout =docutils.core.publish_string(cor, writer=config["SphinxAppWriter"], writer_name = 'html', settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'strict_visitor':False, 'stylesheet_path':'', 'stylesheet':'', 'env':ns["SphinxApp"].env})
            except Exception as ex:
                try:
                    sout =docutils.core.publish_string(".\n" + cor, writer=config["SphinxAppWriter"], writer_name = 'html', settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'strict_visitor':False, 'stylesheet_path':'', 'stylesheet':'', 'env':ns["SphinxApp"].env})
                except Exception as ex:
                    reportMessage('Error in createSQLEntry \'docutils.core.publish_string\' (%s%s): %s' % (prefix, name, str(ex)),'e', config)
                return
            line[6] = '0'
            extractImages(sout, line[1], imgList)
            line[5] = itom.compressData(sout)
        elif (nametype == '06'):
            exec('value = ' + prefix + name, ns)
            line[5] = itom.compressData(html.escape('constant: \n'+ pprint.pformat(ns["value"])))
            line[6] = '0'
        else:
            # wenn der String keine Shortdescription hat dann einfach komplett einfügen
            extractImages(sout, line[1], imgList)
            line[5] = itom.compressData(html.escape(docstr))
            line[6] = '3'
    else:
        # only a link reference, docstr contains the link
        line[5] = itom.compressData(docstr)
        line[6] = '0'
    
    # 7 HTML-Error
    # Wiird bereits bei #7 eingetragen
    # Insert commands into docList
    docList.append(line)
    
def extractImages(docstr, prefix, imgList):
    basepath = os.path.join(itom.getCurrentPath(), "_images")
    result = re.findall(b'src=\"([a-zA-Z0-9\./]+\.[a-zA-Z0-9]+)\"',docstr,re.DOTALL)
    for path in result:
        path = path.decode('utf8')
        filename = os.path.join(basepath, path)
        if os.path.exists(filename):
            with open(filename, 'rb') as fp:
                content = fp.read()
            imgList.append([prefix, path, content])


# writes doclist into the given sql-DB
def createSQLDB(ns, config, filename, doclist, imglist):
    #shortIfpossible(ns)
    try:
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        # Create table
        c.execute("DROP TABLE IF EXISTS itomCTL")
        c.execute('''CREATE TABLE itomCTL (type int, prefix text, name text, param text, sdesc text, doc text, htmlERROR int)''')
        j = 0
        for line in doclist:
            j = j + 1
            line
            try:
                c.execute('INSERT INTO itomCTL VALUES (?,?,?,?,?,?,?)',line)
            except:
                print('ERROR_5: at line ['+str(j)+']: ',line)
            printPercent(j,len(doclist), config)
        
        c.execute("DROP TABLE IF EXISTS itomIMG")
        c.execute('''CREATE TABLE itomIMG (prefix text, href text, blob text)''')
        j = 0
        for line in imglist:
            j = j + 1
            line
            try:
                c.execute('INSERT INTO itomIMG VALUES (?,?,?)',line)
            except:
                print('ERROR_5: at line ['+str(j)+']: ',line)
            printPercent(j,len(imglist), config)
        
        # Save (commit) the changes
        conn.commit()
        print("SQL-DB succesful")
    except:
        reportMessage("while writing the SQL-DB", 'e')
    finally:
        c.close()


def createSQLDBinfo(ns, info, config, filename):
    #shortIfpossible(ns)
    try:
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        # Create table
        c.execute("DROP TABLE IF EXISTS databaseInfo")
        c.execute('''create table databaseInfo (id int, name text, version text, date text, itomMinVersion text)''')
        c.execute('INSERT INTO databaseInfo VALUES (?,?,?,?,?)',info)
        conn.commit()
        print("SQL-DB-Info succesful")
    except:
        reportMessage("while writing the SQL-DB-Info", 'e', config)
    finally:
        c.close()
        
## pseudo Klasse
#class SphinxApp:
    #builder = None
    #class config:
        #numpydoc_edit_link = False
        #numpydoc_use_plots = False
        #numpydoc_show_class_members = True
        #numpydoc_show_inherited_class_members = True
        #numpydoc_class_members_toctree = True
        #numpydoc_citation_re = '[a-z0-9_.-]+'
        #numpydoc_edit_link = False
    #class sphinx_env_config:
        #show_authors = True
    #env = environment.BuildEnvironment(srcdir = "", doctreedir = "", config = sphinx_env_config)
    #env.read_doc("")

#####################
##### Main Programm #####
#####################

def create_database(databasename, dbVersion, itomMinVersion, idList, dummyBuild = False):
    print('')
    print('-------------START-------------')
    
    # Date is automatically detected
    date = time.strftime("%d.%m.%Y")

    # finished db-Info
    dbInfo = [idDict[databasename], databasename, dbVersion, date, itomMinVersion]
    
    stackList = []
    builtinList = []
    ns = {}
    doclist = []
    imglist = []
    
    config = {"manualList":[], \
        "add_builtins": 0, \
        "add_builtin_modules":0, \
        "add_package_modules":0, \
        "add_manual_modules":0, \
        "blacklist": [], \
        "oldPercentage": 0, \
        "remove_all_double_underscore":1 }  # ignore private methods, variables... that start with two underscores }
    
    openReport(config, databasename, True)
    
    if (databasename == 'itom') or \
       (databasename == 'numpy') or \
       (databasename == 'scipy') or \
       (databasename == 'matplotlib'):
        config["manualList"] = [databasename]
        config["add_builtins"] =               0      
        config["add_builtin_modules"] =        0      
        config["add_package_modules"] =        0  
        config["add_manual_modules"] =         1
        if databasename != 'numpy':
            config["blacklist"] = blacklist + ['numpy', ]
    elif databasename == 'builtins':
        config["add_builtins"] =                   1
        config["add_builtin_modules"] =        1
        config["add_package_modules"] =     0
        config["add_manual_modules"] =       0
        config["blacklist"] = blacklist + ['itom','matlab','itomEnum','itomSyntaxCheck','itomUi']
    else:
        config["add_builtins"] =                0
        config["add_builtin_modules"] =     0
        config["add_package_modules"] =  0
        config["add_manual_modules"] =    0
        print('no source selected')
    
    # funktion ispackage in namespace registrieren um sie mit exec ausfuehren zu koennen

    

    confoverrides = {"html_add_permalinks": ""}
    SphinxApp = Sphinx(".", ".", ".", ".", "html", confoverrides = confoverrides)
    SphinxApp.config.numpydoc_use_plots = False
    SphinxApp.config.numpydoc_edit_link = False
    SphinxApp.config.numpydoc_use_plots = False
    SphinxApp.config.numpydoc_show_class_members = True
    SphinxApp.config.numpydoc_show_inherited_class_members = True
    SphinxApp.config.numpydoc_class_members_toctree = True
    SphinxApp.config.numpydoc_citation_re = '[a-z0-9_.-]+'
    SphinxApp.config.numpydoc_edit_link = False
    #SphinxApp.env.read_doc("")

    SphinxApp.env.temp_data['docname'] = ""
    # defaults to the global default, but can be re-set in a document
    SphinxApp.env.temp_data['default_domain'] = SphinxApp.env.domains.get(SphinxApp.env.config.primary_domain)
    
    SphinxApp.env.settings['input_encoding'] = SphinxApp.env.config.source_encoding
    SphinxApp.env.settings['trim_footnote_reference_space'] = \
        SphinxApp.env.config.trim_footnote_reference_space
    SphinxApp.env.settings['gettext_compact'] = SphinxApp.env.config.gettext_compact
    
    SphinxApp.env.app = SphinxApp
    
    SphinxAppWriter = HTMLWriter(SphinxApp.builder) #use sphinx html parser (math-ext works...)
    #SphinxAppWriter = None #use docutils html parser (internal links work better)
    config["SphinxAppWriter"] = SphinxAppWriter
    
    ns['ispackage'] = ispackage
    
    ns['numpydoc'] = numpydoc
    ns['sys'] = sys
    ns['SphinxApp'] = SphinxApp
    
    ns["SphinxBuildEnvironment"] = environment.BuildEnvironment("","",{})

    # If you want the script to replace the file directly... not possible because of access violation
    #filename = 'F:\\itom-git\\build\\itom\\PythonHelp.db'
    #delete old Database
    #os.remove(filename)
    
    
    # Collect all Modules
    
    print('-> collecting all python-modules')
    c = getAllModules(ns, config, stackList, builtinList)
    
    print('-> ',c,' Module werden bearbeitet')
    
    # work through them and get the DOC
    print('-> getting the documentation')
    processModules(ns, config, stackList, builtinList, idList, doclist, imglist)
    
    # Filename is created by databasename and .db
    filename = "%s_v%s.db" % (databasename, dbVersion)
    
    if not dummyBuild:
        # write the DOC into a DB
        print('-> creating the DB')
        createSQLDB(ns, config, filename, doclist, imglist)
        createSQLDBinfo(ns, dbInfo, config, filename)
        
        print('-> inserted objects: '+str(len(doclist)))
        
        print('-> the db file can be found under:')
        print('    '+filename)
        
        # copy to destination
        if (autocopy):
            print('-> the File is now being copied to the destination folder')
            import shutil
            try:
                shutil.copyfile(filename, os.path.abspath(os.path.join(destination,filename)))
            except:
                print('-> the File could not be copied to: '+destination)
        else:
            print('-> autocopy-File is turned off, for changes go to line 49')
    
    print('--------------DONE--------------')
    
    closeReport(config)


if __name__ == "__main__":
    
    globalIDList = {} #the id list is re-used for all elements in task_list
    '''
    database: filename and db-name, possible values: builtins, itom, numpy, scipy, matplotlib
    dbVersion: Always increase the version by one to allow automatic updates
    itomMinVersion: SchemeID, Only change if there was a SchemeID change in Itom
    dummyBuild: only scan module (and create id list), but do not build documentation
    '''
    
    task_list = []
    #task_list.append({"databasename":"builtins", "dbVersion":"301", "itomMinVersion": "1", "dummyBuild":False})
    #task_list.append({"databasename":"itom", "dbVersion":"301", "itomMinVersion": "1", "dummyBuild":False})
    task_list.append({"databasename":"numpy", "dbVersion":"301", "itomMinVersion": "1", "dummyBuild":False})

    for entry in task_list:
        entry["idList"] = globalIDList
        create_database(**entry)