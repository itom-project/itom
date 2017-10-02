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
import inspect, time, sys, os, pkgutil, re, types, sqlite3, docutils.core, keyword
from sphinx import environment
from sphinx.application import Sphinx
import itom
import pprint
import docutils
from sphinx import addnodes

docutils.nodes.NodeVisitor.optional = () #'pending_xref', 'tabular_col_spec', 'autosummary_table', 'autosummary_toc', 'displaymath', 'only', 'toctree')

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

# --->>>Global simple Settings<<<---

# Filename and DB-Name
databasename = 'numpy'

name = databasename

# Always increase the version by one to allow automatic updates
dbVersion = '2'

# Only change if there was a SchemeID change in Itom
itomMinVersion = '1'            # SchemeID

# --->>>Global advanced Settings<<<---

# Name of the Database
id = idDict[name]

# Date is automatically detected
date = time.strftime("%d.%m.%Y")

# Enter modules you ant to add manually
# manualList = ['itom'] # advanced settings

# which kinds of modules do you want to document # advanced settings
#add_builtins =                      1      # e.g. open()
#add_builtin_modules =           1      # e.g. sys
#add_package_modules =        1      # modules which are directories with __init__.py files
#add_manual_modules =          0      # modules from manuallist

# This is how the DB is named! For singlepackage databases use their name as filename!
#name = 'builtins' # advanced settings




if (databasename == 'itom') or (databasename == 'numpy') or (databasename == 'scipy') or (databasename == 'matplotlib'):
    manualList = [databasename]
    add_builtins =               0      
    add_builtin_modules =        0      
    add_package_modules =        0  
    add_manual_modules =         1
    if databasename != 'numpy':
        blacklist += ['numpy', ]
    print(name)
elif databasename == 'builtins':
    add_builtins =                   1
    add_builtin_modules =        1
    add_package_modules =     0
    add_manual_modules =       0
    blacklist += ['itom','matlab','itomEnum','itomSyntaxCheck','itomUi']
    print(name)
else:
    add_builtins =                0
    add_builtin_modules =     0
    add_package_modules =  0
    add_manual_modules =    0
    print('no source selected')


# If you would like to have a Report File for your Databasecreation
createReportFile = 0
reportFile = open("HelpReport.txt","w")
# and what do you want to be in that file:
reportE = 0
reportW = 0
oldPercentage = 0


#------------------------------------------------------------------------------------------------------------------
#-------------------------------Don�t make any changes below this Line-------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------



# Filename is created by name and .db
filename = "%s_v%s.db" % (name, dbVersion)

# finished db-Info
dbInfo = [id, name, dbVersion, date, itomMinVersion]

remove_all_double_underscore = 1  # ignore private methods, variables... that start with two underscores




# This is the path, where the file will be copied after it´s created
autocopy = 1
destination = os.path.abspath(os.path.join( itom.getAppPath(), "help" ))

# remove this import if numpy is not used, else leave it here, because of ufunc in getPyType()
import numpy
import time

from sphinx import locale
locale.init([], "en")


#def textDummy(text):
    #'''overwrites local._ since this uses a translated text that is not available
    #in this context'''
    #return str(text)
#
#locale._ = textDummy

from docutils import languages
language = languages.get_language("en")
language.labels.update(locale.admonitionlabels)

from sphinxext import numpydoc


stackList = []
builtinList = []
ns = {}
doubleID = {}
idList = {}
doclist = []

def printPercent(value, maxim):
    global oldPercentage
    percentage = value/maxim*100
    if round(percentage) > round(oldPercentage):
        print("Writing to DB progress: %d%%   \r" % (percentage))
        oldPercentage = percentage
    return


def closeReport():
    if createReportFile:
        reportFile.write('Timestamp: '+time.asctime(time.localtime())+'\n')
        reportFile.write('Warnings:  '+format(reportW)+'\n')
        reportFile.write('Errors:    '+format(reportE)+'\n')
        reportFile.close()
    return

def reportMessage(message, typ):
    global reportW
    global reportE
    global reportFile
    if typ == 'w':
        reportW += 1
        reportFile.write('Warning'+format(reportW)+': '+message+'\n')
    else:
        reportE += 1
        reportFile.write('Error'+format(reportW)+': '+message+'\n')
    return

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
    if isinstance(ns['testFunc'], numpy.ufunc):
        # This only exists in Numpy
        return('5')
    return '6'


def getAllModules(ns):
    # add Python Modules
    exec('import inspect, pkgutil', ns)
    # add built in Modules
    if add_builtins:
        for name in dir(__builtins__):
            if name not in blacklist:
                stackList.append(name)
                builtinList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w')
    if add_builtin_modules:
        for name in sys.builtin_module_names:
            if name not in blacklist:
                stackList.append(name)
                builtinList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul ',name,' existiert nicht','w')
    if add_manual_modules:
        for name in manualList :
            if name not in blacklist:
                stackList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w')
    if add_package_modules:
        # remove the current directory from the list
        selPath = sys.path
        dirsToRemove = ['',os.getcwd().lower(), os.getcwd()]
        for dtr in dirsToRemove:
            if (dtr in selPath):
                selPath.remove(dtr)
        for module_loader, name, ispkg in pkgutil.iter_modules(selPath):
            if name not in blacklist:
                stackList.append(name)
                builtinList.append(name) # this line adds the module to buildin list to set python infront afterwards
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w')
    return len(stackList)

def processModules(ns):
    global stackList
    start_time = time.time()
    while len(stackList)>0:
        
        if time.time() - start_time > 5:
            print("%i items on stack" % len(stackList))
            start_time = time.time()
        
        if stackList[0].split('.')[-1:][0] not in blacklist and stackList[0] not in blacklist:
            if not inspect.ismodule(stackList[0]):
                if not stackList[0].startswith('__') or stackList[0][0] != '_':
                    processName(stackList[0], ns)
                    del stackList[0]
                else:
                    del stackList[0]
            else:
                del stackList[0]
        else:
            del stackList[0]


def processName(moduleP, ns, recLevel = 0):
    print("process module '%s'" % moduleP)
    # extract prefix from name
    global stackList
    prefixP = moduleP.split('.')[:-1]
    module = moduleP.split('.')[-1:][0]
    if len(prefixP) > 0:
        prefix = '.'.join(prefixP)
        prefix = prefix +'.'
    else:
        prefix = ''
     
    if module[0] == '_' and not module.startswith('__'):
        return
    if remove_all_double_underscore and module.startswith('__'):
        return
    if recLevel > 10 or module in keyword.kwlist:
        return
    try:
        exec('mID = id('+prefix+module+')', ns)
    except:
        reportMessage('Module unknown: '+module,'e')
        return
    if ns['mID'] not in idList:
        idList[ns['mID']] = prefix+module
        if prefix+module in blacklist:
            return
        nametype = getPyType(prefix+module, ns)
        try:
            exec('hasdoc = hasattr('+prefix+module+', "__doc__")', ns)
        except:
            reportMessage(prefix+module, 'e')
        
        if ns['hasdoc']:
            exec('doc = '+prefix+module+'.__doc__', ns)
            createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'])
        elif nametype == '6': #const
            exec('doc = str('+prefix+module+')', ns)
            createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'])
        
        try:
            #this object can only have 'childs' if it is either a class, module or package
            if (nametype == '2' or nametype == '3' or nametype == '4'):
                exec('sublist = inspect.getmembers('+prefix+module+')', ns)
                newlist = [prefix+module+'.'+ s[0] for s in ns['sublist']]
                print("add %i members from %s" % (len(newlist), prefix + module))
                stackList += newlist
                # remove the processed items
                return
        except:
            print('Sublist of : '+prefix+module,'e')
    else:
        # Hier werden die Links eingefuegt!!!
        # da diese Objekte im original bereits bestehen!
        if prefix+module in blacklist:
            return
        nametype = getPyType(prefix+module, ns)
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
            createSQLEntry(doclink, prefix, module, '1'+nametype, 0)
        except:
            reportMessage('Error in: '+prefix+module,'e')
        return
    #exec('class test:\n    class config:\n        numpydoc_edit_link = False', ns)

def createSQLEntry(docstrIn, prefix, name, nametype, id):
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
        
    # 5. Shortdescription
    if (id != 0):
        m = re.search(r'->.*?\n',docstr,re.DOTALL)
        if (m != None):
            s = docstr[m.start()+3:m.end()-2]
            line[4] = s
        else:
            line[4] = ''
    else:
        line[4] = 'This Package is only referenced here. Its original position is: \n'
        
    # 6. Doc
    if (id != 0):
        m = re.search(r'.*?\n',docstr,re.DOTALL)
        if (m != None and nametype != '06'):
            # Shortdescription extrahieren (Enposition der S.Desc finden)
            s = docstr[m.end():]
            
            # String in lines aufsplitten
            lines = s.split('\n')
            ns["lines"] = lines
            # numpy docstring korrigieren
            global types
            try:
                exec('numpydoc.mangle_docstrings(SphinxApp,\''+types[int(nametype)]+'\', '+ prefix + name +'.__name__,'+ prefix + name +', None, lines)', ns)
            except Exception as ex:
                reportMessage('Error in createSQLEntry \'numpydoc.mangle_docstrings\' (%s%s): %s' % (prefix, name, str(ex)),'e')
                return
            lines = ns['lines']
            # Linien wieder zusamensetzen
            cor = "\n".join(lines)
            try:
                sout =docutils.core.publish_string(cor, writer_name='html',settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'stylesheet_path':'', 'stylesheet':'', 'env':SphinxApp.env})
            except Exception as ex:
                reportMessage('Error in createSQLEntry \'docutils.core.publish_string\' (%s%s): %s' % (prefix, name, str(ex)),'e')
                return
            line[6] = '0'
            line[5] = itom.compressData(sout)
        elif (nametype == '06'):
            line[5] = itom.compressData('"'+name+'" is a const with the value: '+docstr)
            line[6] = '0'
        else:
            # wenn der String keine Shortdescription hat dann einfach komplett einfügen
            line[5] = itom.compressData(docstr)
            line[6] = '3'
    else:
        # only a link reference
        if (prefix[:prefix.find('.')] in builtinList) or (prefix in builtinList) or (name in builtinList):
            line[5] = itom.compressData(docstr)
            line[6] = '0'
        else:
            line[5] = itom.compressData(docstr)
            line[6] = '0'
    
    # 7 HTML-Error
    # Wiird bereits bei #7 eingetragen
    # Insert commands into doclist
    doclist.append(line)


# writes doclist into the given sql-DB
def createSQLDB(ns):
    #shortIfpossible(ns)
    try:
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        # Create table
        c.execute("DROP TABLE IF EXISTS itomCTL")
        c.execute('''create table itomCTL (type int, prefix text, name text, param text, sdesc text, doc text, htmlERROR int)''')
        j = 0
        for line in doclist:
            j = j + 1
            line
            try:
                c.execute('INSERT INTO itomCTL VALUES (?,?,?,?,?,?,?)',line)
            except:
                print('ERROR_5: at line ['+str(j)+']: ',line)
            printPercent(j,len(doclist))
        # Save (commit) the changes
        conn.commit()
        print("SQL-DB succesful")
    except:
        reportMessage("while writing the SQL-DB", 'e')
    finally:
        c.close()


def createSQLDBinfo(ns, info):
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
        reportMessage("while writing the SQL-DB-Info", 'e')
    finally:
        c.close()

#####################
##### Main Programm #####
#####################


print('')
print('-------------START-------------')





# funktion ispackage in namespace registrieren um sie mit exec ausfuehren zu koennen

types = {2 : 'module', 3 : 'module', 4 : 'class', 5 : 'method', 6 : 'attribute'}

SphinxApp = Sphinx(".", ".", ".", ".", "html")
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
        
        
        
ns['ispackage'] = ispackage

ns['numpydoc'] = numpydoc
ns['sys'] = sys
ns['SphinxApp'] = SphinxApp

# If you want the script to replace the file directly... not possible because of access violation
#filename = 'F:\\itom-git\\build\\itom\\PythonHelp.db'
#delete old Database
#os.remove(filename)



# Collect all Modules

print('-> collecting all python-modules')
c = getAllModules(ns)

print('-> ',c,' Module werden bearbeitet')

# work through them and get the DOC
print('-> getting the documentation')
processModules(ns)

# write the DOC into a DB
print('-> creating the DB')
createSQLDB(ns)
createSQLDBinfo(ns, dbInfo)

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

closeReport()
