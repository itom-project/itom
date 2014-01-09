#  Systemstuff
import inspect, time, sys, os, pkgutil, re, types, sqlite3, docutils.core, keyword

#


#
import itom

# remove this import if numpy is not used, else leave it here, because of ufunc in getPyType()
import numpy

# some switches
add_builtins =                      0      # e.g. open()
add_builtin_modules =           0      # e.g. sys
add_manual_modules =          1      # modules from manuallist
add_package_modules =        0      # modules which are directories with __init__.py files

remove_all_double_underscore = 1  # Alles was mit zwei Unterstrichen beginnt ignorieren

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


manualList = ['itom']


# This is how the DB is named! For singlepackage databases use their name as filename!
filename = 'PythonHelpNew.db'

stackList = []

ns = {}

doubleID = {}

idList = {}

doclist = []

reportFile = open("HelpReport.txt","w")

reportE = 0
reportW = 0
oldPercentage = 0

def printPercent(value, maxim):
    global oldPercentage
    percentage = value/maxim*100
    if round(percentage) > round(oldPercentage):
        str1 = format(percentage)+'%'
        print("Writing to DB progress: %d%%   \r" % (percentage))
        oldPercentage = percentage
    return


def closeReport():
    #t = 
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
        path = obj.__path__[0]
        if (type(path) is list and len(path) > 0):
            path = path[0]
        if os.path.isdir(path):
            for ext in ('.py', '.pyc', '.pyo'):
                if os.path.isfile(os.path.join(path, '__init__' + ext)):
                    return True
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
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w')
    if add_builtin_modules:
        for name in sys.builtin_module_names:
            if name not in blacklist:
                stackList.append(name)
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
        dirsToRemove = [os.getcwd().lower(), os.getcwd()]
        for dtr in dirsToRemove:
            if (dtr in selPath):
                selPath.remove(dtr)
        for module_loader, name, ispkg in pkgutil.iter_modules(selPath):
            if name not in blacklist:
                stackList.append(name)
                try:
                    exec('import '+name,ns)
                except:
                    reportMessage('Modul '+name+' existiert nicht','w')
    return len(stackList)

def processModules(ns):
    global stackList
    while len(stackList)>0:
        #print(len(stackList))
        if stackList[0].split('.')[-1:][0] not in blacklist and stackList[0] not in blacklist:
            if not inspect.ismodule(stackList[0]):
                if not stackList[0].startswith('__') or stackList[0][0] != '_':
                    try:
                        ret = processName(stackList[0], ns)
                        del stackList[0]
                        if ret is not None:
                            stackList.append(ret)
                    except:
                        del stackList[0]
                        reportMessage(module+stackList[0],'e')
                else:
                    del stackList[0]
            else:
                del stackList[0]
        else:
            del stackList[0]


def processName(moduleP, ns, recLevel = 0):
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
            if ns['hasdoc']:
                exec('doc = '+prefix+module+'.__doc__', ns)
                if (nametype == '6'): # '6' = const
                    exec('doc = str('+prefix+module+')', ns)
                    createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'])
                else:
                    createSQLEntry(ns['doc'], prefix, module, '0'+nametype,ns['mID'])
        except:
            reportMessage(prefix+module, 'e')
        try:
            #this object can only have 'childs' if it is either a class, module or package
            if (nametype == '2' or nametype == '3' or nametype == '4'):
                exec('sublist = inspect.getmembers('+prefix+module+')', ns)
                stackList += [prefix+module+'.'+ s[0] for s in ns['sublist']]
                # remove the processed items
                return
        except:
            print('Sublist of : '+prefix+module,'e')
    else:
        # Hier werden die Links eingefügt!!!
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
            doclink = '<a id=\"HiLink\" href=\"itom://'+prefixL+moduleL+'\">'+prefixL+moduleL+'</a>'
            createSQLEntry(doclink, prefix, module, '1'+nametype, 0)
        except:
            reportMessage('Error in: '+prefix+module,'e')
        return

    exec('class test:\n    class config:\n        numpydoc_edit_link = False', ns)

def createSQLEntry(docstrIn, prefix, name, nametype, id):
    #print(prefix+name)
    # Nametype can be a type or a string... nothing else!!!!
    # create one new Row in Database for every function
    line = [0, 0, 0, 0, 0, 0, 0, 0]
    
    # 0 ID eintragen, jedoch später nicht in DB
    line[0] = id
    
    if type(docstrIn) == str:
        docstr = docstrIn
    else:
        docstr = ''
        
    # 1. Typ bestimmen und eintragen
    line[1] = nametype
    
    # 2. prefix
    line[2] = prefix[:len(prefix)]+name
    
    # 3 prefixL (Lowercase for quick search)
    #prefixL = prefix[:len(prefix)]+name
    #line[3] = prefixL.lower()
    
    # 4. Name
    if (name != ''):
        line[3] = name
    else:
        line[3] = ''
        
    # 5. Parameter
    # Falls ein Befehl länger als 20 Zeichen ist, klappt die erkennung der Parameter nicht mehr
    m = re.search(r'^.{0:20}[(].*?[)]',docstr,re.DOTALL)
    if (m != None):
        s = docstr[m.start():m.end()]
        s2 = s[s.find('('):]
        line[4] = s2
    else:
        line[4] = ''
        
    # 6. Shortdescription
    if (id != 0):
        m = re.search(r'->.*?\n',docstr,re.DOTALL)
        if (m != None):
            s = docstr[m.start()+3:m.end()-2]
            line[5] = s
        else:
            line[5] = ''
    else:
        line.append('This Package is only referenced here. It´s original position is: \n')
        
    # 7. Doc
    if (id != 0):
        m = re.search(r'.*?\n',docstr,re.DOTALL)
        if (m != None and nametype != '06'):
            # Shortdescription extrahieren (Enposition der S.Desc finden)
            s = docstr[m.end():]
            try:
                # -------Test Block--------
                # String in lines aufsplitten
                
                lines = s.split('\n')
                ns["lines"] = lines
                # numpy docstring korrigieren
                global types
                exec('numpydoc.mangle_docstrings(test,\''+types[int(nametype)]+'\', '+line[2]+'.__name__,'+line[2]+', None, lines)', ns)
                lines = ns['lines']
                # Linien wieder zusamensetzen
                cor = "\n".join(lines)
                # ---Ende des Testblocks---
                
                sout =docutils.core.publish_string(cor, writer_name='html',settings_overrides = {'report_level': 5, 'embed_stylesheet': 0, 'stylesheet_path':'', 'stylesheet':''})
                line[7] = '0'
            except:
                sout = s
                line[7] = '1'
            line[6] = itom.compressData(sout)
        elif (nametype == '06'):
            line[6] = itom.compressData('"'+name+'" is a const with the value: '+docstr)
            line[7] = '4'
        else:
            # wenn der String keine Shortdescription hat dann einfach komplett einfügen
            line[6] = itom.compressData(docstr)
            line[7] = '3'
    else:
        # only a link reference
        line[6] = itom.compressData(docstr)
        line[7] = '2'
    
    # 8 HTML-Error
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
            lineS = line[1:8]
            try:
                c.execute('INSERT INTO itomCTL VALUES (?,?,?,?,?,?,?)',lineS)
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


#####################
##### Main Programm #####
#####################


print('')
print('-------------START-------------')

sys.path.append('D:/ITOM/source/itom/docs/userDoc/source/sphinxext')
import numpydoc
import sys



# funktion ispackage in namespace registrieren um sie mit exec ausführen zu können

types = {2 : 'module', 3 : 'module', 4 : 'class', 5 : 'method', 6 : 'attribute'}

# pseudo Klasse
class test:
    class config:
        numpydoc_edit_link = False
        
ns['ispackage'] = ispackage

ns['numpydoc'] = numpydoc
ns['sys'] = sys
ns['test'] = test

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

print('-> inserted objects: '+str(len(doclist)))

print('-> the db file can be found under:')
print('    '+filename)

print('--------------DONE--------------')

closeReport()