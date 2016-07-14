'''
by Markus Gritsch (gritsch@iue.tuwien.ac.at)

Updated to Python 3 (tested with 3.2) by Osye Pritchett (opritche@gmail.com)
it still needs some work but is mostly functional - 2011/05/13 

'''

# please chose type of API-generation:
scenarios = ['pythonAPI','itom','installedPackages','numpy','scipy','matplotlib']
scenario = 'itom'
filename = scenario + '.api'


# if one of these substrings is found in a specific sys.path directory,
# the modules in this particular directory are not processed
excludedirlist = ['Lightflow', 'OpenGL', 'PythonMagick', 'XRC', 'distutils', 
                      'encodings', 'gnome', 'happydoc', 'idle', 'idlelib', 
                      'idlelib ', 'demos',
                      'lib-tk', 'mx', 'plat-linux-i386', 'plat-win', 'pygame', 
                      'pyglade', 'pythonwin', 'scintilla', 'spe', 'test', 'test ',
                      'win32', 'wxPython', 'turtledemo']

# list of modules which should be excluded
excludemodulelist = ['win32traceutil', 'win32pdh', 'perfmondata', 'tzparse',
                     'libqtcmodule-2.2', 'libqtc',
                     'win32com',
                     'GDK', 'GTK', 'GdkImlib', 'GtkExtra', 'Gtkinter', 'gtk', 'GTKconst',
                     'zip_it',
                     'wx.core', 'wx.lib.analogclock', 'wx.lib.analogclockopts',
                     'wx.lib.ErrorDialogs', 'wx.lib.ErrorDialogs_wdr',
                     'wx.lib.wxPlotCanvas', 'antigravity', 'idle', 'test_capi', 'autotest', 
                     'itoDebugger', 'itoStream', 'itomDbgWrapper'
                     #~ 'two_canvases'
                     ]
                     
# python keywords
keywords = ["False", "None", "True", "and", "as", "assert", "break", 
                     "class", "continue", "def", "del", "elif", "else", 
                     "except", "finally", "for", "from", "global", "if", 
                     "import", "in", "is", "lambda", "nonlocal", "not", "or", 
                     "pass", "raise", "return", "try", "while", "with", "yield"]

# switch for excluding modules whose names begin with _
exclude_modules = 1

# list of modules whose contents should be added as global names
addasgloballist = ['noModules']

# list of modules which are otherwise not accessible
# sourcefile-parsing is NOT done for these modules
# also activate the add_manual_modules-switch below
#manuallist = ['os.path','itom']# ['os.path']
manuallist = ['itom']

# import modules of the following type (the dot must be present!!)
moduletypes = ['.py', '.pyd', '.dll', '.so']

# some switches
add_keywords = 0 # e.g. print
add_builtins = 0 # e.g. open()
add_builtin_modules = 0 # e.g. sys
add_manual_modules =0 # modules from manuallist
add_package_modules = 0 # modules which are directories with __init__.py files
add_other_modules = 0 # all the other modules
consider_class_methods = 1
ignore_private_members = 1


if(scenario == 'pythonAPI'):
    add_keywords = 1
    add_builtins = 1
    add_builtin_modules = 1
    excludemodulelist.append('itom')

elif(scenario == 'installedPackages'):
    add_package_modules = 1
elif(scenario == 'matplotlib'):
    manuallist = ['matplotlib','matplotlib.pyplot']
    add_manual_modules = 1
else: #numpy, scipy, itom
    manuallist = [scenario]
    add_manual_modules = 1

#------------------------------------------------------------------------------

import re, sys, os, types

api = {}

def processName(entryprefix, moduleprefix, name, ns):
    if name in keywords:
        return
    
    if ignore_private_members == 1 and name[0] == '_' and len(name) > 2 and name[1] == '_':
        return
    
    exec('hasdoc = hasattr(' + moduleprefix + name + ', "__doc__")', ns)
    exec('nametype = type(' + moduleprefix + name + ')', ns)
    if ns['hasdoc']:
        exec('doc = ' + moduleprefix + name + '.__doc__', ns)
        pattern = re.compile('^ *' + name + r' *\(.*?\)')
        if ns['doc'] and type(ns['doc']) == str: # 'and'-part added by Peter Schoen <schoen@ZTT.Fh-Worms.DE>
            if pattern.search(ns['doc']):
                #~ print(ns['doc'])
                if entryprefix + name not in api:
                    api[entryprefix + name] = entryprefix + str.strip(str.split(ns['doc'], '\n')[0]) + '\n'
                    #return
            else:
                if ns['nametype'] in [type, types.FunctionType]:
                    api[entryprefix + name] = entryprefix + name + '(??) [doc: ' + str.strip(str.split(ns['doc'], '\n')[0]) + ']' + '\n'
    if entryprefix + name not in api:
        if ns['nametype'] == type:
            api[entryprefix + name] = entryprefix + name + '(??) [class]\n'
        elif ns['nametype'] == types.FunctionType:
            api[entryprefix + name] = entryprefix + name + '(??) [function]\n'
        elif ns['nametype'] == types.ModuleType:
            api[entryprefix + name] = entryprefix + name + ':: [module]\n'
        else:
            api[entryprefix + name] = entryprefix + name + '\n'
    
    if consider_class_methods == 1 and ns['nametype'] == type:
        try:
            if hasattr(eval(entryprefix + name),'__dict__'):
                ns2 = eval(entryprefix + name + '.__dict__')
                for name2 in ns2:
                    processName(entryprefix + name + '.', entryprefix + name + '.', name2, ns)
        except:
            print('no recursive handling possible')


def processModule(module, file=''):
    print(' ', '{:<}'.format(module, 22), ': importing ...', end=' ')
    if module in excludemodulelist:
        print('in exclude list')
        return

    if exclude_modules and (module[0] == '_' or '._' in module):
        print('modulename begins with _')
        return

    #~ if module in addasgloballist:
        #~ entryprefix = ''
    #~ else:
        #~ entryprefix = module + '.'
    entryprefix = module + '.'
    for addasglobal in addasgloballist:
        if module[:len(addasglobal)] == addasglobal:
            entryprefix = module[len(addasglobal)+1:]
            break

    ns = {}
    try:
        exec('import ' + module, ns)
        print('ok,', end=' ')
    except:
        print(sys.exc_info()[0])
        return

    print('processing ...', end=' ')
    try:
        exec('names = dir(%s)' % module, ns)
    except:
        print(sys.exc_info()[0])
        return
    for name in ns['names']:
        processName(entryprefix, module + '.', name, ns)
    print('ok,', end=' ')

    # parse module source file if available

    if file[-3:] != '.py':
        print('no source file')
        return
    print('parsing ...', end=' ')
    try:
        f = open(file, 'rt', encoding='utf-8')
    except:
        f = open(file, 'rt')
    finally:
        print(sys.exc_info()[0])
        return
    contents = f.readlines()
    f.close()

    def_p = re.compile(r'^def (\w*)( *\(.*?\)):')
    class_p = re.compile(r'^class +(\w*)')
    init_p = re.compile(r'^[ \t]+def +__init__\(\w*, *(.*?)\):')
    inclass = 0
    classname = ''
    for line in contents:
        def_m = def_p.search(line)
        if def_m:
            name = def_m.group(1)
            if entryprefix + name in api:
                docindex = str.find(api[entryprefix + name], '[doc:')
                if docindex + 1:
                    doc = ' ' + api[entryprefix + name][docindex:] # trailing \n included
                    api[entryprefix + name] = entryprefix + name + def_m.group(2) + doc
                if api[entryprefix + name] == entryprefix + name + '(??) [function]\n':
                    api[entryprefix + name] = entryprefix + name + def_m.group(2) + '\n'

        if inclass:
            init_m = init_p.search(line)
            if init_m:
                if entryprefix + classname in api:
                    docindex = str.find(api[entryprefix + classname], '[doc:')
                    if docindex + 1:
                        doc = ' ' + api[entryprefix + classname][docindex:] # trailing \n included
                        api[entryprefix + classname] = entryprefix + classname + '(' + init_m.group(1) + ')' + doc
                    if api[entryprefix + classname] == entryprefix + classname + '(??) [class]\n':
                        api[entryprefix + classname] = entryprefix + classname + '(' + init_m.group(1) + ')' + '\n'
                inclass = 0
            if not line[0] in ' \t\n':
                inclass = 0

        class_m = class_p.search(line)
        if class_m:
            inclass = 1
            classname = class_m.group(1)
    print('ok')

def processFolder(folder, prefix=''):
    print('processing', folder, end=' ')
    if folder in excludedirlist:
        print('... in exclude list', end=' ')
        folder = ''
        return
    if os.path.split(folder)[1] in excludedirlist:
        print('... in exclude list', end=' ')
        folder = ''
        return
    if folder == '' or not os.path.isdir(folder):
        return

    entries = os.listdir(folder)
    for entry in entries:
        if add_package_modules and \
           os.path.isdir(folder + os.sep + entry) and \
           os.path.isfile(folder + os.sep + entry + os.sep + '__init__.py'):
            # package
            processFolder(folder + os.sep + entry, prefix=prefix+entry+'.')
            print('-done with', folder + os.sep + entry)
        elif prefix and entry == '__init__.py':
            # modules which are directories with __init__.py files
            # The probing of 'prefix' is unfortunately necessary, because of
            # the incorrect behavior of some packages (e.g. PIL) which add
            # their directory to the searchpath via a .pth file AND are
            # packages because of an __init__.py file.
            module = prefix[:-1]
            file = folder + os.sep + entry
            processModule(module, file)
        elif add_other_modules:
            # normal file-modules
            root, ext = os.path.splitext(entry)
            if not ext in moduletypes:
                continue
            if entry[-9:] == 'module.so':
                module = prefix + entry[:-9]
            else:
                module = prefix + root
            file = folder + os.sep + entry
            processModule(module, file)

#------------------------------------------------------------------------------

# keywords
if add_keywords:
    print('\nadding keywords ...', end=' ')
    keywords = ["False", "None", "True", "and", "as", "assert", "break", 
                     "class", "continue", "def", "del", "elif", "else", 
                     "except", "finally", "for", "from", "global", "if", 
                     "import", "in", "is", "lambda", "nonlocal", "not", "or", 
                     "pass", "raise", "return", "try", "while", "with", "yield"]
    for keyword in keywords:
        api[keyword] = keyword + '\n'
    print('ok')

# __builtins__
if add_builtins:
    print('\nadding __builtins__ ...', end=' ')
    for builtin in dir(__builtins__):
        if builtin == 'print': continue # insert, Heribert 07/25/2009 
        if builtin == 'itoDebugger': continue
        processName(entryprefix = '', moduleprefix = '', name = builtin, ns = {})
    print('ok')

# sys.builtin_module_names
if add_builtin_modules:
    print('\nprocessing builtin modules')
    for module in sys.builtin_module_names:
        processModule(module)

# modules specified in manuallist
if add_manual_modules:
    print('\nprocessing modules specified in manuallist')
    for module in manuallist:
        processModule(module)

# modules from sys.path
if add_package_modules or add_other_modules:
    print('\nprocessing searchpath')
    # avoid duplicated entries in sys.path
    folders = {}
    for folder in sys.path:
        folders[folder] = None

    for folder in list(folders.keys()):
        if folder != os.getcwd():
            processFolder(folder)

#------------------------------------------------------------------------------

# sorting
print('sorting api file ...', end=' ')
apilist = list(api.values())

#exclude itom.clc() from api, since clc should never open the autoCompletion list
#such that pressing Return immediately executes the clc command and does not close the
#auto completion list before.
apilist = [l for l in apilist if not l.startswith('itom.clc')]

apilist.sort()
print('done')

# saving
print('saving api file ...', end=' ')

try:
    import os
    [path,thisFile] = os.path.split(__file__)
    filename = os.path.abspath(os.path.join(path,filename))
except:
    pass

print(filename)
print("\n")
print(apilist)
f = open(filename, 'wt')
f.writelines(apilist)
f.close()
print('done\n')