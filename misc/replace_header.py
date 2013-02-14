# updates the copyright information for all .cs files
# usage: call recursive_traversal, with the following parameters
# parent directory, old copyright text content, new copyright text content

import os

def update_source(filename, oldcopyright, copyright):
    utfstr = chr(0xef)+chr(0xbb)+chr(0xbf)
    #fdata = file(filename,"r+").read()
    with open(filename,'r+') as f:
        fdata = f.read()
    isUTF = False
    if (fdata.startswith(utfstr)):
        isUTF = True
        fdata = fdata[3:]
    if (oldcopyright != None):
        if (fdata.startswith(oldcopyright)):
            fdata = fdata[len(oldcopyright):]
    if not (fdata.startswith(copyright)):
        print ("updating "+filename)
        fdata = copyright + fdata
        if (isUTF):
            #file(filename,"w").write(utfstr+fdata)
            with open(filename,'w') as f:
                f.write(utfstr+fdata)
        else:
            #file(filename,"w").write(fdata)
            with open(filename,'w') as f:
                f.write(fdata)
       

def recursive_traversal(dir,  oldcopyright, copyright):
    global excludedir
    fns = os.listdir(dir)
    print ("listing "+dir)
    for fn in fns:
        fullfn = os.path.join(dir,fn)
        if (fullfn in excludedir or fn in excludedirnames):
            continue
        if (os.path.isdir(fullfn)):
            recursive_traversal(fullfn, oldcopyright, copyright)
        else:
            if (fullfn.endswith(".h")):
                update_source(fullfn, oldcopyright, copyright)
            elif (fullfn.endswith(".cpp")):
                update_source(fullfn, oldcopyright, copyright)


# itom main files
excludedir = [""]
excludedirnames = [".svn"]

with open("old_header.txt",'r+') as f:
    oldcright = f.read()
with open("itom_header.txt",'r+') as f:
    cright = f.read()

recursive_traversal("..\\Qitom", oldcright, cright)

# data object, point cloud, folder common, plot (SDK)
excludedir = [""]
excludedirnames = [".svn","Win32","x64"]

with open("old_header.txt",'r+') as f:
    oldcright = f.read()
with open("itom_sdk_header.txt",'r+') as f:
    cright = f.read()

recursive_traversal("..\\DataObject", oldcright, cright)
recursive_traversal("..\\PointCloud", oldcright, cright)
recursive_traversal("..\\common", oldcright, cright)
recursive_traversal("..\\plot", oldcright, cright)