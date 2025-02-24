# updates the copyright information for all .cs files
# usage: call recursive_traversal, with the following parameters
# parent directory, old copyright text content, new copyright text content

import os

global excludedir
excludedir = []
excludedirnames = []


def update_source(filename, oldcopyright, copyright):
    """
    Updates the copyright header of a source file.

    This function reads the content of the specified file, checks for the presence of a UTF-8 BOM,
    and updates the copyright header if necessary. If the file starts with the old copyright header,
    it is replaced with the new copyright header. If the file does not start with the new copyright
    header, the new copyright header is prepended to the file content.

    Args:
        filename (str): The path to the source file to be updated.
        oldcopyright (str): The old copyright header to be replaced. If None, no replacement is done.
        copyright (str): The new copyright header to be added.

    Returns:
        None
    """
    utfstr = chr(0xEF) + chr(0xBB) + chr(0xBF)
    # fdata = file(filename,"r+").read()
    with open(filename, "r+", encoding="utf-8") as file:
        fdata = file.read()
    isUTF = False
    if fdata.startswith(utfstr):
        isUTF = True
        fdata = fdata[3:]
    if oldcopyright is not None:
        if fdata.startswith(oldcopyright):
            fdata = fdata[len(oldcopyright) :]
    if not fdata.startswith(copyright):
        print("updating " + filename)
        fdata = copyright + fdata
        if isUTF:
            # file(filename,"w").write(utfstr+fdata)
            with open(filename, "w", encoding="utf-8") as file:
                file.write(utfstr + fdata)
        else:
            # file(filename,"w").write(fdata)
            with open(filename, "w", encoding="utf-8") as file:
                file.write(fdata)


def recursive_traversal(dir, oldcopyright, copyright):
    global excludedir
    fns = os.listdir(dir)
    print("listing " + dir)
    for fn in fns:
        fullfn = os.path.join(dir, fn)
        if fullfn in excludedir or fn in excludedirnames:
            continue
        if os.path.isdir(fullfn):
            recursive_traversal(fullfn, oldcopyright, copyright)
        else:
            if fullfn.endswith(".h"):
                update_source(fullfn, oldcopyright, copyright)
            elif fullfn.endswith(".cpp"):
                update_source(fullfn, oldcopyright, copyright)


# itom main files
excludedir = [""]
excludedirnames = [".svn"]

with open("old_header.txt", "r+", encoding="utf-8") as f:
    oldcright = f.read()
with open("itom_header.txt", "r+", encoding="utf-8") as f:
    cright = f.read()

recursive_traversal("..\\Qitom", oldcright, cright)

# data object, point cloud, folder common, plot (SDK)
excludedir = [""]
excludedirnames = [".svn", "Win32", "x64"]

with open("old_header.txt", "r+", encoding="utf-8") as f:
    oldcright = f.read()
with open("itom_sdk_header.txt", "r+", encoding="utf-8") as f:
    cright = f.read()

recursive_traversal("..\\DataObject", oldcright, cright)
recursive_traversal("..\\PointCloud", oldcright, cright)
recursive_traversal("..\\common", oldcright, cright)
recursive_traversal("..\\plot", oldcright, cright)
