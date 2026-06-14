# coding=iso-8859-15

"""Parses the stubs file for the itom.algorithms submodule.
This module contains wrapper methods for all algorithms in
algorithm plugins, whose name is a valid Python identifier.
"""

import itom
import re
import hashlib
import os
import sys
import math
import warnings


def generateAlgorithmHash():
    """Generates a md5 checksum over all algorithm plugin names and their versions."""
    h = hashlib.md5()
    plugins = itom.version(dictionary=True, addPluginInfo=True)["plugins"]
    for ap in plugins:
        if plugins[ap]["type"] == "algorithm":
            h.update((";%s-%s" % (ap, plugins[ap]["version"])).encode("utf8"))
    return h.digest()


def parseAlgorithmString(algoItem):
    """Creates the docstring for one algorithm, given by the algoItem dictionary."""
    description = algoItem["description"]
    descriptions = description.split("\n")
    descriptions = ["    " + d for d in descriptions]
    description = "\n".join(descriptions)
    docstring = ""

    if description != "":
        docstring = description[4:]  # remove the first indentation

    funcname = algoItem["name"]

    if "Output Parameters" in algoItem:
        params = algoItem["Output Parameters"]
        if len(params) == 1:
            rettype = params[0]["type"]
        elif len(params) > 1:
            rettype = "Tuple[" + ",".join([p["type"] for p in params]) + "]"
        else:
            rettype = "None"
    else:
        rettype = "None"

    args = []

    if "Mandatory Parameters" in algoItem:
        params = algoItem["Mandatory Parameters"]
        args += [p["name"] + ": " + p["type"] for p in params]

    if "Optional Parameters" in algoItem:
        params = algoItem["Optional Parameters"]
        for p in params:
            idx = p["type"].find(" ")
            if idx > 0:
                p["type"] = p["type"][0:idx]
            # fix to create a valid parameter name
            p["name"] = p["name"].replace(" ", "_")
            if not "value" in p:
                args.append("{name}: {type} = None".format(**p))
            elif p["type"] == "str":
                p["value"] = p["value"].encode("unicode_escape").decode("utf8")
                args.append('{name}: {type} = "{value}"'.format(**p))
            elif p["type"] == "float":
                if math.isinf(p["value"]) or math.isnan(p["value"]):
                    p["value"] = 'float("%s")' % str(p["value"])
                args.append("{name}: {type} = {value}".format(**p))
            else:
                args.append("{name}: {type} = {value}".format(**p))

    arguments = ", ".join(args)

    text = """def %s(%s) -> %s:
    \"\"\"%s
    \"\"\"
    pass""" % (
        funcname,
        arguments,
        rettype,
        docstring,
    )

    return text


def generateAlgorithmStubs():
    """parses all algorithm and returns the full content of the file."""

    algos = itom.filterHelp("", dictionary=1, furtherInfos=1)
    algoItems = []

    for algo in algos:
        if algo.isidentifier():
            algoItems.append(parseAlgorithmString(algos[algo]))

    header = (
        """# coding=iso-8859-15

# algo_hash = %s

from typing import Sequence, Union, Tuple
from itom import dataObject, dataIO, actuator"""
        % generateAlgorithmHash()
    )

    pclVersion = itom.version(dictionary=True)["itom"]["PCL_Version"]
    if re.match(r"^\d+\.\d+(.\d+)?$", pclVersion):
        header += "\nfrom itom import point, pointCloud, polygonMesh"

    header += "\n\n"
    header += "\n\n".join(algoItems)

    return header


def parse_stubs(overwrite: bool = False):
    """entry point method."""
    base_folder: str = os.path.abspath(itom.getAppPath())
    base_folder = os.path.join(base_folder, "itom-packages")
    base_folder = os.path.join(base_folder, "itom-stubs")

    stubs_file: str = os.path.join(base_folder, "algorithms.pyi")

    if sys.hexversion < 0x03050000:
        # Python < 3.5 does not know the typing module.
        # do not generate any stubs.
        if os.path.exists(stubs_file):
            os.remove(stubs_file)
    else:
        algo_hash = generateAlgorithmHash()

        # check if stubs file exists. If so, load the first lines and
        # see if there is a compile_date comment. If the compile data is
        # unchanged, quit. The stubs file is up-to-date.
        uptodate = False
        prefix = "# algo_hash = "

        if os.path.exists(stubs_file):
            with open(stubs_file, "rt") as fp:
                count = 0
                for line in fp:
                    if line.startswith(prefix):
                        line = line[len(prefix) : -1]  # there is a \n at the end
                        if line == algo_hash:
                            uptodate = True
                            break

                    count += 1
                    if count > 5:
                        break

        if uptodate and not overwrite:
            return

        text = generateAlgorithmStubs()

        try:
            if not os.path.exists(base_folder):
                os.makedirs(base_folder)

            with open(stubs_file, "wt") as fp:
                fp.write(text)
        except Exception as ex:
            warnings.warn("Error creating the stubs file: %s" % str(ex), RuntimeWarning)


if __name__ == "__main__":
    parse_stubs(overwrite=True)
