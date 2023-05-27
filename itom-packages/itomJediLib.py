"""This module builds a bridge between the
autocompletion and static analysis package jedi
and the GUI of itom.

License information:

itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public Licence as published by
the Free Software Foundation; either version 2 of the Licence, or (at
your option) any later version.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public Licence for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
"""

import jedi
import sys
import itomStubsGen
import itomAlgorithmsStubsGen
import warnings
from contextlib import contextmanager

__version__ = "1.2.0"

# avoid stack overflow in itom (jedi sometimes sets a recursionlimit of 3000):
maxreclimit = 1800

if jedi.__version__ >= "0.12.0":
    jedienv = jedi.api.environment.InterpreterEnvironment()

ICON_CLASS = ("code-class", ":/classNavigator/icons/class.png")
ICON_FUNC = ("code-function", ":/classNavigator/icons/method.png")
ICON_FUNC_PRIVATE = (
    "code-function",
    ":/classNavigator/icons/method_private.png",
)
ICON_FUNC_PROTECTED = (
    "code-function",
    ":/classNavigator/icons/method_protected.png",
)
ICON_NAMESPACE = ("code-context", ":/classNavigator/icons/namespace.png")
ICON_VAR = ("code-variable", ":/classNavigator/icons/var.png")
ICON_KEYWORD = ("quickopen", ":/classNavigator/icons/var.png")

# parses the stubs file for the itom module (if not up-to-date)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    itomStubsGen.parse_stubs()
    itomAlgorithmsStubsGen.parse_stubs()


@contextmanager
def reduceRecursionLimit():
    """Temporarily limits the sys.recursionlimit to the global variable maxreclimit."""
    currentlimit = sys.getrecursionlimit()
    try:
        global maxreclimit
        if maxreclimit < currentlimit:
            sys.setrecursionlimit(maxreclimit)
        yield
    finally:
        sys.setrecursionlimit(currentlimit)


class StreamHider:
    """A stream class, that emits nothing.

    The stderr output of jedi is redirected to this
    stream in order not display unwanted background
    errors or warnings in itom.
    """

    def __init__(self, channels=("stdout",)):
        self._orig = {ch: None for ch in channels}

    def __enter__(self):
        for ch in self._orig:
            self._orig[ch] = getattr(sys, ch)
            setattr(sys, ch, self)
        return self

    def write(self, string):
        pass

    def flush(self):
        pass

    def __exit__(self, *args):
        for ch in self._orig:
            setattr(sys, ch, self._orig[ch])


def icon_from_typename(name, icon_type):
    """
    Returns the icon resource filename that corresponds to the given typename.
    :param name: name of the completion. Use to make the distinction between
        public and private completions (using the count of starting '_')
    :pram typename: the typename reported by jedi
    :returns: The associate icon resource filename or None.
    """
    ICONS = {
        "CLASS": ICON_CLASS,
        "IMPORT": ICON_NAMESPACE,
        "STATEMENT": ICON_VAR,
        "FORFLOW": ICON_VAR,
        "FORSTMT": ICON_VAR,
        "WITHSTMT": ICON_VAR,
        "GLOBALSTMT": ICON_VAR,
        "MODULE": ICON_NAMESPACE,
        "KEYWORD": ICON_KEYWORD,
        "KEYWORD-PRIV": ICON_KEYWORD,
        "KEYWORD-PROT": ICON_KEYWORD,
        "PARAM": ICON_VAR,
        "PARAM-PRIV": ICON_VAR,
        "PARAM-PROT": ICON_VAR,
        "PROPERTY": ICON_VAR,
        "PROPERTY-PRIV": ICON_VAR,
        "PROPERTY-PROT": ICON_VAR,
        "ARRAY": ICON_VAR,
        "INSTANCEELEMENT": ICON_VAR,
        "INSTANCE": ICON_VAR,
        "FUNCTION": ICON_FUNC,
        "DEF": ICON_FUNC,
        "FUNCTION-PRIV": ICON_FUNC_PRIVATE,
        "FUNCTION-PROT": ICON_FUNC_PROTECTED,
    }
    ret_val = ""
    icon_type = icon_type.upper()
    # jedi 0.8 introduced NamedPart class, which have a string instead of being
    # one
    if hasattr(name, "string"):
        name = name.string
    if icon_type == "FORFLOW" or icon_type == "STATEMENT":
        icon_type = "PARAM"
    if icon_type == "PARAM" or icon_type == "FUNCTION":
        if name.startswith("__"):
            icon_type += "-PRIV"
        elif name.startswith("_"):
            icon_type += "-PROT"
    if icon_type in ICONS:
        ret_val = ICONS[icon_type][1]
    elif icon_type:
        pass
        # _logger().warning("Unimplemented completion icon_type: %s", icon_type)
    return ret_val


def calltipModuleItomModification(sig, params):
    """mod that returns the call signature for all
    methods and classes in the itom module
    based on their special docstrings
    """
    try:
        doc = sig.docstring(raw=False, fast=True)
    except Exception:
        return None

    arrow_idx = doc.find("->")

    if arrow_idx == -1 or not doc.startswith(sig.name):
        return None

    signature = doc[len(sig.name) : arrow_idx].strip()
    signature = signature[1:-1]
    parts = signature.split(",")

    if len(parts) == len(params):
        return parts
    elif (
        len(params) >= 1 and params[0].name == "self" and len(parts) == len(params) - 1
    ):
        return [
            "self",
        ] + parts
    else:
        return None


calltipsModificationList = {"itom": calltipModuleItomModification}


def calltips(code, line, column, path=None):
    """ """
    max_calltip_line_length = 120

    with reduceRecursionLimit():
        with warnings.catch_warnings():
            # avoid UserWarnings if an imported C-Module could not be loaded yet
            warnings.simplefilter("ignore")

            if jedi.__version__ >= "0.17.0":
                script = jedi.Script(code=code, path=path, environment=jedienv)
                signatures = script.get_signatures(line=line + 1, column=column)
            elif jedi.__version__ >= "0.16.0":
                script = jedi.Script(
                    source=code, path=path, encoding="utf-8", environment=jedienv
                )
                signatures = script.get_signatures(line=line + 1, column=column)
            else:
                if jedi.__version__ >= "0.12.0":
                    script = jedi.Script(
                        code,
                        line + 1,
                        column,
                        path,
                        encoding="utf-8",
                        environment=jedienv,
                    )
                else:
                    script = jedi.Script(code, line + 1, column, path, encoding="utf-8")
                signatures = script.call_signatures()

            result = []

        for sig in signatures:
            index = sig.index

            if index is None:
                index = -1

            # create a formatted calltip (current index appear in bold)
            module_name = str(sig.module_name)
            call_name = str(sig.name)
            paramlist = None

            if module_name in calltipsModificationList:
                paramlist = calltipsModificationList[module_name](sig, sig.params)

            if paramlist is None:
                paramlist = [p.description for p in sig.params]

            # remove the prefix "param " from every parameter (if it exists)
            pkwd = "param "
            pkwdlen = len(pkwd)
            for pidx in range(len(paramlist)):
                if paramlist[pidx].startswith(pkwd):
                    paramlist[pidx] = paramlist[pidx][pkwdlen:]

            if index >= 0 and index < len(paramlist):
                paramlist[index] = "<b>%s</b>" % paramlist[index]

            if module_name != "":
                method_name = "%s.%s" % (module_name, call_name)
            else:
                method_name = call_name

            result.append(
                (
                    method_name,
                    paramlist,
                    column,
                    sig.bracket_start[0],
                    sig.bracket_start[1],
                )
            )

        return result


def completions(code, line, column, path, prefix):
    """ """
    with reduceRecursionLimit():
        with warnings.catch_warnings():
            # avoid UserWarnings if an imported C-Module could not be loaded yet
            warnings.simplefilter("ignore")

            if jedi.__version__ >= "0.17.0":
                script = jedi.Script(code=code, path=path, environment=jedienv)
                completions = script.complete(line=line + 1, column=column)
            elif jedi.__version__ >= "0.16.0":
                script = jedi.Script(
                    source=code, path=path, encoding="utf-8", environment=jedienv
                )
                completions = script.complete(line=line + 1, column=column)
            else:
                if jedi.__version__ >= "0.12.0":
                    script = jedi.Script(
                        code,
                        line + 1,
                        column,
                        path,
                        encoding="utf-8",
                        environment=jedienv,
                    )
                else:
                    script = jedi.Script(code, line + 1, column, path, encoding="utf-8")
                completions = script.completions()

        result = []

        # the following pairs of [name, type] will not be returned as possible completion
        blacklist = [
            ["and", "keyword"],
            ["if", "keyword"],
            ["in", "keyword"],
            ["is", "keyword"],
            ["not", "keyword"],
            ["or", "keyword"],
        ]

        # disable error stream to avoid import errors of jedi,
        # which are directly printed to sys.stderr (no exception)
        with StreamHider(("stderr",)) as h:
            for completion in completions:
                if [completion.name, completion.type] in blacklist:
                    continue
                try:
                    desc = completion.description
                    compl_type = completion.type

                    if jedi.__version__ >= "0.16.0":
                        if compl_type == "property":
                            tooltipList = name_tooltip_type_property(completion)
                        else:
                            signatures = completion.get_signatures()
                            if len(signatures) == 0:
                                tooltip = completion.docstring()
                                if tooltip != "":
                                    tooltip = "%s\n\n%s" % (
                                        completion.name,
                                        tooltip,
                                    )
                                tooltipList = [
                                    tooltip,
                                ]
                            elif len(signatures) == 1:
                                tooltip = signatures[0].docstring()
                                # for some properties, signatures[0].docstring() only
                                # contains the return value, but no description, fall back
                                # to completion.docstring() then...
                                if "\n\n" not in tooltip:
                                    tooltip = completion.docstring()
                                # workaround: there seems to be a bug in jedi for
                                # properties that return something with None: NoneType()
                                # is always returned in doc. However, completion.get_type_hint()
                                # returns the real rettype hint. Replace it.
                                # see also: https://github.com/davidhalter/jedi/issues/1695
                                pattern = "NoneType()\n"
                                if tooltip.startswith(pattern):
                                    if jedi.__version__ >= "0.17.0":
                                        rettype = completion.get_type_hint()
                                        if rettype != "":
                                            tooltip = (
                                                rettype
                                                + ": "
                                                + tooltip[len(pattern) :].lstrip()
                                            )
                                    else:
                                        # jedi < 0.17.0 does not have the get_type_hint() method
                                        tooltip = tooltip[len(pattern) :].lstrip()
                                tooltipList = [
                                    tooltip,
                                ]
                            elif len(signatures) > 1:
                                # only use unique signatures
                                docstrings = [
                                    signatures[0].docstring(),
                                ]
                                for s in signatures[1:]:
                                    d = s.docstring()
                                    if d != docstrings[0]:
                                        docstrings.append(d)
                                tooltipList = [d for d in docstrings]
                            else:
                                tooltip = completion.docstring()
                                if tooltip != "":
                                    # if tooltip is empty, use desc as tooltip (done in C++)
                                    if jedi.__version__ >= "0.17.0":
                                        type_hint = completion.get_type_hint()
                                        if type_hint != "" and not tooltip.startswith(
                                            type_hint
                                        ):
                                            tooltip = type_hint + " : " + tooltip
                                    tooltipList = [
                                        tooltip,
                                    ]
                                else:
                                    tooltipList = [
                                        desc,
                                    ]
                    else:
                        tooltipList = [
                            completion.docstring(),
                        ]

                    if compl_type == "function" and len(tooltipList) > 0:
                        """Properties, defined in C, are displayed as funtion.
                        However, if the tooltip starts with 'type : text', it
                        is likely to be a property"""
                        text = tooltipList[0]
                        colon_idx = text.find(":")
                        bracket_idx = text.find("(")

                        if bracket_idx == -1:
                            compl_type = "keyword"
                        elif colon_idx >= 0 and colon_idx < bracket_idx:
                            compl_type = "keyword"

                    result.append(
                        (
                            completion.name,
                            desc,
                            icon_from_typename(completion.name, compl_type),
                            tooltipList,
                        )
                    )
                except Exception as ex:
                    break  # todo, check this further

        return result


def goto_assignments(code, line, column, path, mode=0, encoding="utf-8"):
    """
    mode: 0: goto definition, 1: goto assignment (no follow imports), 2: goto assignment (follow imports)
    """
    with reduceRecursionLimit():
        with warnings.catch_warnings():
            # avoid UserWarnings if an imported C-Module could not be loaded yet
            warnings.simplefilter("ignore")

            if jedi.__version__ >= "0.16.0":
                if jedi.__version__ >= "0.17.0":
                    script = jedi.Script(code=code, path=path, environment=jedienv)
                else:
                    script = jedi.Script(
                        source=code, path=path, encoding="utf-8", environment=jedienv
                    )

                try:
                    if mode == 0:
                        assignments = script.infer(
                            line=line + 1, column=column, prefer_stubs=False
                        )
                    elif mode == 1:
                        assignments = script.goto(
                            line=line + 1,
                            column=column,
                            follow_imports=False,
                            prefer_stubs=False,
                        )
                    else:
                        assignments = script.goto(
                            line=line + 1,
                            column=column,
                            follow_imports=True,
                            prefer_stubs=False,
                        )
                except Exception:
                    assignments = []

            else:
                if jedi.__version__ >= "0.12.0":
                    script = jedi.Script(
                        code,
                        line + 1,
                        column,
                        path,
                        encoding="utf-8",
                        environment=jedienv,
                    )
                else:
                    script = jedi.Script(code, line + 1, column, path, encoding="utf-8")

                try:
                    if mode == 0:
                        assignments = script.goto_definitions()
                    elif mode == 1:
                        assignments = script.goto_assignments(follow_imports=False)
                    else:
                        assignments = script.goto_assignments(follow_imports=True)
                except Exception:
                    assignments = []

        result = []
        for assignment in assignments:
            if (
                assignment.full_name
                and assignment.full_name != ""
                and (
                    assignment.module_path is None
                    or not str(assignment.module_path).endswith("pyi")
                )
            ):
                result.append(
                    (
                        str(assignment.module_path)
                        if assignment.module_path is not None
                        else "",
                        assignment.line - 1 if assignment.line else -1,
                        assignment.column if assignment.column else -1,
                        assignment.full_name,
                    )
                )

        if len(result) == 0 and len(assignments) > 0 and mode == 0:
            # instead of 'infer' try 'goto' instead
            result = goto_assignments(
                code, line, column, path, mode=1, encoding=encoding
            )

        return result


def name_tooltip_type_module(item):
    """Generates a description text for a given item, whose type is 'module'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    heading = "Module %s" % (item.name,)
    body = item.docstring()
    if body is not None and body != "":
        tooltip = "%s\n\n%s" % (heading, body)
    else:
        tooltip = heading
    return [
        tooltip,
    ]


def name_tooltip_type_statement(item):
    """Generates a description text for a given item, whose type is 'statement'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    typehint = item.get_type_hint()
    if typehint != "":
        heading = "%s: %s" % (item.name, typehint)
    else:
        heading = item.name

    body = item.docstring()
    if body is not None and body != "":
        tooltip = "%s\n\n%s" % (heading, body)
    else:
        tooltip = heading
    return [
        tooltip,
    ]


def name_tooltip_type_instance(item):
    """Generates a description text for a given item, whose type is 'instance'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    return [
        item.docstring(),
    ]


def name_tooltip_type_param(item):
    """Generates a description text for a given item, whose type is 'param'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    return [
        item.description,
    ]


def name_tooltip_type_property(item):
    """Generates a description text for a given item, whose type is 'property'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    name = item.name
    rettype = item.get_type_hint()
    docstring = item.docstring()

    if docstring != "":
        return [
            "%s: %s\n\n%s" % (name, rettype, docstring),
        ]
    else:
        return [
            "%s: %s" % (name, rettype),
        ]


def name_tooltip_type_general(item):
    """Generates a description text for a given item, whose type is any other.

    Any other type is any type, that is not covered by the specific methods
    name_tooltip_type_<specific_type>, for example 'class', 'function', 'path'.

    The description text can consist of one or more possible strings, each
    one having a format for a tooltip. The first line is usually considered
    to be a headline, followed by two newline characters and a multiline description
    string.

    Parameters
    ----------
    item : jedi.api.classes.Name
        is the item whose tooltip docstring should be generated.

    Returns
    -------
        list of str
            One or multiple possible tooltips for the given item.
    """
    desc = item.description

    if jedi.__version__ >= "0.16.0":
        signatures = item.get_signatures()
        if len(signatures) == 0:
            tooltip = item.docstring()
            name = item.name
            if jedi.__version__ < "0.18.0":
                # properties are also reported as type function
                rettype = item.get_type_hint()
                if rettype != "":
                    name += ": %s" % rettype
            if tooltip != "":
                tooltip = "%s\n\n%s" % (name, tooltip)
            else:
                tooltip = name
            tooltipList = [
                tooltip,
            ]
        elif len(signatures) == 1:
            tooltip = signatures[0].docstring()
            # for some properties, signatures[0].docstring() only
            # contains the return value, but no description, fall back
            # to item.docstring() then...
            if "\n\n" not in tooltip:
                tooltip = item.docstring()
            # workaround: there seems to be a bug in jedi for
            # properties that return something with None: NoneType()
            # is always returned in doc. However, item.get_type_hint()
            # returns the real rettype hint. Replace it.
            # see also: https://github.com/davidhalter/jedi/issues/1695
            pattern = "NoneType()\n"
            if tooltip.startswith(pattern):
                if jedi.__version__ >= "0.17.0":
                    rettype = item.get_type_hint()
                    if rettype != "":
                        tooltip = rettype + ": " + tooltip[len(pattern) :].lstrip()
                else:
                    # jedi < 0.17.0 does not have the get_type_hint() method
                    tooltip = tooltip[len(pattern) :].lstrip()
            tooltipList = [
                tooltip,
            ]
        elif len(signatures) > 1:
            # only use unique signatures
            docstrings = [
                signatures[0].docstring(),
            ]
            for s in signatures[1:]:
                d = s.docstring()
                if d != docstrings[0]:
                    docstrings.append(d)
            tooltipList = [d for d in docstrings]
        else:
            tooltip = item.docstring()
            if tooltip != "":
                # if tooltip is empty, use desc as tooltip (done in C++)
                if jedi.__version__ >= "0.17.0":
                    type_hint = item.get_type_hint()
                    if type_hint != "" and not tooltip.startswith(type_hint):
                        tooltip = type_hint + " : " + tooltip
                tooltipList = [
                    tooltip,
                ]
            else:
                tooltipList = [
                    desc,
                ]
    else:
        tooltipList = [
            item.docstring(),
        ]

    return tooltipList


def get_help(code, line, column, path):
    """ """
    with reduceRecursionLimit():
        with warnings.catch_warnings():
            # avoid UserWarnings if an imported C-Module could not be loaded yet
            warnings.simplefilter("ignore")

            if jedi.__version__ >= "0.17.0":
                script = jedi.Script(code=code, path=path, environment=jedienv)
                helps = script.help(line=line + 1, column=column)
            elif jedi.__version__ >= "0.16.0":
                script = jedi.Script(
                    source=code, path=path, encoding="utf-8", environment=jedienv
                )
                helps = script.help(line=line + 1, column=column)
            else:
                if jedi.__version__ >= "0.12.0":
                    script = jedi.Script(
                        code,
                        line + 1,
                        column,
                        path,
                        encoding="utf-8",
                        environment=jedienv,
                    )
                else:
                    script = jedi.Script(code, line + 1, column, path, encoding="utf-8")
                helps = script.help()

        results = []
        # disable error stream to avoid import errors of jedi,
        # which are directly printed to sys.stderr (no exception)
        with StreamHider(("stderr",)) as h:
            for h in helps:
                if h.type == "keyword":
                    continue
                try:
                    desc = h.description

                    itemType = h.type.lower()

                    if itemType == "module":
                        tooltips = name_tooltip_type_module(h)
                    elif itemType == "statement":
                        tooltips = name_tooltip_type_statement(h)
                    elif itemType == "instance":
                        tooltips = name_tooltip_type_instance(h)
                    elif itemType == "param":
                        tooltips = name_tooltip_type_param(h)
                    elif itemType == "property":
                        tooltips = name_tooltip_type_property(h)
                    elif itemType == "keyword":
                        continue
                    else:
                        # class, function, path
                        tooltips = name_tooltip_type_general(h)

                    results.append((desc, tooltips))
                except Exception as ex:
                    break  # todo, check this further

        return results


if __name__ == "__main__":

    text = "bla = 4\ndata = 2\ndata = data + 3\nprint(data)"
    print(goto_assignments(text, 3, 8, "", 0))
    print(goto_assignments(text, 3, 8, "", 1))
    print(goto_assignments(text, 3, 8, "", 2))

    text = """def test():
    data = b''
    start = [1,2]
    while(True):
        if len(start) > 0:
            print(data)#data += b'hallo'
            start = []
        else:
            break
    return data"""

    print(goto_assignments(text, 9, 13, "", 0))
    print(goto_assignments(text, 9, 13, "", 1))

    source = """def my_func():
    print ('called')

alias = my_func
my_list = [1, None, alias]
inception = my_list[2]

inception()"""
    print(goto_assignments(source, 7, 1, "", 0))
    print(goto_assignments(source, 7, 1, "", 1))

    print(calltips("from itom import dataObject\ndataObject.copy(", 1, 16))
    print(calltips("from itom import dataObject\na = dataObject()\na.copy(", 2, 7))
    print(completions("import win", 0, 10, "", ""))
    print(calltips("from itom import dataObject\ndataObject.zeros(", 1, 17))
    result = completions("Pdm[:,i] = m[02,i]*P[:,i]", 0, 15, "", "")
    print(calltips("from itom import dataObject\ndataObject([4,5], 'u", 1, 17))
    print(calltips("def test(a, b=2):\n    pass\ntest(", 2, 5))

    print(completions("from itom import dataObject\ndataO, 'u", 1, 5, "", ""))
    print(completions("1", 0, 1, "", ""))
    print(completions("import numpy as np\nnp.arr", 1, 6, "", ""))
    print(goto_assignments("import numpy as np\nnp.ones([1,2])", 1, 5, ""))
    print(goto_assignments("def test(a,b):\n    pass\n\ntest(2,3)", 3, 2, ""))
    result = completions("import itom\nitom.loadID", 1, 9, "", "")
    result = completions("import itom\nitom.region().bounding", 1, 21, "", "")
    result = completions("import itom\nitom.region.ELLIPSE", 1, 16, "", "")
    result = completions("import itom\nitom.pointCloud(", 1, 12, "", "")

    text = "import itom\nitom."
    completions(text, 1, 5, "", "")
