# -*- coding: utf-8 -*-
"""
    pyitom domain
    ~~~~~~~~~~~~~~~~~~~~~
"""

import re

from docutils import nodes
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.roles import XRefRole
from sphinx.locale import l_, _
from sphinx.domains import Domain, ObjType, Index
from sphinx.directives import ObjectDescription
from sphinx.util.nodes import make_refnode
from sphinx.util.compat import Directive
from sphinx.util.docfields import Field, GroupedField, TypedField


# REs for Python signatures
py_sig_re = re.compile(
    r'''^ ([\w.]*\.)?            # class name(s)
          (\w+)  \s*             # thing name
          (?: \((.*)\)           # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)


def _pseudo_parse_arglist(signode, filtername, arglist):
    """"Parse" a list of arguments separated by commas.

    Arguments can have "optional" annotations given by enclosing them in
    brackets.  Currently, this will split at any comma, even if it's inside a
    string literal (e.g. default argument value).
    """
    paramlist = addnodes.desc_parameterlist()
    strname = "\"%s\"" % filtername
    paramlist += addnodes.desc_name(strname, strname)
    
    if (not arglist is None and len(arglist) > 0):
        paramlist += addnodes.desc_annotation(",", ",")
    
    stack = [paramlist]
    try:
        for argument in arglist.split(','):
            argument = argument.strip()
            ends_open = ends_close = 0
            while argument.startswith('['):
                stack.append(addnodes.desc_optional())
                stack[-2] += stack[-1]
                argument = argument[1:].strip()
            while argument.startswith(']'):
                stack.pop()
                argument = argument[1:].strip()
            while argument.endswith(']'):
                ends_close += 1
                argument = argument[:-1].strip()
            while argument.endswith('['):
                ends_open += 1
                argument = argument[:-1].strip()
            if argument:
                stack[-1] += addnodes.desc_parameter(argument, argument)
            while ends_open:
                stack.append(addnodes.desc_optional())
                stack[-2] += stack[-1]
                ends_open -= 1
            while ends_close:
                stack.pop()
                ends_close -= 1
        if len(stack) != 1:
            raise IndexError
    except IndexError:
        # if there are too few or too many elements on the stack, just give up
        # and treat the whole argument list as one argument, discarding the
        # already partially populated paramlist node
        signode += addnodes.desc_parameterlist()
        signode[-1] += addnodes.desc_parameter(arglist, arglist)
    else:
        signode += paramlist


class PyItomObject(ObjectDescription):
    """
    Description of a general itom Python object.
    """
    option_spec = {
        'annotation': directives.unchanged,
    }

    #doc_field_types = [
        #TypedField('parameter', label=l_('Parameters'),
                   #names=('param', 'parameter', 'arg', 'argument',
                          #'keyword', 'kwarg', 'kwparam'),
                   #typerolename='obj', typenames=('paramtype', 'type'),
                   #can_collapse=True),
        #TypedField('variable', label=l_('Variables'), rolename='obj',
                   #names=('var', 'ivar', 'cvar'),
                   #typerolename='obj', typenames=('vartype',),
                   #can_collapse=True),
        #GroupedField('exceptions', label=l_('Raises'), rolename='exc',
                     #names=('raises', 'raise', 'exception', 'except'),
                     #can_collapse=True),
        #Field('returnvalue', label=l_('Returns'), has_arg=False,
              #names=('returns', 'return')),
        #Field('returntype', label=l_('Return type'), has_arg=False,
              #names=('rtype',)),
    #]
    
    doc_field_types = [
        TypedField('paramsmand', label=l_('Mandatory parameters'),
                   names=('parammand', 'param'),
                   typerolename='obj', typenames=('parammandtype', 'type'),
                   can_collapse=True),
        TypedField('paramsopt', label=l_('Optional parameters'),
                   names=('paramopt',),
                   typerolename='obj', typenames=('paramopttype',),
                   can_collapse=True),
        TypedField('returnvalue', label=l_('Returns'),
                   names=('returns', 'return'),
                   typerolename='obj', typenames=('returntype','rtype'),
                   can_collapse=True),
        GroupedField('exceptions', label=l_('Raises'), rolename='exc',
                     names=('raises', 'raise', 'exception', 'except'),
                     can_collapse=True),
    ]

    def get_signature_prefix(self, sig):
        """May return a prefix to put before the object name in the
        signature.
        """
        return ''

    def needs_arglist(self):
        """May return true if an empty argument list is to be generated even if
        the document contains none.
        """
        return False

    def handle_signature(self, sig, signode):
        """Transform a Python signature into RST nodes.

        Return (fully qualified name of the thing, classname if any).

        If inside a class, the current class name is handled intelligently:
        * it is stripped from the displayed name if present
        * it is added to the full name (return value) if not present
        """
        m = py_sig_re.match(sig)
        if m is None:
            raise ValueError
        name_prefix, name, arglist, retann = m.groups()

        # determine module and class name (if applicable), as well as full name
        add_module = True
        if name_prefix:
            classname = name_prefix.rstrip('.')
            fullname = name_prefix + name
        else:
            classname = ''
            fullname = name

        signode['module'] = ""
        signode['class'] = classname
        signode['fullname'] = fullname

        sig_prefix = "itom.filter" #self.get_signature_prefix(sig)
        signode += addnodes.desc_annotation(sig_prefix, sig_prefix)

        #name_prefix = "("
        #signode += addnodes.desc_addname(name_prefix, name_prefix)

        anno = self.options.get('annotation')
        
        name2 = "\"%s\"" % name
        #signode += addnodes.desc_name(name2, name2)
        #if not arglist:
        #    if self.needs_arglist():
        #       # for callables, add an empty parameter list
        #        signode += addnodes.desc_parameterlist()
        #    if retann:
        #        signode += addnodes.desc_returns(retann, retann)
        #    if anno:
        #        signode += addnodes.desc_annotation(' ' + anno, ' ' + anno)
        #    return fullname, name_prefix

        _pseudo_parse_arglist(signode, name, arglist)
        if retann:
            signode += addnodes.desc_returns(retann, retann)
        if anno:
            signode += addnodes.desc_annotation(' ' + anno, ' ' + anno)
        return fullname, name_prefix

    def get_index_text(self, name):
        """Return the text for the index entry of the object."""
        raise NotImplementedError('must be implemented in subclasses')

    def add_target_and_index(self, name_cls, sig, signode):
        fullname = name_cls[0]
        # note target
        if fullname not in self.state.document.ids:
            signode['names'].append(fullname)
            signode['ids'].append(fullname)
            signode['first'] = (not self.names)
            self.state.document.note_explicit_target(signode)
            objects = self.env.domaindata['pyitom']['objects']
            if fullname in objects:
                self.state_machine.reporter.warning(
                    'duplicate object description of %s, ' % fullname +
                    'other instance in ' +
                    self.env.doc2path(objects[fullname][0]) +
                    ', use :noindex: for one of them',
                    line=self.lineno)
            objects[fullname] = (self.env.docname, self.objtype)

        indextext = self.get_index_text(name_cls)
        if indextext:
            self.indexnode['entries'].append(('single', indextext,
                                              fullname, ''))

    def before_content(self):
        # needed for automatic qualification of members (reset in subclasses)
        pass

    def after_content(self):
        pass


class PyItomModuleLevel(PyItomObject):
    """
    Description of an object on module level (filter).
    """

    def needs_arglist(self):
        return self.objtype == 'filter'

    def get_index_text(self, name_cls):
        if self.objtype == 'filter':
            return _('%s (plugin filter)') % name_cls[0]
        else:
            return ''


class PyItomXRefRole(XRefRole):
    def process_link(self, env, refnode, has_explicit_title, title, target):
        #refnode['py:module'] = env.temp_data.get('py:module')
        #refnode['py:class'] = env.temp_data.get('py:class')
        if not has_explicit_title:
            title = title.lstrip('.')   # only has a meaning for the target
            target = target.lstrip('~') # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot+1:]
        # if the first character is a dot, search more specific namespaces first
        # else search builtins first
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        
        # skip parens
        if title[-2:] == '()':
            title = title[:-2]
            
        return title, target


class PyItomDomain(Domain):
    """Python language domain."""
    name = 'pyitom'
    label = 'PyItom'
    object_types = {
        'filter':     ObjType(l_('filter'),      'filt', 'obj'),
    }

    directives = {
        'filter':        PyItomModuleLevel,
    }
    roles = {
        'filt':  PyItomXRefRole(fix_parens=True),
    }
    initial_data = {
        'objects': {},  # fullname -> docname, objtype
    }
    indices = [
    ]

    def clear_doc(self, docname):
        for fullname, (fn, _) in list(self.data['objects'].items()):
            if fn == docname:
                del self.data['objects'][fullname]

    def find_obj(self, env, name, type, searchmode=0):
        """Find a Python object for "name", perhaps using the given module
        and/or classname.  Returns a list of (name, object entry) tuples.
        """
        # skip parens
        if name[-2:] == '()':
            name = name[:-2]

        if not name:
            return []

        objects = self.data['objects']
        matches = []

        newname = None
        if searchmode == 1:
            objtypes = self.objtypes_for_role(type)
            if objtypes is not None:
                if not newname:
                    if name in objects and objects[name][1] in objtypes:
                        newname = name
                    else:
                        # "fuzzy" searching mode
                        searchname = '.' + name
                        matches = [(oname, objects[oname]) for oname in objects
                                   if oname.endswith(searchname)
                                   and objects[oname][1] in objtypes]
        else:
            # NOTE: searching for exact match, object type is not considered
            if name in objects:
                newname = name
        if newname is not None:
            matches.append((newname, objects[newname]))
        return matches

    def resolve_xref(self, env, fromdocname, builder,
                     type, target, node, contnode):
        searchmode = node.hasattr('refspecific') and 1 or 0
        matches = self.find_obj(env, target, type, searchmode)
        if not matches:
            return None
        elif len(matches) > 1:
            env.warn_node(
                'more than one target found for cross-reference '
                '%r: %s' % (target, ', '.join(match[0] for match in matches)),
                node)
        name, obj = matches[0]
        
        return make_refnode(builder, fromdocname, obj[0], name,
                                contnode, name)

    def get_objects(self):
        for refname, (docname, type) in self.data['objects'].items():
            yield (refname, refname, type, docname, refname, 1)

def setup(app):
    app.add_domain(PyItomDomain)