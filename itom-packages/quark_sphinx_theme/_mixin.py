# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.


"""
Support module to build a class from a base class and one or several mixins.

A ``mixin`` is simply a class defining any number of methods. When mixed into a
base class, methods in the mixin override any methods of the same name in the
base class. When specifying multiple mixins, they are assumed to be in
descending order of precedence, i.e. earlier mixins override later ones.

The base class is available in the ``__super__`` attribute on the final combined
class. Overridden methods can use it to delegate to the base class method.
Delegating to another mixin is not supported.

Mixin classes may optionally define a class ``__name_prefix`` attribute. When
combining classes, this value will be incorporated into the name.
"""


def _name_tag(mixin):
    if hasattr(mixin, '__name_tag__'):
        return mixin.__name_tag__
    else:
        return mixin.__name__


class _NewStyleBase(object):
    """Helper class to make sure all compound classes include ``object``."""

    __name_tag__ = ''

    def __init__(self, *args, **kwargs):
        object.__init__(self)


def create_compound_class(base_class, mixins, name=None):
    """Create a combined class from a base class and several mixins.

    The elements of the ``mixin`` sequence are mixed into the base class; the
    first element in the sequence has the highest priority. If ``name`` is
    supplied, it sets the name of the final class, otherwise a name is derived
    from the base class and mixins.
    """
    all_bases = list(mixins)
    all_bases.append(base_class)
    if not issubclass(base_class, object):
        all_bases.append(_NewStyleBase)
    clsname = name if name else '_'.join(_name_tag(base) for base in all_bases
                                         if _name_tag(base))

    def __init__(self, *args, **kwargs):
        for base in reversed(all_bases):
            if hasattr(base, '__init__'):
                base.__init__(self, *args, **kwargs)

    cls_dict = {
        '__init__': __init__,
        '__super__': base_class,
    }

    return type(clsname, tuple(all_bases), cls_dict)
