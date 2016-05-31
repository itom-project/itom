# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.

from docutils import nodes


class LiteralBlocksMixin(object):
    __name_tag__ = 'Literal'

    def __init__(self, *args, **kwargs):
        pass

    def __append_epilog(self):
        self.body.append('</td></tr></tbody></table>')

    def visit_literal_block(self, node):
        self.body.append('<table class="-x-quark-literal-block">'
                         '<tbody><tr>'
                         '<td width="100%" class="-x-quark-literal-block-td">')
        try:
            self.__super__.visit_literal_block(self, node)
        except nodes.SkipNode:
            self.__append_epilog()
            raise

    def depart_literal_block(self, node):
        self.__super__.depart_literal_block(self, node)
        self.__append_epilog()
