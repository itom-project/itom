# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.

from pygments.formatters import HtmlFormatter


class CompatHtmlFormatter(HtmlFormatter):
    def _wrap_pre(self, inner):
        # TODO: Yeah, this is pretty awkward.
        for i, s in HtmlFormatter._wrap_pre(self, inner):
            yield i, s.replace('<span></span>', '')


class HTMLCompatMixin(object):
    __name_tag__ = "Compat"

    def __init__(self, *args, **kwargs):
        self.highlighter.formatter = CompatHtmlFormatter

    def visit_citation(self, node):
        no_id_node = node.copy()
        no_id_node.delattr('ids')
        self.body.append('<div id="%s" class="-x-quark-citation-wrapper">'
                         % node.get('ids')[0])
        self.__super__.visit_citation(self, no_id_node)

    def depart_citation(self, node):
        self.__super__.depart_citation(self, node)
        self.body.append('</div>')

    def visit_footnote(self, node):
        no_id_node = node.copy()
        no_id_node.delattr('ids')
        self.body.append('<div id="%s" class="-x-quark-footnote-wrapper">'
                         % node.get('ids')[0])
        self.__super__.visit_footnote(self, no_id_node)

    def depart_footnote(self, node):
        self.__super__.depart_footnote(self, node)
        self.body.append('</div>')
