# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.


class BoxesMixin(object):
    __name_tag__ = 'Boxes'

    def __init__(self, *args, **kwargs):
        pass

    def visit_admonition(self, node, name=''):
        self.body.append('<table class="-x-quark-box -x-quark-admonition %s">'
                         '<tbody><tr>'
                         '<td width="100%%" class="-x-quark-box-td">'
                         % ('-x-quark-%s' % name if name else ''))
        self.__super__.visit_admonition(self, node, name)

    def depart_admonition(self, node=None):
        self.__super__.depart_admonition(self, node)
        self.body.append("</td></tr></tbody></table>")

    def visit_topic(self, node):
        self.body.append('<table class="-x-quark-box -x-quark-topic">'
                         '<tbody><tr>'
                         '<td width="100%" class="-x-quark-box-td">')
        self.__super__.visit_topic(self, node)

    def depart_topic(self, node):
        self.__super__.depart_topic(self, node)
        self.body.append('</td></tr></tbody></table>')

    def visit_sidebar(self, node):
        self.body.append('<table class="-x-quark-box -x-quark-sidebar">'
                         '<tbody><tr>'
                         '<td width="35%" class="-x-quark-box-td">')
        self.__super__.visit_sidebar(self, node)

    def depart_sidebar(self, node):
        self.__super__.depart_sidebar(self, node)
        self.body.append('</td></tr></tbody></table>')
