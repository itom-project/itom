# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.

from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.writers.html import HTMLTranslator, SmartyPantsHTMLTranslator
from pygments.formatters import HtmlFormatter

from .. import __version__


class QuarkHtmlFormatter(HtmlFormatter):
    def _wrap_pre(self, inner):
        # TODO: Yeah, this is pretty awkward.
        for i, s in HtmlFormatter._wrap_pre(self, inner):
            yield i, s.replace('<span></span>', '')


class HTMLTranslatorMixin(object):
    __super__ = None
    __mixin_prefix__ = "Quark"

    def __init__(self, *args, **kwargs):
        self.highlighter.formatter = QuarkHtmlFormatter

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


def create_html_translator_class(base_class, mixin=HTMLTranslatorMixin,
                                 name=None):
    class _HTMLTranslatorClass(mixin, base_class):
        __super__ = base_class

        def __init__(self, *args, **kwargs):
            base_class.__init__(self, *args, **kwargs)
            mixin.__init__(self, *args, **kwargs)

    _HTMLTranslatorClass.__name__ = name if name else (mixin.__mixin_prefix__ +
                                                       base_class.__name__)
    return _HTMLTranslatorClass


QuarkHTMLTranslator = create_html_translator_class(HTMLTranslator)
QuarkSmartyPantsHTMLTranslator = create_html_translator_class(
    SmartyPantsHTMLTranslator)


_KNOWN_TRANSLATORS = {
    HTMLTranslator: QuarkHTMLTranslator,
    SmartyPantsHTMLTranslator: QuarkSmartyPantsHTMLTranslator,
}


def setup_translators(app):
    # So this is a bit hacky; the "proper" way would be to use the new
    # app.set_translator function added in Sphinx 1.3, but that's much more
    # awkward because
    #  a) it's new in 1.3
    #  b) there's no extension hook after the configuration is loaded, but
    #     before the builder is created; meaning we would have to replace the
    #     translator class much more broadly and sort out any settings later.
    if (
            isinstance(app.builder, StandaloneHTMLBuilder) and
            app.config.quark_use_html_translator):
        try:
            app.builder.translator_class = \
                _KNOWN_TRANSLATORS[app.builder.translator_class]
        except KeyError:
            pass


def setup(app):
    app.add_config_value('quark_use_html_translator', True, 'html')
    app.connect('builder-inited', setup_translators)
    return {
        'version': __version__,
        'parallel_read_safe': True,
    }
