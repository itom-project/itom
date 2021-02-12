# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.

from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.writers.html import HTMLTranslator, SmartyPantsHTMLTranslator

from ... import __version__
from ..._mixin import create_compound_class
from .compat import HTMLCompatMixin
from .boxes import BoxesMixin
from .literal_blocks import LiteralBlocksMixin


KNOWN_TRANSLATORS = [
    HTMLTranslator,
    SmartyPantsHTMLTranslator,
]
# List so it has a stable order. Descending order of precedence.
FEATURE_CLASSES = [
    ("boxes", BoxesMixin),
    ("literal_blocks", LiteralBlocksMixin),
    ("compat", HTMLCompatMixin),
]
ALL_FEATURES = [f[0] for f in FEATURE_CLASSES]
# This may change.
DEFAULT_FEATURES = ALL_FEATURES


def setup_translators(app):
    # So this is a bit hacky; the "proper" way would be to use the new
    # app.set_translator function added in Sphinx 1.3, but that's much more
    # awkward because
    #  a) it's new in 1.3
    #  b) there's no extension hook after the configuration is loaded, but
    #     before the builder is created; meaning we would have to replace the
    #     translator class much more broadly and sort out any settings later.
    cls = app.builder.translator_class
    is_html = isinstance(app.builder, StandaloneHTMLBuilder)
    if is_html and cls in KNOWN_TRANSLATORS:
        mixins = []
        for feature, mixin in FEATURE_CLASSES:
            if feature in app.config.quark_html_rewrite_features:
                mixins.append(mixin)
        app.builder.translator_class = create_compound_class(cls, mixins)


def setup(app):
    app.add_config_value("quark_html_rewrite_features", DEFAULT_FEATURES, "html")
    app.connect("builder-inited", setup_translators)
    return {
        "version": __version__,
        "parallel_read_safe": True,
    }
