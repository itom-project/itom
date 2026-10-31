#
# itom documentation build configuration for the HTML builder.
#
# This file loads the common configuration from ../conf.py (used for the
# QtHelp build) and only overrides the settings that are specific to the
# HTML output (theme, logo, extensions). Keep all shared settings such as
# intersphinx inventories, extensions and autodoc options in ../conf.py so
# that both configurations cannot drift apart.

import os

_thisDir = os.path.dirname(os.path.abspath(__file__))
_baseConf = os.path.normpath(os.path.join(_thisDir, "..", "conf.py"))

# execute the base configuration in this namespace. __file__ is temporarily
# set to the base file, so that all paths in ../conf.py are resolved relative
# to the "source" directory.
_g = globals()
_oldFile = _g.get("__file__")
_g["__file__"] = _baseConf
with open(_baseConf, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _baseConf, "exec"), _g)
_g["__file__"] = _oldFile

# -- HTML specific overrides ---------------------------------------------------

extensions += ["sphinx_favicon", "sphinx_copybutton"]

toc_object_entries = True

logo = "../../../../Qitom/icons/itomicon/itomLogo3_128.png"

html_theme = "pydata_sphinx_theme"
html_theme_path = []
html_theme_options = {
    "show_nav_level": 4,
    "navigation_depth": 4,
    "show_toc_level": 5,
    "collapse_navigation": True,
    "pygment_light_style": "vs",
    "pygment_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/itom-project",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "ITO",
            "url": "https://www.ito.uni-stuttgart.de/",
            "icon": "_static/ITO_Logo_neu_II.png",
            "type": "local",
        },
    ],
}

html_show_sourcelink = True
html_show_sphinx = True
html_title = "itom Documentation"
html_short_title = "itom"
html_logo = logo
html_favicon = logo
html_css_files = ["html.css"]
html_static_path = ["../_static"]
