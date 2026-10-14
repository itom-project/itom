"""Generator for the itom application styles 'modernLight' and 'modernDark'.

Both styles are created from the same template (modernStyle.qss.in) and differ only
in their color palette (see PALETTES below). This guarantees that the light and dark
variant always have the same structure.

The script renders the indicator images (arrows, check boxes, radio buttons, tree
branches, dock buttons) from inline SVG definitions into PNG files (normal and @2x
for high-dpi screens) and compiles them into a binary Qt resource file (.rcc).

Requirements: Python 3 with PySide6 (or PyQt6 with the same API) and the Qt
resource compiler 'rcc'.

Usage:
    python generate_modern_styles.py [--rcc PATH_TO_RCC]

Output (in the parent folder 'styles/stylesheets'):
    modernLight/modernLight.qss, modernLight/modernLight.rcc
    modernDark/modernDark.qss, modernDark/modernDark.rcc
and the matching code editor themes (in 'styles/editorThemes'):
    ModernLight.ini, ModernDark.ini

Notes:
    - The .rcc files are written in format version 2 with zlib compression,
      such that both Qt5 and Qt6 builds of itom can load them.
    - The editor themes do not contain font weights, since the numeric weight scale
      differs between Qt5 (0-99) and Qt6 (1-1000). The current weights are kept on import.
    - Only the .qss and .rcc files are deployed by CMake. Do not edit them
      directly, but change the template or the palette and re-run this script.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_ROOT = os.path.dirname(HERE)  # styles/stylesheets
RESOURCE_PREFIX = "modern_icons"

PALETTES = {
    "modernLight": {
        "bg": "#f6f7f9",            # window background, between the panels
        "panel": "#ffffff",         # docks, editors, lists
        "panelAlt": "#fafbfc",      # alternating rows
        "line": "#e3e5ea",          # borders and separators
        "lineStrong": "#c9cdd4",    # borders of inputs, hovered borders
        "text": "#1f2328",
        "textSecondary": "#5b626c",
        "textDisabled": "#a0a6ae",
        "accent": "#2f6fde",
        "accentHover": "#2862c7",
        "accentText": "#ffffff",    # text on accent
        "accentBg": "#e8f0fd",      # selection background
        "accentBgText": "#174ea6",  # text on selection background
        "hover": "#eef1f5",
        "pressed": "#e3e7ed",
        "scrollHandle": "#c9cdd4",
        "scrollHandleHover": "#a9afb8",
        "tooltipBg": "#1f2328",
        "tooltipText": "#ffffff",
        "progressChunk": "#9ec0f3",  # lighter than accent, such that text stays readable
    },
    "modernDark": {
        "bg": "#1b1d21",
        "panel": "#23262b",
        "panelAlt": "#272a30",
        "line": "#33373e",
        "lineStrong": "#474c55",
        "text": "#e6e8eb",
        "textSecondary": "#9aa1ab",
        "textDisabled": "#5d636c",
        "accent": "#5b93f0",
        "accentHover": "#76a6f3",
        "accentText": "#0f1218",
        "accentBg": "#22324d",
        "accentBgText": "#d6e4fc",
        "hover": "#2c3036",
        "pressed": "#33373e",
        "scrollHandle": "#474c55",
        "scrollHandleHover": "#5d636c",
        "tooltipBg": "#e6e8eb",
        "tooltipText": "#1b1d21",
        "progressChunk": "#3a5f9e",
    },
}

# ------------------------------------------------------------------------------------------
# indicator images: name -> (svg template, size in px at 1x)
# placeholders: {fg} = main color, {bg} = fill color, {line} = border color
# ------------------------------------------------------------------------------------------
_CHEVRON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<path d="{d}" fill="none" stroke="{fg}" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_CHEVRONS = {
    "down": "M4.5 6.5 L8 10 L11.5 6.5",
    "up": "M4.5 9.5 L8 6 L11.5 9.5",
    "left": "M9.5 4.5 L6 8 L9.5 11.5",
    "right": "M6.5 4.5 L10 8 L6.5 11.5",
}
_CHECKBOX_BOX = (
    '<rect x="1.5" y="1.5" width="13" height="13" rx="3" fill="{bg}" stroke="{line}" stroke-width="1"/>'
)
_CHECKBOX_ON = (
    '<rect x="1" y="1" width="14" height="14" rx="3" fill="{fg}"/>'
)
_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">{body}</svg>'

IMAGES = {}
for _name, _d in _CHEVRONS.items():
    IMAGES["arrow_%s" % _name] = _CHEVRON.replace("{d}", _d)
IMAGES["branch_closed"] = IMAGES["arrow_right"]
IMAGES["branch_open"] = IMAGES["arrow_down"]
IMAGES["checkbox_unchecked"] = _SVG.format(body=_CHECKBOX_BOX)
IMAGES["checkbox_checked"] = _SVG.format(
    body=_CHECKBOX_ON
    + '<path d="M4.5 8.2 L7 10.5 L11.5 5.5" fill="none" stroke="{bg}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
)
IMAGES["checkbox_indeterminate"] = _SVG.format(
    body=_CHECKBOX_ON
    + '<path d="M4.5 8 L11.5 8" stroke="{bg}" stroke-width="1.8" stroke-linecap="round"/>'
)
IMAGES["radio_unchecked"] = _SVG.format(
    body='<circle cx="8" cy="8" r="6.5" fill="{bg}" stroke="{line}" stroke-width="1"/>'
)
IMAGES["radio_checked"] = _SVG.format(
    body='<circle cx="8" cy="8" r="7" fill="{fg}"/><circle cx="8" cy="8" r="2.8" fill="{bg}"/>'
)
IMAGES["close"] = _SVG.format(
    body='<path d="M3 3 L13 13 M13 3 L3 13" stroke="{fg}" stroke-width="1.8" stroke-linecap="round"/>'
)
IMAGES["undock"] = _SVG.format(
    body='<rect x="2" y="5" width="9" height="9" rx="1.5" fill="none" stroke="{fg}" stroke-width="1.6"/>'
    '<path d="M5.5 2 L14 2 L14 10.5" fill="none" stroke="{fg}" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
)


def image_variants(palette):
    """returns a dict: file base name -> svg source with resolved colors."""
    result = {}
    states = {
        "": dict(fg=palette["textSecondary"], bg=palette["panel"], line=palette["lineStrong"]),
        "_hover": dict(fg=palette["text"], bg=palette["panel"], line=palette["accent"]),
        "_disabled": dict(fg=palette["textDisabled"], bg=palette["bg"], line=palette["line"]),
    }
    for name, svg in IMAGES.items():
        for suffix, colors in states.items():
            c = dict(colors)
            if name.startswith(("checkbox_checked", "checkbox_indeterminate", "radio_checked")):
                # checked indicators use the accent color as fill
                c["fg"] = palette["accent"] if suffix != "_disabled" else palette["textDisabled"]
                c["bg"] = palette["accentText"] if suffix != "_disabled" else palette["bg"]
                if suffix == "_hover":
                    c["fg"] = palette["accentHover"]
            src = svg
            for key, value in c.items():
                src = src.replace("{%s}" % key, value)
            result[name + suffix] = src
    return result


def render_png(svg_source, size, filename):
    renderer = QSvgRenderer(QByteArray(svg_source.encode("utf-8")))
    if not renderer.isValid():
        raise RuntimeError("invalid svg for %s" % filename)
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    if not image.save(filename, "PNG"):
        raise RuntimeError("could not save %s" % filename)


def find_rcc(explicit):
    candidates = [explicit] if explicit else []
    candidates += ["rcc", "rcc-qt6", "rcc-qt5"]
    try:
        import PySide6

        candidates.append(os.path.join(os.path.dirname(PySide6.__file__), "Qt", "libexec", "rcc"))
        candidates.append(os.path.join(os.path.dirname(PySide6.__file__), "rcc.exe"))
    except ImportError:
        pass
    for c in candidates:
        if c and (os.path.isfile(c) or shutil.which(c)):
            return shutil.which(c) or c
    raise RuntimeError("Qt resource compiler 'rcc' not found. Use --rcc to specify it.")


def generate(style_name, palette, template, rcc):
    target_dir = os.path.join(TARGET_ROOT, style_name)
    os.makedirs(target_dir, exist_ok=True)

    # drop the leading comment of the template, the generated file gets its own header
    qss = template.split("*/", 1)[1].lstrip() if template.lstrip().startswith("/*") else template
    for key, value in palette.items():
        qss = qss.replace("{{%s}}" % key, value)
    qss = qss.replace("{{icons}}", ":/%s/rc" % RESOURCE_PREFIX)
    if "{{" in qss:
        raise RuntimeError("unresolved placeholder in template: %s" % qss[qss.index("{{"):][:40])

    header = (
        "/* itom application style '%s'.\n"
        "   GENERATED FILE - do not edit. Change modernStyleSources/modernStyle.qss.in\n"
        "   or the palette in modernStyleSources/generate_modern_styles.py and re-run the script. */\n\n"
        % style_name
    )
    with open(os.path.join(target_dir, style_name + ".qss"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header + qss)

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "rc"))
        files = []
        for name, svg in sorted(image_variants(palette).items()):
            for scale, suffix in ((1, ""), (2, "@2x")):
                rel = f"rc/{name}{suffix}.png"
                render_png(svg, 16 * scale, os.path.join(tmp, rel))
                files.append(rel)
        qrc = os.path.join(tmp, style_name + ".qrc")
        with open(qrc, "w", encoding="utf-8") as f:
            f.write('<RCC>\n  <qresource prefix="%s">\n' % RESOURCE_PREFIX)
            for rel in files:
                f.write("    <file>%s</file>\n" % rel)
            f.write("  </qresource>\n</RCC>\n")
        subprocess.check_call(
            [rcc, "--binary", "--format-version", "2", "--compress-algo", "zlib",
             "-o", os.path.join(target_dir, style_name + ".rcc"), qrc],
            cwd=tmp,
        )
    print("generated %s (%d images)" % (style_name, len(files)))

# ------------------------------------------------------------------------------------------
# code editor themes (styles/editorThemes), matching the application styles
# ------------------------------------------------------------------------------------------
EDITOR_THEMES_DIR = os.path.join(os.path.dirname(TARGET_ROOT), "editorThemes")

# ids of ito::StyleItem::StyleType (Qitom/codeEditor/syntaxHighlighter/codeEditorStyle.h)
EDITOR_STYLE_IDS = {
    "default": 0, "comment": 1, "number": 2, "string": 3, "keyword": 5, "docstring": 6,
    "class": 8, "function": 9, "operator": 10, "decorator": 15, "namespace": 1002,
    "type": 1003, "keywordReserved": 1004, "builtin": 1005, "definition": 1006,
    "instance": 1007, "tag": 1009, "self": 1010, "punctuation": 1011, "constant": 1012,
    "operatorWord": 1013, "streamOutput": 1014, "streamError": 1015,
}
ITALIC_STYLES = ("comment", "docstring")

EDITOR_THEMES = {
    "ModernLight": {
        "syntax": {
            "default": "#1f2328", "comment": "#6e7781", "number": "#b25a00", "string": "#1d7a4f",
            "keyword": "#a2349a", "docstring": "#1d7a4f", "class": "#9a6700", "function": "#2f6fde",
            "operator": "#1f2328", "decorator": "#8250df", "namespace": "#1f2328", "type": "#9a6700",
            "keywordReserved": "#a2349a", "builtin": "#0e7490", "definition": "#2f6fde",
            "instance": "#1f2328", "tag": "#a2349a", "self": "#8250df", "punctuation": "#5b626c",
            "constant": "#b25a00", "operatorWord": "#a2349a", "streamOutput": "#1f2328",
            "streamError": "#cf222e",
        },
        "editor": {
            "paperBackgroundColor": "#ffffff",
            "caretBackgroundColor": "#f3f5f8",
            "caretForegroundColor": "#1f2328",
            "markerScriptErrorBackgroundColor": "#ffebe9",
            "markerCurrentBackgroundColor": "#fff3c4",
            "markerInputForegroundColor": "#e8f0fd",
            "markerErrorForegroundColor": "#ffd8d3",
            "whitespaceForegroundColor": "#c9cdd4",
            "whitespaceBackgroundColor": "#00000000",
            "unmatchedBraceBackgroundColor": "#00000000",
            "unmatchedBraceForegroundColor": "#cf222e",
            "matchedBraceBackgroundColor": "#dbe7fb",
            "matchedBraceForegroundColor": "#1f2328",
            "selectionBackgroundColor": "#cfe0fb",
            "selectionForegroundColor": "#1f2328",
            "markerSameStringBackgroundColor": "#ece6fb",
        },
    },
    "ModernDark": {
        "syntax": {
            "default": "#e6e8eb", "comment": "#7d848e", "number": "#e6a35c", "string": "#7cc89a",
            "keyword": "#d78ee6", "docstring": "#7cc89a", "class": "#e5c07b", "function": "#7fb0ff",
            "operator": "#e6e8eb", "decorator": "#c4a7f7", "namespace": "#e6e8eb", "type": "#e5c07b",
            "keywordReserved": "#d78ee6", "builtin": "#5fc4d6", "definition": "#7fb0ff",
            "instance": "#e6e8eb", "tag": "#d78ee6", "self": "#c4a7f7", "punctuation": "#9aa1ab",
            "constant": "#e6a35c", "operatorWord": "#d78ee6", "streamOutput": "#e6e8eb",
            "streamError": "#ff7b72",
        },
        "editor": {
            "paperBackgroundColor": "#23262b",
            "caretBackgroundColor": "#2a2d33",
            "caretForegroundColor": "#e6e8eb",
            "markerScriptErrorBackgroundColor": "#4a2326",
            "markerCurrentBackgroundColor": "#4a4126",
            "markerInputForegroundColor": "#22324d",
            "markerErrorForegroundColor": "#5a2a2a",
            "whitespaceForegroundColor": "#474c55",
            "whitespaceBackgroundColor": "#00000000",
            "unmatchedBraceBackgroundColor": "#00000000",
            "unmatchedBraceForegroundColor": "#ff7b72",
            "matchedBraceBackgroundColor": "#2f4468",
            "matchedBraceForegroundColor": "#e6e8eb",
            "selectionBackgroundColor": "#2f4468",
            "selectionForegroundColor": "#e6e8eb",
            "markerSameStringBackgroundColor": "#3b3256",
        },
    },
}


def generate_editor_theme(theme_name, theme, font_family="Consolas", point_size=10):
    lines = []
    for key, style_id in EDITOR_STYLE_IDS.items():
        lines += [
            "[PythonLexerStyle%d]" % style_id,
            "backgroundColor=#000000",
            "backgroundColorAlpha=0",
            "foregroundColor=%s" % theme["syntax"][key],
            "foregroundColorAlpha=255",
            "fontFamily=%s" % font_family,
            "pointSize=%d" % point_size,
            "italic=%s" % ("true" if key in ITALIC_STYLES else "false"),
            "",
        ]
    lines.append("[CodeEditor]")
    lines += [f"{k}={v}" for k, v in theme["editor"].items()]
    lines.append("caretBackgroundShow=true")
    path = os.path.join(EDITOR_THEMES_DIR, theme_name + ".ini")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("generated editor theme %s" % theme_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rcc", help="path to the Qt resource compiler (rcc)")
    args = parser.parse_args()

    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # required for QImage/QPainter
    rcc = find_rcc(args.rcc)

    with open(os.path.join(HERE, "modernStyle.qss.in"), encoding="utf-8") as f:
        template = f.read()

    for style_name, palette in PALETTES.items():
        generate(style_name, palette, template, rcc)

    for theme_name, theme in EDITOR_THEMES.items():
        generate_editor_theme(theme_name, theme)


if __name__ == "__main__":
    main()
