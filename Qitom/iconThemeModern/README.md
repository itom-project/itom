# Modern icon themes

The modern icon themes contain a refined version of the classic icons of itom. The motifs and colors are
the same, but the icons look more modern. They match the application styles `modernLight` and
`modernDark` (`styles/stylesheets`).

What is changed (see `generate_modern_icons.py`):

* The gloss and the shading of colored areas are flattened to calm color areas. Neutral areas (paper,
  metal, outlines) keep their detail.
* The colors are slightly calmer. Light outlines of paper icons are slightly crisper.
* In the dark variant, black glyphs and outlines are lifted, such that they stay visible on dark
  backgrounds (e.g. floppy disk, clipboard, printer).
* Logos, splash screens, animations and svg images are not changed.

Select them in the properties dialog, section general > application style > icon theme:
`modernAuto`, `modernBright` or `modernDark`. With the default value `auto`, the refined icons are
used automatically, if the selected style is one of the modern styles.

## How the icon themes work (read this before changing a qrc file)

`qitom.qrc` is compiled into the application. The icon themes are binary resource files
(`iconThemeBright.rcc`, `iconThemeDark.rcc`, `iconThemeModernBright.rcc`, `iconThemeModernDark.rcc`),
and exactly one of them is registered at startup (`MainApplication`, after the compiled-in
resources).

**Qt uses the first registered resource, if a resource path exists more than once.** The compiled-in
resources are registered first, such that a theme file can never replace an icon of `qitom.qrc`.
Therefore:

* An icon, whose appearance depends on the theme, must **not** be listed in `qitom.qrc`.
  It has to be listed in all four theme files (`iconThemeBright.qrc`, `iconThemeDark.qrc` and the two
  generated `iconThemeModern*.qrc`), always with the same resource path (prefix and alias).
* Icons that do not depend on the theme (logos, splash screens, animations, ...) stay in `qitom.qrc`.

## Files

| File | Purpose |
|---|---|
| `generate_modern_icons.py` | creates the refined images (`bright/`, `dark/`) from the classic icons and `../iconThemeModernBright.qrc`, `../iconThemeModernDark.qrc` |

The images in `bright/` and `dark/` and the two `iconThemeModern*.qrc` files are generated. Do not edit them,
but change the classic icons or the parameters in the generator and run `python generate_modern_icons.py`
(requires Python with numpy, opencv-python and pillow).

## Add a new themed icon

1. Add the icon to `iconThemeBright.qrc` and `iconThemeDark.qrc` (not to `qitom.qrc`).
2. Run the generator.
