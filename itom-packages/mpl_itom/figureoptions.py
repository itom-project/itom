# Copyright � 2009 Pierre Raybaut
# Licensed under the terms of the MIT License
# see the mpl licenses directory for a copy of the license


"""Module that provides a GUI-based editor for matplotlib's figure options."""

import os.path
import re

import matplotlib
from matplotlib import cm, colors as mcolors, markers, image as mimage
from mpl_itom import _dialogEditProperties


LINESTYLES = {
    "-": "Solid",
    "--": "Dashed",
    "-.": "DashDot",
    ":": "Dotted",
    "None": "None",
}

DRAWSTYLES = {
    "default": "Default",
    "steps-pre": "Steps (Pre)",
    "steps": "Steps (Pre)",
    "steps-mid": "Steps (Mid)",
    "steps-post": "Steps (Post)",
}

MARKERS = markers.MarkerStyle.markers


def figure_edit(matplotlibplotUiItem, axes, parent=None):
    """Edit matplotlib figure options"""
    sep = (None, None)  # separator

    # Get / General
    # Cast to builtin floats as they have nicer reprs.
    xmin, xmax = map(float, axes.get_xlim())
    ymin, ymax = map(float, axes.get_ylim())
    general = [
        ("Title", axes.get_title()),
        sep,
        (None, "<b>X-Axis</b>"),
        ("Left", xmin),
        ("Right", xmax),
        ("Label", axes.get_xlabel()),
        ("Scale", [axes.get_xscale(), "linear", "log", "logit"]),
        sep,
        (None, "<b>Y-Axis</b>"),
        ("Bottom", ymin),
        ("Top", ymax),
        ("Label", axes.get_ylabel()),
        ("Scale", [axes.get_yscale(), "linear", "log", "logit"]),
        sep,
        ("(Re-)Generate automatic legend", False),
    ]

    # Save the unit data
    xconverter = axes.xaxis.converter
    yconverter = axes.yaxis.converter
    xunits = axes.xaxis.get_units()
    yunits = axes.yaxis.get_units()

    # Sorting for default labels (_lineXXX, _imageXXX).
    def cmp_key(label):
        match = re.match(r"(_line|_image)(\d+)", label)
        if match:
            return match.group(1), int(match.group(2))
        else:
            return label, 0

    # Get / Curves
    linedict = {}
    for line in axes.get_lines():
        label = line.get_label()
        if label == "_nolegend_":
            continue
        linedict[label] = line
    curves = []

    def prepare_data(d, init):
        """Prepare entry for FormLayout.
        `d` is a mapping of shorthands to style names (a single style may
        have multiple shorthands, in particular the shorthands `None`,
        `"None"`, `"none"` and `""` are synonyms); `init` is one shorthand
        of the initial style.
        This function returns an list suitable for initializing a
        FormLayout combobox, namely `[initial_name, (shorthand,
        style_name), (shorthand, style_name), ...]`.
        """
        if matplotlib.__version__ >= "3.0.0":
            if init not in d:
                d = {**d, init: str(init)}
        # Drop duplicate shorthands from dict (by overwriting them during
        # the dict comprehension).
        name2short = {name: short for short, name in d.items()}
        # Convert back to {shorthand: name}.
        short2name = {short: name for name, short in name2short.items()}
        # Find the kept shorthand for the style specified by init.
        canonical_init = name2short[d[init]]
        # Sort by representation and prepend the initial value.
        return [canonical_init] + sorted(
            short2name.items(), key=lambda short_and_name: short_and_name[1]
        )

    if matplotlib.__version__ >= "3.0.0":
        # BEGIN OF MATPLOTLIB 3.X
        curvelabels = sorted(linedict, key=cmp_key)
        for label in curvelabels:
            line = linedict[label]
            color = mcolors.to_hex(
                mcolors.to_rgba(line.get_color(), line.get_alpha()), keep_alpha=True
            )
            ec = mcolors.to_hex(
                mcolors.to_rgba(line.get_markeredgecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            fc = mcolors.to_hex(
                mcolors.to_rgba(line.get_markerfacecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            curvedata = [
                ("Label", label),
                sep,
                (None, "<b>Line</b>"),
                ("Line style", prepare_data(LINESTYLES, line.get_linestyle())),
                ("Draw style", prepare_data(DRAWSTYLES, line.get_drawstyle())),
                ("Width", line.get_linewidth()),
                ("Color (RGBA)", color),
                sep,
                (None, "<b>Marker</b>"),
                ("Style", prepare_data(MARKERS, line.get_marker())),
                ("Size", line.get_markersize()),
                ("Face color (RGBA)", fc),
                ("Edge color (RGBA)", ec),
            ]
            curves.append([curvedata, label, ""])
        # Is there a curve displayed?
        has_curve = bool(curves)

        # Get ScalarMappables.
        mappabledict = {}
        for mappable in [*axes.images, *axes.collections]:
            label = mappable.get_label()
            if label == "_nolegend_" or mappable.get_array() is None:
                continue
            mappabledict[label] = mappable
        mappablelabels = sorted(mappabledict, key=cmp_key)
        mappables = []

        if matplotlib.__version__ < "3.3.0":
            cm_cmaps = cm.cmap_d
        elif matplotlib.__version__ >= "3.6.0":
            cm_cmaps = cm._colormaps
        else:
            cm_cmaps = cm._cmap_registry

        cmaps = [(cmap, name) for name, cmap in sorted(cm_cmaps.items())]
        for label in mappablelabels:
            mappable = mappabledict[label]
            cmap = mappable.get_cmap()
            if cmap not in cm_cmaps.values():
                cmaps = [(cmap, cmap.name), *cmaps]
            low, high = mappable.get_clim()
            mappabledata = [
                ("Label", label),
                ("Colormap", [cmap.name] + cmaps),
                ("Min. value", low),
                ("Max. value", high),
            ]
            if hasattr(mappable, "get_interpolation"):  # Images.
                interpolations = [
                    (name, name) for name in sorted(mimage.interpolations_names)
                ]
                mappabledata.append(
                    ("Interpolation", [mappable.get_interpolation(), *interpolations])
                )
            mappables.append([mappabledata, label, ""])
        # Is there a scalarmappable displayed?
        has_sm = bool(mappables)

        datalist = [(general, "Axes", "")]
        if curves:
            datalist.append((curves, "Curves", ""))
        if mappables:
            datalist.append((mappables, "Images, etc.", ""))

        def apply_callback(data):
            """This function will be called to apply changes"""
            orig_xlim = axes.get_xlim()
            orig_ylim = axes.get_ylim()

            general = data.pop(0)
            curves = data.pop(0) if has_curve else []
            mappables = data.pop(0) if has_sm else []
            if data:
                raise ValueError("Unexpected field")

            # Set / General
            (
                title,
                xmin,
                xmax,
                xlabel,
                xscale,
                ymin,
                ymax,
                ylabel,
                yscale,
                generate_legend,
            ) = general

            if axes.get_xscale() != xscale:
                axes.set_xscale(xscale)
            if axes.get_yscale() != yscale:
                axes.set_yscale(yscale)

            axes.set_title(title)
            axes.set_xlim(xmin, xmax)
            axes.set_xlabel(xlabel)
            axes.set_ylim(ymin, ymax)
            axes.set_ylabel(ylabel)

            # Restore the unit data
            axes.xaxis.converter = xconverter
            axes.yaxis.converter = yconverter
            axes.xaxis.set_units(xunits)
            axes.yaxis.set_units(yunits)
            axes.xaxis._update_axisinfo()
            axes.yaxis._update_axisinfo()

            # Set / Curves
            for index, curve in enumerate(curves):
                line = linedict[curvelabels[index]]
                (
                    label,
                    linestyle,
                    drawstyle,
                    linewidth,
                    color,
                    marker,
                    markersize,
                    markerfacecolor,
                    markeredgecolor,
                ) = curve
                line.set_label(label)
                line.set_linestyle(linestyle)
                line.set_drawstyle(drawstyle)
                line.set_linewidth(linewidth)
                rgba = mcolors.to_rgba(color)
                line.set_alpha(None)
                line.set_color(rgba)
                if marker != "none":
                    line.set_marker(marker)
                    line.set_markersize(markersize)
                    line.set_markerfacecolor(markerfacecolor)
                    line.set_markeredgecolor(markeredgecolor)

            # Set ScalarMappables.
            for index, mappable_settings in enumerate(mappables):
                mappable = mappabledict[mappablelabels[index]]
                if len(mappable_settings) == 5:
                    label, cmap, low, high, interpolation = mappable_settings
                    mappable.set_interpolation(interpolation)
                elif len(mappable_settings) == 4:
                    label, cmap, low, high = mappable_settings
                mappable.set_label(label)
                mappable.set_cmap(cm.get_cmap(cmap))
                mappable.set_clim(*sorted([low, high]))

            # re-generate legend, if checkbox is checked
            if generate_legend:
                draggable = None
                ncol = 1
                if axes.legend_ is not None:
                    old_legend = axes.get_legend()
                    draggable = old_legend._draggable is not None
                    ncol = old_legend._ncol
                new_legend = axes.legend(ncol=ncol)
                if matplotlib.__version__ >= "3.0.0":
                    if new_legend:
                        new_legend.set_draggable(draggable)
                else:
                    if new_legend:
                        new_legend.draggable(draggable)

            # Redraw
            figure = axes.get_figure()
            figure.canvas.draw()
            if not (axes.get_xlim() == orig_xlim and axes.get_ylim() == orig_ylim):
                figure.canvas.toolbar.push_current()

        # END OF MATPLOTLIB 3.X
    else:
        # BEGIN OF MATPLOTLIB 2.X
        curvelabels = sorted(linedict, key=cmp_key)
        for label in curvelabels:
            line = linedict[label]
            color = mcolors.to_hex(
                mcolors.to_rgba(line.get_color(), line.get_alpha()), keep_alpha=True
            )
            ec = mcolors.to_hex(
                mcolors.to_rgba(line.get_markeredgecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            fc = mcolors.to_hex(
                mcolors.to_rgba(line.get_markerfacecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            curvedata = [
                ("Label", label),
                sep,
                (None, "<b>Line</b>"),
                ("Line style", prepare_data(LINESTYLES, line.get_linestyle())),
                ("Draw style", prepare_data(DRAWSTYLES, line.get_drawstyle())),
                ("Width", line.get_linewidth()),
                ("Color (RGBA)", color),
                sep,
                (None, "<b>Marker</b>"),
                ("Style", prepare_data(MARKERS, line.get_marker())),
                ("Size", line.get_markersize()),
                ("Face color (RGBA)", fc),
                ("Edge color (RGBA)", ec),
            ]
            curves.append([curvedata, label, ""])
        # Is there a curve displayed?
        has_curve = bool(curves)

        # Get / Images
        imagedict = {}
        for image in axes.get_images():
            label = image.get_label()
            if label == "_nolegend_":
                continue
            imagedict[label] = image
        imagelabels = sorted(imagedict, key=cmp_key)
        images = []
        cmaps = [(cmap, name) for name, cmap in sorted(cm.cmap_d.items())]
        for label in imagelabels:
            image = imagedict[label]
            cmap = image.get_cmap()
            if cmap not in cm.cmap_d.values():
                cmaps = [(cmap, cmap.name)] + cmaps
            low, high = image.get_clim()
            imagedata = [
                ("Label", label),
                ("Colormap", [cmap.name] + cmaps),
                ("Min. value", low),
                ("Max. value", high),
                (
                    "Interpolation",
                    [image.get_interpolation()]
                    + [(name, name) for name in sorted(mimage.interpolations_names)],
                ),
            ]
            images.append([imagedata, label, ""])
        # Is there an image displayed?
        has_image = bool(images)

        datalist = [(general, "Axes", "")]
        if curves:
            datalist.append((curves, "Curves", ""))
        if images:
            datalist.append((images, "Images", ""))

        def apply_callback(data):
            """This function will be called to apply changes"""
            orig_xlim = axes.get_xlim()
            orig_ylim = axes.get_ylim()

            general = data.pop(0)
            curves = data.pop(0) if has_curve else []
            images = data.pop(0) if has_image else []
            if data:
                raise ValueError("Unexpected field")

            # Set / General
            (
                title,
                xmin,
                xmax,
                xlabel,
                xscale,
                ymin,
                ymax,
                ylabel,
                yscale,
                generate_legend,
            ) = general

            if axes.get_xscale() != xscale:
                axes.set_xscale(xscale)
            if axes.get_yscale() != yscale:
                axes.set_yscale(yscale)

            axes.set_title(title)
            axes.set_xlim(xmin, xmax)
            axes.set_xlabel(xlabel)
            axes.set_ylim(ymin, ymax)
            axes.set_ylabel(ylabel)

            # Restore the unit data
            axes.xaxis.converter = xconverter
            axes.yaxis.converter = yconverter
            axes.xaxis.set_units(xunits)
            axes.yaxis.set_units(yunits)
            axes.xaxis._update_axisinfo()
            axes.yaxis._update_axisinfo()

            # Set / Curves
            for index, curve in enumerate(curves):
                line = linedict[curvelabels[index]]
                (
                    label,
                    linestyle,
                    drawstyle,
                    linewidth,
                    color,
                    marker,
                    markersize,
                    markerfacecolor,
                    markeredgecolor,
                ) = curve
                line.set_label(label)
                line.set_linestyle(linestyle)
                line.set_drawstyle(drawstyle)
                line.set_linewidth(linewidth)
                rgba = mcolors.to_rgba(color)
                line.set_alpha(None)
                line.set_color(rgba)
                if marker is not None:
                    line.set_marker(marker)
                    line.set_markersize(markersize)
                    line.set_markerfacecolor(markerfacecolor)
                    line.set_markeredgecolor(markeredgecolor)

            # Set / Images
            for index, image_settings in enumerate(images):
                image = imagedict[imagelabels[index]]
                label, cmap, low, high, interpolation = image_settings
                image.set_label(label)
                image.set_cmap(cm.get_cmap(cmap))
                image.set_clim(*sorted([low, high]))
                image.set_interpolation(interpolation)

            # re-generate legend, if checkbox is checked
            if generate_legend:
                draggable = None
                ncol = 1
                if axes.legend_ is not None:
                    old_legend = axes.get_legend()
                    draggable = old_legend._draggable is not None
                    ncol = old_legend._ncol
                new_legend = axes.legend(ncol=ncol)
                if new_legend:
                    new_legend.draggable(draggable)

            # Redraw
            figure = axes.get_figure()
            figure.canvas.draw()
            if not (axes.get_xlim() == orig_xlim and axes.get_ylim() == orig_ylim):
                figure.canvas.toolbar.push_current()

        # END OF MATPLOTLIB 2.X

    _dialogEditProperties.fedit(
        matplotlibplotUiItem, datalist, title="Figure options", apply=apply_callback
    )
