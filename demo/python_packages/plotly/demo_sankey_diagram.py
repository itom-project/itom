"""Sankey diagram
==============

This demo shows how to use the Python ``plotly`` package under itom.

In order to show plotly outputs in itom figures, it is only necessary
to import the ``itomPlotlyRenderer`` module once at the beginning of your
script. This adds an itom specific renderer to Plotly and uses it as default.

The plotly outputs are then shown in an itom designerPlugin with the name
``plotlyPlot``.
"""

import itomPlotlyRenderer

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoSankeyDiagram.png'

import plotly.graph_objects as go
import json

with open("sankey_energy.json") as file:
    data = json.load(file)

# override gray link colors with 'source' colors
opacity = 0.4
# change 'magenta' to its 'rgba' value to add opacity
data["data"][0]["node"]["color"] = [
    "rgba(255,0,255, 0.8)" if color == "magenta" else color
    for color in data["data"][0]["node"]["color"]
]
data["data"][0]["link"]["color"] = [
    data["data"][0]["node"]["color"][src].replace("0.8", str(opacity))
    for src in data["data"][0]["link"]["source"]
]

fig = go.Figure(
    data=[
        go.Sankey(
            valueformat=".0f",
            valuesuffix="TWh",
            # Define nodes
            node=dict(
                pad=15,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=data["data"][0]["node"]["label"],
                color=data["data"][0]["node"]["color"],
            ),
            # Add links
            link=dict(
                source=data["data"][0]["link"]["source"],
                target=data["data"][0]["link"]["target"],
                value=data["data"][0]["link"]["value"],
                label=data["data"][0]["link"]["label"],
                color=data["data"][0]["link"]["color"],
            ),
        )
    ]
)

fig.update_layout(
    title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, "
    "Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
    font_size=10,
)
fig.show()
