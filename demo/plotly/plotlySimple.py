"""This demo shows how to use the Python plotly package under itom.

In order to show plotly outputs in itom figures, it is only necessary
to import the ``itomPlotlyRenderer`` module once at the beginning of your
script. This adds an itom specific renderer to Plotly and uses it as default.

The plotly outputs are then shown in an itom designerPlugin with the name
``plotlyPlot``.
"""

import itomPlotlyRenderer

# x and y given as array_like objects
import plotly.express as px

fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()

import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))

# it is also possible to pass a renderer name to the ``show`` command.
# Passing ``itom`` is not necessary since it is set as default, however
# it would also be possible to for instance pass ``browser`` such that
# this figure is opened in a new browser tab.
fig.show(renderer="itom")
