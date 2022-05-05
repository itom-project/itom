"""Simple
=========

This demo shows how to use the Python ``plotly`` package under itom.

In order to show plotly outputs in itom figures, it is only necessary
to import the ``itomPlotlyRenderer`` module once at the beginning of your
script. This adds an itom specific renderer to Plotly and uses it as default.

The plotly outputs are then shown in an itom designerPlugin with the name
``plotlyPlot``.
"""

import itomPlotlyRenderer

import plotly.express as px
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPloty.png'

df = px.data.tips()
fig = px.bar(df, x='sex', y='total_bill', facet_col='day', color='smoker', barmode='group',
             template='presentation+plotly'
             )
fig.update_layout(height=400)
fig