2D image plots
****************

"itom2DQWTFigure" and "itom2DGVFigure" are the basic plots for visualization of image like DataObjects.
Both plots has line-picker and point pickers included. By pressing "Ctrl" during picker movement, the picker can only be moved 
horizontal or vertical according to the mouse movement.

"itom2DQWTFigure" is designs as a visualization of metrical data, false color or topography measurements.
It supports the axis-scaling / axis offset of DataObjects, offers axis-tags and meta-data handling.
It does not offer plotting of real color images.
All DataTypes are accepted. To plot complex objects, it is possible to select between the following modes: "absolut", "phase", "real" and "imaginary".
The data is plotted mathematically correct. This means the value at [0,0] is in the lower left position.

"itom2DQWTFigure" is designs as a fast visualization of images, e.g. direct grabber output or colored images. 
It allows the ploting of real colors (at the moment only 24-bit or 32-bit). It does not handle meta-data.
All DataTypes are accepted. To plot complex objects, it is possible to select between the following modes: "absolut", "phase", "real" and "imaginary".
The data is plotted image orientated. This means the value at [0,0] is in the upper left position.