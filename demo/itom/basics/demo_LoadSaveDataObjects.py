"""Load and save dataObject
=========================

This demo shows how to **save** and **load** ``dataObjects``
to/from image formats as well as native itom formats."""

from itom import dataObject
from itom import algorithms
from itom import rgba
from itom import plot
from itom import saveIDC
from itom import loadIDC
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoLoadSaveDataObjects.png'

###############################################################################
# Create a colored dataObject of type ``rgba32``.

rgba32 = dataObject([100, 100], "rgba32")

###############################################################################
# Set all pixels to a gray value.
# Therefore ``red=green=blue`` with no transparency,
# what means that alpha has to be set to the maximal value of ``255``.
rgba32[0:100, 0:100] = rgba(150, 150, 150, 255)

"""insert a red, green and blue bar in the picture wich are not complete intransparent"""
rgba32[10:30, :] = rgba(255, 0, 0, 150)
rgba32[50:70, :] = rgba(0, 255, 0, 150)
rgba32[80:100, :] = rgba(0, 0, 255, 150)
"""show the image"""
plot(rgba32)

###############################################################################
# Save the ``dataObject`` as a *.tiff file with a rgba color palette.
algorithms.saveTiff(rgba32, "pic_rgba.tiff", "rgba")

###############################################################################
# Reload the picture as it was, that is of type ``rgba32``.
reload_tiff_rgba = dataObject()
algorithms.loadAnyImage(reload_tiff_rgba, "pic_rgba.tiff", "asIs")

###############################################################################
# Save the ``dataObject`` as a *.tiff file with a rgb color palette,
# which causes that the transparency of the bars will be ignored. 
# If ``gray`` or ``gray16`` is choosen as color palette the colored
# ``dataObject`` will be converted to a gray image 
algorithms.saveTiff(rgba32, "pic_rgb.tiff", "rgb")

###############################################################################
# Reload the picture as it was, that is of type ``rgba32``
# with all alpha values set to ``255`` (no transparency).
reload_tiff_rgb = dataObject()
algorithms.loadAnyImage(reload_tiff_rgb, "pic_rgb.tiff", "asIs")

###############################################################################
# Save the ``dataObject`` as a *.png file with a ``gray`` color palette
# (also ``gray16`` and all colored palettes are supportted).
algorithms.savePNG(rgba32, "pic_gray.png", "gray")

###############################################################################
# Reload the picture as it was, that is of type ``gray`` (type ``uint8``) 
reload_png_gray = dataObject()
algorithms.loadAnyImage(reload_png_gray, "pic_gray.png", "asIs")

###############################################################################
# Save the ``dataObject`` as a *.pgm with a 16bit grayscale
# (``gray`` and ``gray16`` are only supported for gray images).
algorithms.savePGM(rgba32, "pic_gray.pgm", "gray16")

###############################################################################
# Load the *.pgm file as it was, that is of type ``gray``
# (type ``uint16`` due to the 16bit gray color palette) 
reload_pgm_gray16 = dataObject()
algorithms.loadAnyImage(reload_pgm_gray16, "pic_gray.pgm", "asIs")

###############################################################################
# Save the ``dataObject`` as an *.idc file (itom data collection,
# saved using Python module ``pickle``) therefore it must be wrapped into a ``dictionary``.
dataDict = {"data": rgba32}
saveIDC("pic_idc.idc", dataDict)

###############################################################################
# Load the *.idc file as it was, that is of type ``dictionary``.
loaded_dic = loadIDC("pic_idc.idc")
reload_img = loaded_dic["data"]

###############################################################################
# Copy the dataObject
rgba32_1 = rgba32

###############################################################################
# Save both (also more possible) in one *.idc file.
dic_1 = {"data_1": rgba32, "data_2": rgba32_1}
loaded_dic_1 = saveIDC("multi_pic_idc.idc", dic_1)

###############################################################################
# In this section a ``uint8`` ``dataObject`` is created and saved in false colors.
# create a gray image of type uint8
uint8 = dataObject([100, 100], "uint8")

# insert blocks with values of 0.0, 1.0, 50 and 100
uint8[0:25, :] = 0
uint8[25:50, :] = 1
uint8[50:75, :] = 50
uint8[75:100, :] = 100

###############################################################################
# Save as *.tiff file colored in the ``hotIron`` color palette.
# Other palettes are for example ``grayMarked`` or ``falseColor``.
algorithms.saveTiff(uint8, "pic_uint8.tiff", "hotIron")

###############################################################################
# This section shows how to save floating point ``dataObjects`` as a image.
# create a gray image of type float32
float32 = dataObject([100, 100], "float32")

# insert blocks with values of 0.0, 1.0, 50 and 100
float32[0:25, :] = 0.0
float32[25:50, :] = 1.0
float32[50:75, :] = 50.0
float32[75:100, :] = 100.0

###############################################################################
# Save the ``float32`` ``dataObject`` as a *.png file
# with a ``falseColor`` palette (here ``hotIron`` is used,
# others are for example ``grayMarked`` or ``falseColor``).
# If you save a ``dataObject`` of type float the color palette is spaced between
# ``[0, 1]`` ->all values above ``1.0`` will be clipped to the maximum value.
algorithms.savePNG(float32, "pic_falseColor.png", "hotIron")

###############################################################################
# Reload the saved *.png as a ``uint8`` ``dataObject``
# ->all steps with values above 1.0 have the same gray value.
reload_png_falseColor = dataObject()
algorithms.loadAnyImage(reload_png_falseColor, "pic_falseColor.png", "GRAY")

###############################################################################
# To get rid of the problem above you need to normalize your
# ``dataObject`` between ``0.0`` and ``1.0`` using the function ``normalize``.
normfloat32 = float32.normalize(0.0, 1.0, "float32")
algorithms.savePNG(normfloat32, "pic_normalized_falseColor.png", "hotIron")

###############################################################################
# Reload the image as a ``uint8`` ``dataObject``
# ->all steps are included.
reload_normalized_falseColor = dataObject()
algorithms.loadAnyImage(
    reload_normalized_falseColor,
    "pic_normalized_falseColor.png",
    "GRAY",
)