"""1D func animation
===============

This example demonstrates how matplotlib can be used to create
animated movie and export these in the mp4 movie format.
It is shown here with some random generated 2d images, which ware plotted via matplotlib.
By using the figure handle the animation is created. So you can plot your matplot
figures in your own way and used some similar syntax to create an animation.

First of all you must install the matplotlib package, e.g. from
https://pypi.org/project/matplotlib/

Then you must install the ffmpeg codec. A detailed description can be found on:
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/

The build version of the ffmpeg codec can be downloaded here:
http://ffmpeg.zeranoe.com/builds/

Download and unzip the builds files to your harddrive. Typically the folder is like:
C:\\Program files\\ffmpeg

The bin folder of ffmpeg must be added to the path variables of your system:
C:\\Program files\\ffmpeg\\bin

Finally start the command prompt and run the command:
C:\\Program files\\ffmpeg\\bin\\ffmpeg.exe -codecs

or easier:
ffmpeg -codecs

"""

import numpy as np

###############################################################################
# Import namespaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


###############################################################################
# Function to generate parametric plot
def parametric_plot(T, e):
    t = np.linspace(0, T, 1000)
    x = np.cos(-t)
    y = np.cos(-t + e)

    plt.plot(x, y, color="b")

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"T = {T:.2f}, e = {e:.2f}")


###############################################################################
# Function to calculate ellipse parameters
def calculate_ellipse_params(T, e):
    t = np.linspace(0, T, 1000)
    x = np.cos(-t)
    y = np.cos(-t + e)

    # Calculate width and height
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    # Calculate covariance matrix
    covariance_matrix = np.cov(x, y)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Get angle of rotation (angle of major axis with respect to x-axis)
    major_axis_index = np.argmax(eigenvalues)
    major_axis_vector = eigenvectors[:, major_axis_index]
    angle_rad = np.arctan2(major_axis_vector[1], major_axis_vector[0])
    angle_deg = np.degrees(angle_rad)

    return width, height, angle_deg


###############################################################################
# Calculate ellipse parameters for a full oscillation
T_full = 2 * np.pi
e_full = 45
width, height, angle_deg = calculate_ellipse_params(T_full, e_full)

# Create figure and plot ellipse
fig, ax = plt.subplots()


# Setting up the animation
def update(frame):
    ax.clear()
    parametric_plot(frame, e)


e = 45
ani = FuncAnimation(
    fig, update, frames=np.linspace(0.01, T_full, 100), repeat=True, interval=50
)

plt.show()
