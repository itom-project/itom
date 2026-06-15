import numpy as np

try:
    import distortion
except ModuleNotFoundError:
    from . import distortion

import itom

"""Demo script to show the usage and performance of the distortion module"""


def distortionTest():
    itom.close("all")

    center = (330, 250)  # center point of undistorted grid (x,y)
    pitchX = 24  # px
    pitchY = 22  # px
    pitch = (pitchX, pitchY)
    rotation = 0
    cols = 22
    rows = 18
    k1 = 5e-7
    k2 = 5e-12
    k3 = 0

    points_u = distortion.getPointGrid([rows, cols], pitch, center, rotation, "3d")
    points_d = distortion.getPointGridDistorted(
        [rows, cols], pitch, center, rotation, k1, k2, k3, "3d"
    )

    x0 = distortion.guessInitialParameters(
        points_d, rows, cols, withDistortion=False, withRotation=True
    )

    coeffs_coarse, opt_coarse_info = distortion.fitGrid(
        points_d, rows=rows, cols=cols, x0=x0, withDistortion=False, withRotation=True,
    )
    coeffs_fine, opt_fine_info = distortion.fitGrid(
        points_d,
        rows=rows,
        cols=cols,
        x0=coeffs_coarse,
        withDistortion=True,
        withRotation=False,
    )
    coeffs_fine2, opt_fine2_info = distortion.fitGrid(
        points_d,
        rows=rows,
        cols=cols,
        x0=coeffs_fine,
        withDistortion=True,
        withRotation=True,
    )

    points_corrected = distortion.undistortPointGrid(
        points_d, rows, cols, coeffs_fine, repr="3d"
    )

    itom.plot(
        distortion.drawPointGrid(points_u, rows, cols),
        properties={"title": "undistorted image"},
    )
    itom.plot(
        distortion.drawPointGrid(points_d, rows, cols),
        properties={"title": "distorted image"},
    )
    itom.plot(
        distortion.drawPointGrid(points_corrected, rows, cols),
        properties={"title": "corrected image"},
    )
    itom.plot(
        distortion.createDistortionMap(coeffs_fine, points_d, rows, cols),
        properties={
            "title": "Distortion",
            "colorBarVisible": True,
            "colorMap": "falseColor",
        },
    )

    print(
        "coefficients (design): %.2f %.2f %.2f %.2f %.3f %.4fe-6 %.4fe-12 %.4fe-18"
        % (pitchX, pitchY, *center, rotation, k1 * 1e6, k2 * 1e12, k3 * 1e18)
    )
    print(
        "coefficients (coarse): %.2f %.2f %.2f %.2f %.3f %.4fe-6 %.4fe-12 %.4fe-18"
        % (*coeffs_coarse[0:5], 0, 0, 0)
    )
    print(
        "coefficients (optim1): %.2f %.2f %.2f %.2f %.3f %.4fe-6 %.4fe-12 %.4fe-18"
        % (
            *coeffs_fine[0:5],
            coeffs_fine[5] * 1e6,
            coeffs_fine[6] * 1e12,
            coeffs_fine[7] * 1e18,
        )
    )
    print(
        "coefficients (optim2): %.2f %.2f %.2f %.2f %.3f %.4fe-6 %.4fe-12 %.4fe-18"
        % (
            *coeffs_fine2[0:5],
            coeffs_fine2[5] * 1e6,
            coeffs_fine2[6] * 1e12,
            coeffs_fine2[7] * 1e18,
        )
    )


if __name__ == "__main__":
    distortionTest()
