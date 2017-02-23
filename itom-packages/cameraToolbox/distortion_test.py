import numpy as np
import distortion

'''Demo script to show the usage and performance of the distortion module'''

close('all')

center = (330, 250) #center point of undistorted grid (x,y)
pitch = 43 #px
rotation = 0
cols = 11
rows = 9
k1 = 5e-7
k2 = 5e-12
k3 = 0

points_u = distortion.getPointGrid([9,11], pitch, center, rotation, '3d')
points_d = distortion.getPointGridDistorted([9,11], pitch, center, rotation, k1, k2, k3, '3d')

x0 = distortion.guessInitialParameters(points_d, rows, cols, withDistortion = False)

coeffs_coarse, opt_coarse_info = distortion.fitGrid(points_d, rows = rows, cols = cols, x0 = x0, withDistortion = False)
coeffs_fine_init = distortion.expandInitialParametersByDistortion(coeffs_coarse)
coeffs_fine, opt_fine_info = distortion.fitGrid(points_d, rows = rows, cols = cols, x0 = coeffs_fine_init, withDistortion = True)

points_corrected = distortion.undistortPointGrid(points_d, rows, cols, coeffs_fine, repr = '3d')

plot(distortion.drawPointGrid(points_u, rows, cols), properties={"title":"undistorted image"})
plot(distortion.drawPointGrid(points_d, rows, cols), properties={"title":"distorted image"})
plot(distortion.drawPointGrid(points_corrected, rows, cols), properties={"title":"corrected image"})
plot(distortion.createDistortionMap(coeffs_fine, points_d, rows, cols), properties={"title":"Distortion", \
    "colorBarVisible":True, "colorMap":"falseColor"})

print("coefficients (design): %.2f %.2f %.2f %.3f %.4fe6 %.4fe12 %.4fe18" % (pitch, *center, rotation, k1*1e6, k2*1e12, k3*1e18))
print("coefficients (coarse): %.2f %.2f %.2f %.3f %.4fe6 %.4fe12 %.4fe18" % (*coeffs_coarse[0:4], 0, \
      0,0))
print("coefficients (optim): %.2f %.2f %.2f %.3f %.4fe6 %.4fe12 %.4fe18" % (*coeffs_fine[0:4], coeffs_fine[4]*1e6, \
      coeffs_fine[5]*1e12,coeffs_fine[6]*1e18))
