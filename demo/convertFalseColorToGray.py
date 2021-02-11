# img1 contains an rgba-image with false color encoded color information and gray values
# the gray values stay like they are, the false color values are decoded in HSV color space
# and gray values with respect to their hue are determined. The resulting image is called img2
import colorsys

[height, width] = img1.shape
img2 = img1.copy()

for m in range(height):
    for n in range(width):
        px = img1[m, n]
        [h, s, v] = colorsys.rgb_to_hsv(px.r / 255.0, px.g / 255.0, px.b / 255.0)
        if s > 0.3:
            h_ = int(v * h * 255)
            img2[m, n] = itom.rgba(h_, h_, h_)
