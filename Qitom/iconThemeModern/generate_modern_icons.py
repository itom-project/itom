"""Generator of the icon themes 'modern bright' and 'modern dark' of itom.

The modern icon themes contain a refined version of the classic icons. The motifs and colors of the icons
are kept, but they look more modern:

* The gloss and the shading of the colored areas are flattened to calm color areas. Neutral areas (paper,
  metal, outlines) keep their detail.
* The colors are slightly calmer, light paper outlines are slightly crisper.
* In the dark variant, black glyphs and outlines are lifted, such that they stay visible on dark backgrounds.

The icon themes are loaded like the classic icon themes (iconThemeBright.rcc and iconThemeDark.rcc): a binary
resource file is registered after the built-in resources. See README.md for the rules of the resource files.

Input:
    ../iconThemeBright.qrc, ../iconThemeDark.qrc   the classic icons (all icons that depend on the theme)

Output (this folder and the parent folder 'Qitom'):
    bright/*.png, dark/*.png                    the refined images
    ../iconThemeModernBright.qrc, ../iconThemeModernDark.qrc
    (the .rcc files are built by CMake, see iconThemes/CMakeLists.txt)

Usage:
    python generate_modern_icons.py [--rcc PATH_TO_RCC --rcc-out FOLDER]

Requirements: Python 3 with numpy, opencv-python and pillow.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
QITOM = os.path.dirname(HERE)

# images that are not refined: logos, splash screens and animations
KEEP_CLASSIC = ("itomicon/", "splash")


def read_resources(qrc_name):
    """returns a list of (prefix, alias, file) of a qrc file."""
    with open(os.path.join(QITOM, qrc_name), encoding="utf-8") as f:
        text = f.read()
    result = []
    prefix = None
    for m in re.finditer(r'<qresource prefix="([^"]*)">|<file(?: alias="([^"]*)")?>([^<]*)</file>', text):
        if m.group(1) is not None:
            prefix = m.group(1)
        else:
            result.append((prefix, m.group(2) or m.group(3), m.group(3)))
    return result


def _fill_transparent(rgb, alpha):
    """fills the color of transparent pixels with the colors of the neighbourhood, such that the filters do
    not mix in the (arbitrary) color of invisible pixels."""
    num = cv2.GaussianBlur(rgb * alpha[..., None], (0, 0), 3)
    den = cv2.GaussianBlur(alpha, (0, 0), 3)[..., None]
    return np.where(alpha[..., None] > 0.5, rgb, num / np.maximum(den, 1e-4))


def refine(image, dark=False, strength=1.0, saturation=0.95, scale=3):
    """returns the refined version of an RGBA image."""
    a = np.asarray(image.convert("RGBA")).astype(np.float32)
    rgb0, alpha = a[..., :3], a[..., 3] / 255.0  # the original alpha channel (soft edges) is kept
    rgb = _fill_transparent(rgb0, alpha)

    # flatten gradients and gloss (at a higher resolution, such that the edges stay smooth)
    big = np.clip(cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), 0, 255)
    flat = big.astype(np.uint8)
    for _ in range(2):
        flat = cv2.bilateralFilter(flat, 9, 40 * strength, 6)
    flat = cv2.pyrMeanShiftFiltering(flat, 5, 22 * strength, 1)
    flat = cv2.resize(flat, (rgb0.shape[1], rgb0.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

    # colored areas are flattened, neutral areas (paper, metal, outlines) keep their detail
    hsv0 = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    colored = cv2.GaussianBlur(np.clip((hsv0[..., 1] / 255.0 - 0.25) / 0.35, 0, 1), (0, 0), 0.8)
    mixed = rgb * (1 - colored[..., None]) + flat * colored[..., None]

    hsv = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 - (1 - saturation) * colored), 0, 255)  # calmer colors

    # crisper outlines of light objects (e.g. paper), such that they do not fade out on a white background
    light_neutral = (hsv[..., 1] < 40) & (hsv[..., 2] > 190) & (hsv[..., 2] < 245) & (alpha > 0.3)
    hsv[..., 2] = np.where(light_neutral, hsv[..., 2] * 0.93, hsv[..., 2])

    if dark:  # black glyphs and outlines stay visible on dark backgrounds, the contrast is kept
        neutral = (hsv[..., 1] < 70) & (hsv[..., 2] < 120)
        hsv[..., 2] = np.where(neutral, 95 + hsv[..., 2] * 0.85, hsv[..., 2])

    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(np.dstack([out, (alpha * 255).astype(np.uint8)]), "RGBA")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rcc", help="path of the Qt resource compiler, compiles the .rcc files")
    parser.add_argument("--rcc-out", help="folder for the compiled .rcc files (with --rcc)")
    args = parser.parse_args()

    for variant in ("bright", "dark"):
        classic = read_resources("iconThemeBright.qrc" if variant == "bright" else "iconThemeDark.qrc")
        out_root = os.path.join(HERE, variant)
        shutil.rmtree(out_root, ignore_errors=True)

        lines = ["<RCC>"]
        current_prefix = None
        refined = 0

        for prefix, alias, filename in classic:
            if prefix != current_prefix:
                if current_prefix is not None:
                    lines.append("    </qresource>")
                lines.append(f'    <qresource prefix="{prefix}">')
                current_prefix = prefix

            target = filename  # not refined images keep the classic image
            if filename.lower().endswith(".png") and not any(k in filename for k in KEEP_CLASSIC):
                folder = os.path.join(out_root, prefix.strip("/"))
                os.makedirs(folder, exist_ok=True)
                name = alias.replace("/", "__")
                if not name.lower().endswith(".png"):
                    name += ".png"
                path = os.path.join(folder, name)
                refine(Image.open(os.path.join(QITOM, filename)), dark=(variant == "dark")).save(path, "PNG", optimize=True)
                target = os.path.relpath(path, QITOM).replace(os.sep, "/")
                refined += 1

            lines.append(f'        <file alias="{alias}">{target}</file>')

        lines.append("    </qresource>")
        lines.append("</RCC>")
        qrc_name = f"iconThemeModern{variant.capitalize()}.qrc"
        with open(os.path.join(QITOM, qrc_name), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"generated {qrc_name}: {len(classic)} icons, {refined} refined")

        if args.rcc:
            out_dir = args.rcc_out or HERE
            os.makedirs(out_dir, exist_ok=True)
            subprocess.check_call(
                [
                    args.rcc,
                    "--binary",
                    "--format-version",
                    "2",
                    "--compress-algo",
                    "zlib",
                    "-o",
                    os.path.join(out_dir, f"iconThemeModern{variant.capitalize()}.rcc"),
                    os.path.join(QITOM, qrc_name),
                ],
                cwd=QITOM,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
