"""This script exports the svg files to png files with different
sizes using Inkscape (indicate abs path to inkscape.exe below).

It is assumed that the main color in the svg raw files is #ff0000.
This color is replaced by different colors, depending on icon subtype.
"""

from itom import rgba, filter
import tempfile
import subprocess               # May want to use subprocess32 instead
import os


def saveSVGasPNG(svgFilename: str,
                 outputFilename: str,
                 colorStr: str,
                 width: int = 32):
    bytestring = open(svgFilename).read().encode('utf-8')
    bytestring = bytestring.replace(b"#ff0000", colorStr)

    tempdir = tempfile.gettempdir()

    outfile = os.path.join(tempdir, "itomtemp.svg")

    with open(outfile, 'wb') as fp:
        fp.write(bytestring)

    cmd_list = [ r'C:\Program Files\Inkscape\inkscape.exe', '-z',
                 '--export-png', outputFilename,
                 '--export-width', str(width),
                 '--export-height', str(width),
                 outfile ]

    # Invoke the command.  Divert output that normally goes to stdout or stderr.
    p = subprocess.Popen( cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE )

    # Below, < out > and < err > are strings or < None >, derived from stdout and stderr.
    out, err = p.communicate()      # Waits for process to terminate

    # Maybe do something with stdout output that is in < out >
    # Maybe do something with stderr output that is in < err >

    if p.returncode:
        raise Exception( 'Inkscape error: ' + (err or '?')  )



def colorFromHex(number: int):
    red = (number & 0xff0000) >> 16
    green = (number & 0x00ff00) >> 8
    blue = (number & 0x0000ff)
    return rgba(red, green, blue)

if __name__ == "__main__":

    infolder = r"."
    outfolder = r".\..\rc"

    infolder = os.path.abspath(infolder)
    outfolder = os.path.abspath(outfolder)

    for infile in os.listdir(infolder):
        if infile.endswith(".svg"):
            print("process", infile)

            outfile = os.path.join(outfolder, os.path.basename(infile)[0:-4])
            infile = os.path.join(infolder, infile)

            saveSVGasPNG(infile, outfile + "_focus.png", b"#ff9800", width=32)
            saveSVGasPNG(infile, outfile + "_focus@2x.png", b"#ff9800", width=64)

            saveSVGasPNG(infile, outfile + "_pressed.png", b"#996819", width=32)
            saveSVGasPNG(infile, outfile + "_pressed@2x.png", b"#996819", width=64)


    #filename = r"C:\itom\sources\itom\styles\stylesheets\darkOrangeStyle2\svg\radio_checked.svg"

    #saveSVGasPNG(filename, filename + ".png", b"#ff9800")
