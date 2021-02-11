from MLJ050 import *

zstage = MLJ050()

print(zstage.HWinfo)
print("SN = " + str(zstage.SN))

zstage.jogUp()
zstage.jogUp()
zstage.jogDown()
print(zstage.getPos())
zstage.moveHome()
zstage.moveAbs(1.2)
zstage.moveRel(10.01)
print(zstage.getPos())
zstage.moveHome()
print(zstage.getPos())

del zstage
