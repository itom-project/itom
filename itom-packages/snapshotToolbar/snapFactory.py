import itom
import snapshot as sn


class SnapFactory:
    def __init__(self):

        self.buttons = []
        self.buttons.append(itom.addButton("Snapshot", "Snap!", self.createChild))

    def createChild(self):
        name = self.checkInstance()
        setattr(self, name, sn.Snapshot(name))

    def checkInstance(self):
        i = 1
        name = "snap_%03i" % (i)
        while name in self.__dict__.keys():
            i = i + 1
            name = "snap_%03i" % (i)
        return name


if __name__ == "__main__":
    snapFactory = SnapFactory()
