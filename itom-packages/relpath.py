import os
import inspect


def scriptDirectory(frame=None):
    """
    returns absolute path of the script that calls this method.
    Raises exception if this method is directly called from
    command line

    If you provide an inspect-frame, the absolute path of this frame
    is returned.
    """

    if frame is None:
        curframe = inspect.currentframe()
        frame = curframe.f_back  # get caller frame

    if frame is None:
        raise RuntimeError("inspect could not get frame of calling method")

    file = inspect.getfile(frame)

    if file == "<string>":
        raise RuntimeError("this script has been called by the command line")

    return os.path.dirname(os.path.abspath(file))


def makeAbsInScriptDir(relPath, frame=None):

    if frame is None:
        curframe = inspect.currentframe()
        frame = curframe.f_back  # get caller frame

    if frame is None:
        raise RuntimeError("inspect could not get frame of calling method")

    return os.path.abspath(os.path.join(scriptDirectory(frame), relPath))
