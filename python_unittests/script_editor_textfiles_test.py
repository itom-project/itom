import os
import shutil
import tempfile
import unittest

import itom


class ScriptEditorTextFilesTest(unittest.TestCase):
    """Opens text-like files (besides python files) in the script editor.

    The test only checks that opening the files does not raise an error.
    The syntax highlighting itself is tested in the C++ unittest
    ``unittest_codeeditor``.

    The files must stay on disk as long as itom is running: the script editor
    watches its files and shows a modal dialog if a file is deleted. Therefore
    a fixed directory in the temp folder is used, that is cleared at the start
    of the next test run (and not at the end of this run).
    """

    CONTENTS = {
        "test.txt": "plain text\nsecond line\n",
        "test.json": '{"a": 1, "b": [true, null], "c": "text"}\n',
        "test.yaml": "# comment\nkey: value\nlist:\n  - 1\n  - two\n",
        "test.yml": "key: value\n",
        "test.xml": '<?xml version="1.0"?>\n<!-- c -->\n<a b="c">text</a>\n',
        "test.html": "<html><body>text</body></html>\n",
        "test.ini": "; comment\n[section]\nkey = value\n",
        "test.cfg": "[section]\nkey = value\n",
        "test.toml": "[table]\nkey = 1\n",
        "test.md": "# Headline\n\n* item\n\n```\ncode\n```\n",
        "test.rst": "Title\n=====\n\n.. note:: text\n",
        "test.cmake": "# comment\nif(WIN32)\n  set(A ${B})\nendif()\n",
        "CMakeLists.txt": "cmake_minimum_required(VERSION 3.12)\nproject(x)\n",
        "test.bat": "REM comment\necho %PATH%\n",
        "test.cmd": "@echo off\n",
        "test.csv": "a,b\n1,2\n",
        "test.log": "line 1\nline 2\n",
        "noSuffix": "plain text without suffix\n",
    }

    tmpdirname = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = os.path.join(tempfile.gettempdir(), "itom_editor_textfiles_test")
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)
        os.makedirs(cls.tmpdirname)

    def test_openTextFiles(self):
        for name, content in self.CONTENTS.items():
            filename = os.path.join(self.tmpdirname, name)

            with open(filename, "w", encoding="utf-8") as fp:
                fp.write(content)

            # raises a RuntimeError if the file cannot be opened
            itom.openScript(filename)

            # opening an already opened file again must not fail, too
            itom.openScript(filename)

    # Hint: opening a missing or a binary file is refused with a modal message box
    # in the GUI. This blocks an automated test run, therefore it is not tested here.
    # The decision which file is editable is covered by the C++ unittest_codeeditor.


if __name__ == "__main__":
    unittest.main(module="script_editor_textfiles_test", exit=False)
