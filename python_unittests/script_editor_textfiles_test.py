import os
import tempfile
import unittest

import itom


class ScriptEditorTextFilesTest(unittest.TestCase):
    """Opens text-like files (besides python files) in the script editor.

    The test only checks that opening the files does not raise an error.
    The syntax highlighting and the disabled python features have to be
    checked manually (see the documentation of the script editor).
    """

    CONTENTS = {
        "test.txt": "plain text\nsecond line\n",
        "test.json": '{"a": 1, "b": [true, null], "c": "text"}\n',
        "test.yaml": "# comment\nkey: value\nlist:\n  - 1\n  - two\n",
        "test.xml": '<?xml version="1.0"?>\n<!-- c -->\n<a b="c">text</a>\n',
        "test.ini": "; comment\n[section]\nkey = value\n",
        "test.md": "# Headline\n\n* item\n\n```\ncode\n```\n",
        "test.cmake": "# comment\nif(WIN32)\n  set(A ${B})\nendif()\n",
        "test.bat": "REM comment\necho %PATH%\n",
    }

    def test_openTextFiles(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for name, content in self.CONTENTS.items():
                filename = os.path.join(tmpdirname, name)

                with open(filename, "w", encoding="utf-8") as fp:
                    fp.write(content)

                # raises a RuntimeError if the file cannot be opened
                itom.openScript(filename)

    def test_openMissingFile(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # a non-existing file must not crash itom. The editor shows a message
            # or ignores the call. Both are fine, as long as no exception
            # other than a RuntimeError is raised.
            try:
                itom.openScript(os.path.join(tmpdirname, "missing.json"))
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main(module="script_editor_textfiles_test", exit=False)
