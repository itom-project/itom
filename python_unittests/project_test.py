import json
import os
import shutil
import tempfile
import unittest

import itom


class ProjectTest(unittest.TestCase):
    """Tests the project API of itom (itom.openProject, newProject, saveProject,
    closeProject, getProjectInfo).

    The project files are created in a temporary directory. The current working
    directory is restored at the end, since opening a project may change it.
    """

    def setUp(self):
        self.cwd = itom.getCurrentPath()
        self.tmp = os.path.normpath(tempfile.mkdtemp(prefix="itom_project_"))
        os.makedirs(os.path.join(self.tmp, "src", "lib"))
        with open(os.path.join(self.tmp, "src", "main.py"), "w") as fp:
            fp.write("x = 1\n")

    def tearDown(self):
        itom.closeProject()
        itom.setCurrentPath(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _norm(self, path):
        return os.path.normcase(os.path.normpath(path))

    def test_no_project(self):
        itom.closeProject()
        self.assertIsNone(itom.getProjectInfo())
        itom.closeProject()  # closing twice is fine
        with self.assertRaises(RuntimeError):
            itom.saveProject()

    def test_new_save_open_close(self):
        itom.newProject(self.tmp, "Demo")
        info = itom.getProjectInfo()
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Demo")
        self.assertEqual(self._norm(info["projectDir"]), self._norm(self.tmp))
        self.assertEqual(
            self._norm(info["path"]), self._norm(os.path.join(self.tmp, "Demo.itomproject"))
        )
        self.assertEqual([self._norm(f) for f in info["folders"]], [self._norm(self.tmp)])
        self.assertIsNone(info["mainScript"])
        self.assertEqual(self._norm(info["workingDirectory"]), self._norm(self.tmp))
        self.assertEqual(info["pythonPath"], [])
        self.assertIn("__pycache__", info["excludes"])
        self.assertFalse(info["saved"])
        self.assertFalse(os.path.exists(info["path"]))

        # the cwd follows the project (default setting)
        self.assertEqual(self._norm(itom.getCurrentPath()), self._norm(self.tmp))

        itom.saveProject()
        self.assertTrue(itom.getProjectInfo()["saved"])
        with open(info["path"]) as fp:
            data = json.load(fp)
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["name"], "Demo")
        self.assertEqual(data["folders"][0]["path"], ".")

        itom.closeProject()
        self.assertIsNone(itom.getProjectInfo())

        # reopen from file
        itom.openProject(info["path"])
        info2 = itom.getProjectInfo()
        self.assertEqual(info2["name"], "Demo")
        self.assertTrue(info2["saved"])

    def test_open_project_file_with_settings(self):
        path = os.path.join(self.tmp, "full.itomproject")
        with open(path, "w") as fp:
            json.dump(
                {
                    "version": 1,
                    "name": "Full",
                    "folders": [{"path": "src", "name": "Source"}, "."],
                    "mainScript": "src/main.py",
                    "workingDirectory": "src",
                    "pythonPath": ["src/lib"],
                    "excludes": ["*.tmp"],
                },
                fp,
            )

        itom.openProject(path)
        info = itom.getProjectInfo()
        self.assertEqual(info["name"], "Full")
        self.assertEqual(
            [self._norm(f) for f in info["folders"]],
            [self._norm(os.path.join(self.tmp, "src")), self._norm(self.tmp)],
        )
        self.assertEqual(
            self._norm(info["mainScript"]), self._norm(os.path.join(self.tmp, "src", "main.py"))
        )
        self.assertEqual(
            self._norm(info["workingDirectory"]), self._norm(os.path.join(self.tmp, "src"))
        )
        self.assertEqual(
            [self._norm(p) for p in info["pythonPath"]],
            [self._norm(os.path.join(self.tmp, "src", "lib"))],
        )
        self.assertEqual(info["excludes"], ["*.tmp"])
        self.assertEqual(self._norm(itom.getCurrentPath()), self._norm(os.path.join(self.tmp, "src")))

        # opening another project replaces the first one
        itom.newProject(self.tmp)
        self.assertEqual(itom.getProjectInfo()["name"], os.path.basename(self.tmp))

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            itom.openProject(os.path.join(self.tmp, "missing.itomproject"))
        with self.assertRaises(RuntimeError):
            itom.newProject(os.path.join(self.tmp, "no_such_folder"))

        broken = os.path.join(self.tmp, "broken.itomproject")
        with open(broken, "w") as fp:
            fp.write("{ not json")
        with self.assertRaises(RuntimeError):
            itom.openProject(broken)

        self.assertIsNone(itom.getProjectInfo())

        # save to a folder that does not exist
        itom.newProject(self.tmp)
        with self.assertRaises(RuntimeError):
            itom.saveProject(os.path.join(self.tmp, "nope", "x.itomproject"))

        with self.assertRaises(TypeError):
            itom.openProject()
        with self.assertRaises(TypeError):
            itom.newProject()


if __name__ == "__main__":
    unittest.main(module="project_test", exit=False)
