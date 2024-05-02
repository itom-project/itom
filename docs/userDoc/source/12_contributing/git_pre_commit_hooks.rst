.. _git_pre_commit_hooks:

Git Pre-Commit Hooks
********************

Itom makes use of pre-commit hooks to check for compliance with our coding guidelines.
The pre-commit hooks are run before each commit and your commit can fail accordingly.

When a new hook is added, It's usually a good idea to run the hooks before new commit is added.
Python offers a framework to run and apply the commit-hooks manuall to your code.

.. code-block:: bash

    pip install pre_commit

Then, got to the main folder of the respective itom repository (e.g. itomProject, itom, plugins or designerplugins), which includes the ``pre-commit`` configuration
file **.pre-commit-config.yaml** and run.:

.. code-block:: bash
    
    pre_commit run --all-files
