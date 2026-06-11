.. _git_pre_commit_hooks:

Git Pre-Commit Hooks
********************

Itom makes use of `pre-commit <https://pre-commit.com/>`_ hooks for
identifying simple issues such as missing semicolons, trailing whitespace or misspellings
before submission to git. The `pre-commit <https://pre-commit.com/>`_ hooks are run
before each commit and your commit can fail accordingly.

After cloning the |itom| repositories (`itomProject <https://github.com/itom-project/itomProject>`_ or
`itom <https://github.com/itom-project/itom>`_, `plugins <https://github.com/itom-project/plugins>`_,
`designerPlugins <https://github.com/itom-project/designerPlugins>`_) hooks must be install e.g. using pip.

.. code-block:: bash

    python -m pip install pre_commit

Git hooks must be installed into the ``.git`` folder of each repo.

.. code-block:: bash

    python -m pre_commit install

Pre_commits can be executed manually:

.. code-block:: bash

    python -m pre-commit run --all-files

A practical tip is to also run pre_commits manually to reduce frustration when you want or need to create a commit.
Failed hooks ensure that a commit cannot be set.

To execute specific hooks, their names can be passed as an additional argument (e.g. codespell).

.. code-block:: bash

    python -m pre-commit run --all-files codespell
