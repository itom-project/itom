pluginDoc >> template
------------------------

This folder contains files and templates, that are used to create the overall plugin qthelp project from
the single html files, shipped with every single plugin.

The final plugin documentation for qthelp is generated once the help viewer is opened in itom for the
first time, or if any plugin help file has changed. Then, the placeholders in index.html and itomPluginDoc.qhp
are filled and the final help is generated in the help folder of itom.