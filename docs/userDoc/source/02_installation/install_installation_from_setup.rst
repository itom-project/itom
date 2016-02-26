.. include:: ../include/global.inc

Installation from setup
************************

Currently, there are both a *32bit* and a *64bit* setup version of |itom| available (Windows only). For linux users there is no pre-compiled package available.
The use of the setup version is recommended for users that mainly want to work with |itom| without developing new algorithms or plugins. The setup
can be downloaded from https://bitbucket.org/itom/itom/downloads. The setup intalls the core application, a number of useful hardware plugins and the designer plugins which provide plotting functionalities to |itom|.

For your convenience you can download an installer package for Microsoft Windows from `<https://bitbucket.org/itom/itom/downloads>`.
There are several different types of setups available:

* Windows, 32bit, |itom| only
* Windows, 64bit, |itom| only
* Windows, 32bit, |itom|, Python + packages: numpy, scipy, matplotlib, PIL (optional)
* Windows, 64bit, |itom|, Python + packages: numpy, scipy, matplotlib, PIL (optional)

These setups do not contain any plugins or designer-plugins.

In the course of the installation, the following third-party componenents will be installed along with |itom|:

1. Microsoft Visual C++ 2010 Runtime Libraries (x86 or x64)
2. Python 3.2.3 (optional)
3. Python package *numpy* 1.6.2(optional)
4. Python package *scipy* 0.10.1 (optional)
5. Python package *matplotlib* 1.2.x (optional)
6. Python package *PIL* 1.1.7 (optional)

In the following we will guide you through the installation setup with a couple of screenshots:

.. figure:: images/itom_install/setup01.jpg
    :alt: Language selection
    :scale: 100%
    :align: center
    
    Please select your desired language for the setup.

.. figure:: images/itom_install/setup02.jpg
    :alt: Start screen
    :scale: 100%
    :align: center
    
    The start screen of the setup will appear.

.. figure:: images/itom_install/setup03.jpg
    :scale: 100%
    :align: center
    
    Read the license text and agree to it.

.. figure:: images/itom_install/setup04.jpg
    :scale: 100%
    :align: center
    
    Choose where to install |itom| on your file system. If you press next, the installer checks if the
    chosen directory already exists and warns if so. Please agree if you want to really install itom in the
    existing directory.

.. figure:: images/itom_install/setup05.jpg
    :alt: components
    :scale: 100%
    :align: center
    
    Depending on your version of the setup, you now need to (de)select some optional components.
    The SDK is important if you want to develop your own plugins for |itom|. If you have the extended
    setup version, you can also select that python including some important packages is directly installed.
    This is only recommended, if you do not have python in a similar version already installed on your
    computer. You can also manually install and/or update python or its packages before or after this
    setup.

.. figure:: images/itom_install/setup06.jpg
    :scale: 100%
    :align: center
    
    Select the name of |itom| in your Windows start menu.

.. figure:: images/itom_install/setup07.jpg
    :scale: 100%
    :align: center
    
    Choose whether you want to have an |itom| shortcut on your desktop

.. figure:: images/itom_install/setup08.jpg
    :scale: 100%
    :align: center
    
    Now, a summary of the installation steps is given. Press next if you want to start the installation...

.. figure:: images/itom_install/setup09.jpg
    :scale: 100%
    :align: center
    
    The installation is executed now. |itom| is not copying any files in another folder than the indicated program
    folder (besides python or any python-packages). However this setup creates an application entry in the Windows
    registry in order to allow an uninstall by the default Windows control panel and to check if any version of 
    |itom| already has been installed. When uninstalling |itom|, the registry entry is removed, too.

.. figure:: images/itom_install/setup11.png
    :scale: 100%
    :align: center
    
    If not already available, the Microsoft Visual C++ 2010 Runtime Libraries are installed now. 

.. figure:: images/itom_install/setup12.png
    :scale: 100%
    :align: center
    
Depending on your selected components, python and/or any python packages are now installed:

.. figure:: images/itom_install/install08.png
    :scale: 100%
    :align: center
    
    Install Python 3.4.2 (or the version shipped with your setup version) for the current user or for all users.

.. figure:: images/itom_install/install10.png
    :scale: 100%
    :align: center
    
    Customize your Python installation. We recommend leaving everything as is.

.. figure:: images/itom_install/install11.png
    :scale: 100%
    :align: center
    
    You've completed the Python installation as well. We're getting getting closer.

.. figure:: images/itom_install/install12.png
    :scale: 100%
    :align: center
    
    Now, depending on the selected Python packages, several command lines appear that install the shipped packages 
    using the Python internal tool *pip*. The packages are all installed by whl files (obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/).

Finally, the entire setup is finished:

.. figure:: images/itom_install/setup10.jpg
    :scale: 100%
    :align: center
    
That's it:

.. figure:: images/itom_install/install21.png
    :width: 100%
    :align: center

This is |itom|!

