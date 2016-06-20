.. include:: ../include/global.inc

.. _build-raspi:

Build on Raspberry Pi
======================

This section describes how |itom| and its plugins are built on a Raspberry Pi with the Raspbian operating system (based on Debian).
For general information about building |itom| under linux, see the :ref:`build instructions for linux <build-linux>`.

In the following, all required steps are indicated to get the dependencies, get the sources and compile itom by the command line.
The folder structure is chosen to be equal than the one from the linux instructions, however itom will only be build in a release version.

Please execute the following commands in the command line to get the dependencies for |itom| (comments after the hash-tag should not be copied to the command line):

.. code-block:: bash
    
    sudo apt-get install cmake cmake-gui git
    sudo apt-get install python3 python3-dev python3-numpy python3-pip
    sudo apt-get install libqt5webkit5 libqt5webkit5-dev libqt5widgets5 libqt5xml5 libqt5svg5 libqt5svg5-dev libqt5gui5 libqt5designer5 libqt5concurrent5
    sudo apt-get install libqt5scintilla2-dev
    sudo apt-get install libqt5scintilla2-11  #(usually, this comes with libqt5scintilla2-dev, the suffix -11 might be different for different distributions)
    sudo apt-get install qttools5-dev-tools qttools5-dev
    sudo apt-get update && sudo apt-get install build-essential
    sudo apt-get install libopencv-dev
    sudo apt-get install libv4l-dev #this is optional to get the video for linux drivers
    sudo apt-get install xsdcxx libxerces-c3.1 libxerces-c-dev #this is optional to being able to compile the x3p plugin

If you want to compile |itom| with support from the Point Cloud Library, also get the following packages:

.. code-block:: bash
    
    sudo apt-get install libpcl-dev libproj-dev

Now, change to the base directory, where the sources and builds of itom and its plugins should be placed. The following commands are not executed
with super-user rights; prepend *sudo* if this is required:

.. code-block:: bash
    
    mkdir itom
    cd itom
    mkdir sources
    cd sources
    git clone https://bitbucket.org/itom/itom.git #if there is warning due to a missing SSL certificate, see the hints below
    git clone https://bitbucket.org/itom/plugins.git
    git clone https://bitbucket.org/itom/designerPlugins.git
    cd ..
    mkdir build
    cd build
    mkdir itom
    cd itom
    cmake -G "Unix Makefiles" -DBUILD_WITH_PCL=OFF ../../sources/itom #If PCL-support should be enabled, replace OFF by ON
    make
    cd ..
    mkdir designerPlugins
    cd designerPlugins
    cmake -G "Unix Makefiles" -DBUILD_WITH_PCL=OFF -DITOM_SDK_DIR=../itom/SDK ../../sources/designerPlugins #If PCL-support should be enabled, replace OFF by ON
    make
    cd ..
    mkdir plugins
    cd plugins
    cmake -G "Unix Makefiles" -DBUILD_WITH_PCL=OFF -DITOM_SDK_DIR=../itom/SDK ../../sources/plugins #If PCL-support should be enabled, replace OFF by ON

Hints
--------------

If there is an SSL certificate error in the git clone process, try to add::
    
    git -c http.sslVerify=false clone ...

to the *clone* command

Execute
--------------

Run the file **qitom** in the *build/itom* directory. Please give itom the rights to write files in the directory, e.g. the settings.ini file.

.. figure:: images/screenshot_raspberry_pi.png
    :scale: 100%
    :align: center

Camera of Raspberry Pi
-------------------------

If you want to access the optional camera of the Raspberry Pi, you can do this using the **v4l2** or **OpenCVGrabber** plugins. Before doing this, you have
to start the v4l2 camera driver process (every time the raspi is started, or place it in the autostart script)::
    
    sudo modprobe bcm2835-v4l2
