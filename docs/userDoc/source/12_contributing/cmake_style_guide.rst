.. _cmake-style-guide:

CMake Style Guide
************************

General
============

To put in in one sentence: be as careful when writing the CMake files as when you are writing C++ code.


Indentation
============

Indent all code correctly, i.e. the body of

    * if/else/endif
    * foreach/endforeach
    * while/endwhile
    * macro/endmacro
    * function/endfunction

Use spaces for indenting, 4 spaces preferably. Use the same amount of spaces for indenting as is used in the rest of the file. Do not use tabs.

Naming
==========

**Functions**: *lower_case* name. Ex:

.. code-block:: cmake

    do_something(...)

**Local variables**: *lower_case* name. Local variables are used exclusively inside the file that contained them, and their values were simply passed as parameters to CMake functions. Ex:

.. code-block:: cmake

    set(some_variable "...")

**Global variables**: *UPPER_CASE* name. Global variables (can also be called "export variables") are intended for exporting up/down-stream via the environment variable mechanism. Ex:

.. code-block:: cmake

    set(SOME_VARIABLE "..." CACHE ...)

**Control statements**: *lower_case* name without repeat the condition in the closing brackets. Ex:

.. code-block:: cmake

    if(condition)
      ...
    else() # not repeat condition
      ...
    endif() # not repeat condition

**Operators**: *UPPER_CASE* name. Ex:

.. code-block:: cmake

    if(condition STREQUAL "")

**Directives and/or extra options**:  *UPPER_CASE* name. Ex:

.. code-block:: cmake

    do_something(... USE_THIS)
    file(COPY ...)

End commands
===============

To make the code easier to read, use empty commands for endforeach(), endif(), endfunction(), endmacro() and endwhile(). Also, use empty else() commands.

For example, do this:

.. code-block:: cmake

    if(FOOVAR)
       some_command(...)
    else()
       another_command(...)
    endif()

and not this:

.. code-block:: cmake

    if(BARVAR)
       some_other_command(...)
    endif(BARVAR)

Examples
================

An real-world example:

.. code-block:: cmake

    function(set_platform system_name)
      if(${system_name} MATCHES "Darwin")
        set(PLATFORM "darwin")
      elseif(${system_name} MATCHES "Linux")
        set(PLATFORM "linux")
      else()
        set(PLATFORM "")
      endif()
    endfunction()

    cmake_minimum_required(VERSION 3.0)
    set_platform(${CMAKE_SYSTEM_NAME})


References
==================

This style guide is mainly taken from

* NuPic (http://nupic.docs.numenta.org/1.0.2/contributing/cmake-style-guide.html)
* KDE (https://community.kde.org/Policies/CMake_Coding_Style)
