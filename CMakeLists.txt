project(itom)

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
SET (ITOM_APP_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "base path to itom")
SET (ITOM_SDK_DIR ${CMAKE_CURRENT_BINARY_DIR}/SDK CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

ADD_SUBDIRECTORY(DataObject)
ADD_SUBDIRECTORY(PointCloud)
ADD_SUBDIRECTORY(QPropertyEditor)
ADD_SUBDIRECTORY(Qitom)

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/itom_doxygen.dox.in ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/itom_doxygen.dox )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/create_doc.py.in ${CMAKE_CURRENT_BINARY_DIR}/docs/userDoc/create_doc.py )

IF(MSVC10)
    IF(CMAKE_CL_64)
        configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/compile_debug64.bat ${CMAKE_CURRENT_BINARY_DIR}/compile_debug.bat COPYONLY)
        configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/compile_release64.bat ${CMAKE_CURRENT_BINARY_DIR}/compile_release.bat COPYONLY)
    ELSE(CMAKE_CL_64)
        configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/compile_debug32.bat ${CMAKE_CURRENT_BINARY_DIR}/compile_debug.bat COPYONLY)
        configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/compile_release32.bat ${CMAKE_CURRENT_BINARY_DIR}/compile_release.bat COPYONLY)
    ENDIF(CMAKE_CL_64)
ENDIF(MSVC10)

#ADD_SUBDIRECTORY(designerPluginSource)
#ADD_SUBDIRECTORY(pluginSource)