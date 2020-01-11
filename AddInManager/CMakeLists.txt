set(target_name addinmanager)
project(${target_name}) 

cmake_minimum_required(VERSION 3.0.2)
   
option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
option(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
option(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)

include("../ItomBuildMacros.cmake")
INIT_ITOM_LIBRARY() #apply general project settings, CMake configurations, policies... for itom and its libraries
    
if(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE SHARED)
    
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    add_definitions(-DADDINMGR_DLL -D_ADDINMGR_DLL)
else()
    set(LIBRARY_TYPE STATIC)
    
    #if dataObject is static, add -fPIC as compiler flag for linux
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif(UNIX)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)



if(BUILD_WITH_PCL)
    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
    add_definitions(-DUSEPCL)
endif()

#at first, get the available Qt version
FIND_PACKAGE_QT(ON Core)
#requires Qt5 and Qt5WebEngine in order to built a built-in help viewer, since QtAssistant only has a limited set of rendering options since Qt5.6.
FIND_PACKAGE_QT(ON Core Widgets Xml Designer Svg PrintSupport LinguistTools)

find_package(OpenCV COMPONENTS core REQUIRED)
find_package(VisualLeakDetector QUIET)

if(BUILD_UNICODE)
    add_definitions(-DUNICODE -D_UNICODE)
endif(BUILD_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()


include_directories( 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
#    ${ITOM_SDK_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${OpenCV_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
    ../common
    ../DataObject
    ../PointCloud
)

set(lib_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/addInMgrDefines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/algoInterfaceValidator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/apiFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerPrivate.h
)

set(lib_SOURCES

    ${CMAKE_CURRENT_SOURCE_DIR}/pluginModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algoInterfaceValidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/apiFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerPrivate.cpp
)

if(MSVC)
    list(APPEND lib_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")

add_library(${target_name} ${LIBRARY_TYPE} ${lib_SOURCES} ${lib_HEADERS} ${lib_HEADERS_MOC} ${lib_UI_MOC} ${lib_RCC_MOC} ${EXISTING_TRANSLATION_FILES})

if(BUILD_ITOMLIBS_SHARED)
    target_link_libraries(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${QT5_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} ${QT_LIBRARIES} dataobject itomCommonLib itomCommonQtLib )
endif()

#translation
set(FILES_TO_TRANSLATE ${lib_SOURCES} ${lib_HEADERS} ${lib_UI})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/${target_name}_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(addinmanager COPY_SOURCES COPY_DESTINATIONS)
