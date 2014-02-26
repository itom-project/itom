SET (target_name itomCommonQtLib)
project(${target_name}) 

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)


SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMCOMMONQT_DLL -D_ITOMCOMMONQT_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
    
    ADD_DEFINITIONS(-DITOM_CORE -D_ITOM_CORE) #necessary for not publishing the ITOM_API_FUNCS (in every dll or exe ITOM_API_FUNCS need to be
                                               #published with void** only one (!) time. if this lib is statically linked to itom, it is already
                                               #defined within itom and must not be defined in this lib but with extern void**. If this lib is
                                               #shared, it also needs to be defined there.
    
ENDIF(BUILD_ITOMLIBS_SHARED)
    

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (QT5_FOUND FALSE)
if (CMAKE_VERSION VERSION_GREATER 2.8.7)
    if (POLICY CMP0020)
        cmake_policy(SET CMP0020 NEW)
    endif (POLICY CMP0020)
    
    if (WIN32)
        find_package(WindowsSDK REQUIRED)
        set (CMAKE_PREFIX_PATH "${WINDOWSSDK_PREFERRED_DIR}/Lib/")
        set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${WINDOWSSDK_PREFERRED_DIR}/Lib/)
    endif (WIN32)
    find_package(Qt5 COMPONENTS Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools QUIET)

    if (NOT ${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
        set(QT5_FOUND TRUE)
        set(CMAKE_AUTOMOC ON)
    endif (NOT ${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
endif (CMAKE_VERSION VERSION_GREATER 2.8.7)
if (NOT QT5_FOUND)
    find_package(Qt4 REQUIRED)
endif (NOT QT5_FOUND)


find_package(OpenCV REQUIRED COMPONENTS core)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

ADD_DEFINITIONS(-DITOMCOMMONQT_MOC)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)


INCLUDE_DIRECTORIES( 
#    ${QT_QTCORE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${OpenCV_DIR}/lib
)

IF (NOT QT5_FOUND)
    INCLUDE(${QT_USE_FILE})
ENDIF (NOT QT5_FOUND)

set(itomCommonQt_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonQtVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperActuator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresPrimitives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractItomDesignerPlugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractNode.h
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/drawItem.h
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/multiPointPickerMachine.h
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/userInteractionPlotPicker.h
)

IF(NOT QT5_FOUND)
    QT4_WRAP_CPP_ITOM(itomCommonQt_HEADERS_MOC ${itomCommonQt_HEADERS})
ENDIF(NOT QT5_FOUND)

set(itomCommonQt_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/addInGrabber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/addInInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/helperActuator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/helperGrabber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/helperCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedFunctionsQt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedStructuresPrimitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedStructuresQt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/AbstractDObjFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/AbstractFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/AbstractNode.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/drawItem.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/multiPointPickerMachine.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/../plot/sources/userInteractionPlotPicker.cpp
)

if(MSVC)
	list(APPEND itomCommonQt_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif(MSVC)

add_library(${target_name} ${LIBRARY_TYPE} ${itomCommonQt_SOURCES} ${itomCommonQt_HEADERS} ${itomCommonQt_HEADERS_MOC})

IF (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${OpenCV_LIBS} dataobject qpropertyeditor itomCommonLib)
ELSE (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTXML_LIBRARY} ${OpenCV_LIBS} dataobject qpropertyeditor itomCommonLib)
ENDIF (QT5_FOUND)

IF (QT5_FOUND)
    qt5_use_modules(${target_name} Widgets Gui Xml)
ENDIF (QT5_FOUND)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)
