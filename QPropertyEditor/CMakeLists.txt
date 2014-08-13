SET (target_name qpropertyeditor)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
  
IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)
 
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

FIND_PACKAGE_QT(ON Core PrintSupport Widgets)

add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)


set(qpropertyeditor_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/ColorCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/BooleanCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FontEditor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EnumProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Property.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QVariantDelegate.h
)

IF (QT5_FOUND)
#    QT5_WRAP_CPP(qproperteditor_HEADERS_MOC ${qpropertyeditor_HEADERS})
ELSE (QT5_FOUND)
    QT4_WRAP_CPP_ITOM(qproperteditor_HEADERS_MOC ${qpropertyeditor_HEADERS})
ENDIF (QT5_FOUND)

set(qpropertyeditor_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ColorCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BooleanCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FontEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EnumProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Property.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QVariantDelegate.cpp
)

add_library(${target_name} ${LIBRARY_TYPE} ${qpropertyeditor_SOURCES} ${qpropertyeditor_HEADERS} ${qproperteditor_HEADERS_MOC})

TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES})
IF (QT5_FOUND)
    qt5_use_modules(${target_name} ${QT_COMPONENTS})
ENDIF (QT5_FOUND)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
SET(SOURCEFILE ${CMAKE_CURRENT_SOURCE_DIR}/../QPropertyEditor)
FILE(COPY ${SOURCEFILE}/ColorCombo.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/BooleanCombo.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/FontEditor.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/EnumProperty.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/Property.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QPropertyEditorWidget.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QPropertyModel.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QVariantDelegate.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
#ADD_OUTPUTLIBRARY_TO_SDK_LIB(${target_name} COPY_SOURCES COPY_DESTINATIONS)
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)