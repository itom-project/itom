SET (target_name qpropertyeditor)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 

if(CMAKE_HOST_WIN32)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		string (REPLACE "/DWIN32" "/D_WIN64" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) 
	else()
		#ok
	endif()
endif()
  
#compile as static library (always)
SET(LIBRARY_TYPE STATIC)
 
#if QPropertyEditor is static, add -fPIC as compiler flag for linux
IF(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF(UNIX)
 
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

find_package(Qt4 REQUIRED)

ADD_DEFINITIONS(${QT_DEFINITIONS})
add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

INCLUDE_DIRECTORIES( ${QT_QTCORE_INCLUDE_DIR}
	${QT_QTSVG_INCLUDE_DIR}
)

INCLUDE(${QT_USE_FILE})

set(qpropertyeditor_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/ColorCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/BooleanCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/EnumProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/Property.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QPropertyEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QPropertyModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QVariantDelegate.h
)

QT4_WRAP_CPP(qproperteditor_HEADERS_MOC ${qpropertyeditor_HEADERS})

set(qpropertyeditor_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/ColorCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/BooleanCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/EnumProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/Property.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QPropertyEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QPropertyModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QPropertyEditor/QVariantDelegate.cpp
)

add_library(${target_name} ${LIBRARY_TYPE} ${qpropertyeditor_SOURCES} ${qpropertyeditor_HEADERS} ${qproperteditor_HEADERS_MOC})

TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES})

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
SET(SOURCEFILE ${CMAKE_CURRENT_SOURCE_DIR}/../QPropertyEditor/src/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/ColorCombo.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/BooleanCombo.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/EnumProperty.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/Property.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QPropertyEditorWidget.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QPropertyModel.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
FILE(COPY ${SOURCEFILE}/QVariantDelegate.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_OUTPUTLIBRARY_TO_SDK_LIB(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)
