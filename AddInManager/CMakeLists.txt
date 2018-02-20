SET (target_name addinmanager)
project(${target_name}) 

cmake_minimum_required(VERSION 2.8)
  
IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()

if (POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW) #raise an CMake error if an imported target, containing ::, could not be found
ENDIF (POLICY CMP0028)
   
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
    
IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DADDINMGR_DLL -D_ADDINMGR_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if dataObject is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

IF(BUILD_WITH_PCL)
	#find_package(ITOM_SDK COMPONENTS dataobject pointcloud itomCommonLib itomCommonQtLib REQUIRED)
	find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
	ADD_DEFINITIONS(-DUSEPCL)
ELSE(BUILD_WITH_PCL)
	#find_package(ITOM_SDK COMPONENTS dataobject itomCommonLib itomCommonQtLib REQUIRED)
ENDIF(BUILD_WITH_PCL)

#at first, get the available Qt version
FIND_PACKAGE_QT(ON Core)
#requires Qt5 and Qt5WebEngine in order to built a built-in help viewer, since QtAssistant only has a limited set of rendering options since Qt5.6.
FIND_PACKAGE_QT(ON Core Widgets Xml Designer Svg PrintSupport LinguistTools)

find_package(OpenCV COMPONENTS core REQUIRED)
find_package(VisualLeakDetector QUIET)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)


INCLUDE_DIRECTORIES( 
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

LINK_DIRECTORIES(
#    ${OpenCV_DIR}/lib
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
endif(MSVC)

#################################################################
# Qt related pre-processing of the files above
# (These methods create the moc, rcc and uic process.)
#################################################################
if (QT5_FOUND)
    #if automoc if OFF, you also need to call QT5_WRAP_CPP here
    #QT5_WRAP_UI(plugin_UI_MOC ${lib_UI})
    #QT5_ADD_RESOURCES(plugin_RCC_MOC ${lib_RCC})
else (QT5_FOUND)
    QT4_WRAP_CPP_ITOM(lib_HEADERS_MOC ${lib_HEADERS})
    #QT4_WRAP_UI_ITOM(lib_UI_MOC ${lib_UI})
    #QT4_ADD_RESOURCES(plugin_RCC_MOC ${lib_RCC})
endif (QT5_FOUND)

file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")

ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${lib_SOURCES} ${lib_HEADERS} ${lib_HEADERS_MOC} ${lib_UI_MOC} ${lib_RCC_MOC} ${EXISTING_TRANSLATION_FILES})

IF(BUILD_ITOMLIBS_SHARED)
    TARGET_LINK_LIBRARIES(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${QT5_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} ${QT_LIBRARIES} dataobject itomCommonLib itomCommonQtLib )
ENDIF(BUILD_ITOMLIBS_SHARED)
IF (QT5_FOUND AND CMAKE_VERSION VERSION_LESS 3.0.2)
    qt5_use_modules(${target_name} ${QT_COMPONENTS}) #special command for Qt5
ENDIF (QT5_FOUND AND CMAKE_VERSION VERSION_LESS 3.0.2)

#translation
set (FILES_TO_TRANSLATE ${lib_SOURCES} ${lib_HEADERS} ${lib_UI})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/addInMgrDefines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)
# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/addInManager.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)
# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pluginModel.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)
# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/paramHelper.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)
# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/algoInterfaceValidator.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)
# FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerVersion.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/AddInManager)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/${target_name}_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(addinmanager COPY_SOURCES COPY_DESTINATIONS)
