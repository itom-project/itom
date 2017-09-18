SET (target_name itomShapeLib)
project(${target_name}) 
         
cmake_minimum_required(VERSION 2.8.9)
   
#CMAKE Policies
IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()
    
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
 
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
      
include("../ItomBuildMacros.cmake")

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMSHAPE_DLL -D_ITOMSHAPE_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomShape is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)

    ADD_DEFINITIONS(-DITOM_CORE -D_ITOM_CORE) #necessary for not publishing the ITOM_API_FUNCS (in every dll or exe ITOM_API_FUNCS need to be
                                               #published with void** only one (!) time. if this lib is statically linked to itom, it is already
                                               #defined within itom and must not be defined in this lib but with extern void**. If this lib is
                                               #shared, it also needs to be defined there.
ENDIF(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#FIND_PACKAGE_QT(ON Core Xml PrintSupport Svg Designer LinguistTools)
FIND_PACKAGE_QT(ON Core Designer)
find_package(OpenCV COMPONENTS core imgproc REQUIRED)

#IF(BUILD_WITH_PCL)
#    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
#    ADD_DEFINITIONS(-DUSEPCL -D_USEPCL)
#ENDIF(BUILD_WITH_PCL)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

ADD_DEFINITIONS(-DITOMSHAPE_MOC)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES( 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
)

LINK_DIRECTORIES(
    ${OpenCV_DIR}/lib
)

set(itomShape_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shapeDObject.h
)

IF(NOT QT5_FOUND)
    QT4_WRAP_CPP_ITOM(itomShape_HEADERS_MOC ${itomShape_HEADERS})
ENDIF(NOT QT5_FOUND)

set(itomShape_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shapeDObject.cpp
)

if(MSVC)
    list(APPEND itomShape_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif(MSVC)

file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations END STEP 1

add_library(${target_name} ${LIBRARY_TYPE} ${itomShape_SOURCES} ${itomShape_HEADERS} ${itomShape_HEADERS_MOC} ${EXISTING_TRANSLATION_FILES})

TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${OpenCV_LIBS} ${QT5_LIBRARIES} dataobject itomCommonLib)

IF (QT5_FOUND AND CMAKE_VERSION VERSION_LESS 3.0.2)
    qt5_use_modules(${target_name} ${QT_COMPONENTS})
ENDIF (QT5_FOUND AND CMAKE_VERSION VERSION_LESS 3.0.2)

#translation
set (FILES_TO_TRANSLATE ${itomShape_SOURCES} ${itomShape_HEADERS})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/itomShapeLib_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

FOREACH(headerFile ${itomShape_HEADERS}) 
    FILE(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/shape FILES_MATCHING PATTERN "*.h")
ENDFOREACH(headerFile)