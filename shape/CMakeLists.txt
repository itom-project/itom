set(target_name itomShapeLib)
project(${target_name}) 
           
cmake_minimum_required(VERSION 3.0.2)

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
option(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
option(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
option(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
      
include("../ItomBuildMacros.cmake")
INIT_ITOM_LIBRARY() #apply general project settings, CMake configurations, policies... for itom and its libraries

if(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE SHARED)
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    add_definitions(-DITOMSHAPE_DLL -D_ITOMSHAPE_DLL)
else(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE STATIC)
    
    #if itomShape is static, add -fPIC as compiler flag for linux
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif(UNIX)

    add_definitions(-DITOM_CORE -D_ITOM_CORE) #necessary for not publishing the ITOM_API_FUNCS (in every dll or exe ITOM_API_FUNCS need to be
                                               #published with void** only one (!) time. if this lib is statically linked to itom, it is already
                                               #defined within itom and must not be defined in this lib but with extern void**. If this lib is
                                               #shared, it also needs to be defined there.
endif(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#FIND_PACKAGE_QT(ON Core Xml PrintSupport Svg Designer LinguistTools)
FIND_PACKAGE_QT(ON Core Designer LinguistTools)
find_package(OpenCV COMPONENTS core imgproc REQUIRED)

if(BUILD_UNICODE)
    add_definitions(-DUNICODE -D_UNICODE)
endif(BUILD_UNICODE)

add_definitions(-DITOMSHAPE_MOC)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
)

link_directories(
    ${OpenCV_DIR}/lib
)

set(itomShape_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shapeDObject.h
)

set(itomShape_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shapeDObject.cpp
)

if(MSVC)
    list(APPEND itomShape_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations END STEP 1

add_library(${target_name} ${LIBRARY_TYPE} ${itomShape_SOURCES} ${itomShape_HEADERS} ${itomShape_HEADERS_MOC} ${EXISTING_TRANSLATION_FILES})

target_link_libraries(${target_name} ${QT_LIBRARIES} ${OpenCV_LIBS} ${QT5_LIBRARIES} dataobject itomCommonLib)



#translation
set(FILES_TO_TRANSLATE ${itomShape_SOURCES} ${itomShape_HEADERS})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/itomShapeLib_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

foreach(headerFile ${itomShape_HEADERS}) 
    file(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/shape FILES_MATCHING PATTERN "*.h")
endforeach(headerFile)