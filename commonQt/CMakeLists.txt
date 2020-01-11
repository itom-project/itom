set(target_name itomCommonQtLib)
project(${target_name}) 
    cmake_minimum_required(VERSION 2.8.9)
       
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
    add_definitions(-DITOMCOMMONQT_DLL -D_ITOMCOMMONQT_DLL)
else(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif(UNIX)

    add_definitions(-DITOM_CORE -D_ITOM_CORE) #necessary for not publishing the ITOM_API_FUNCS (in every dll or exe ITOM_API_FUNCS need to be
                                               #published with void** only one (!) time. if this lib is statically linked to itom, it is already
                                               #defined within itom and must not be defined in this lib but with extern void**. If this lib is
                                               #shared, it also needs to be defined there.
endif(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE_QT(ON Core Xml PrintSupport Widgets Svg Designer LinguistTools)
#FIND_PACKAGE_QT(ON Core Widgets)
find_package(OpenCV COMPONENTS core REQUIRED)

#if(BUILD_WITH_PCL)
#    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
#    add_definitions(-DUSEPCL -D_USEPCL)
#endif(BUILD_WITH_PCL)

if(BUILD_UNICODE)
    add_definitions(-DUNICODE -D_UNICODE)
endif(BUILD_UNICODE)

add_definitions(-DITOMCOMMONQT_MOC)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(BUILD_WITH_PCL)
    if(BUILD_ITOMLIBS_SHARED)
        set(POINTCLOUD_LIBRARIES pointcloud)
    else(BUILD_ITOMLIBS_SHARED)
        set(POINTCLOUD_LIBRARIES ${PCL_LIBRARIES} pointcloud)
    endif(BUILD_ITOMLIBS_SHARED)
else(BUILD_WITH_PCL)
    set(POINTCLOUD_LIBRARIES "")
endif(BUILD_WITH_PCL)

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/../shapes
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${CMAKE_CURRENT_BINARY_DIR}/../PointCloud
    ${OpenCV_DIR}/lib
)

set(itomCommonQt_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonQtVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterfaceVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/abstractAddInDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/abstractAddInConfigDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/pluginThreadCtrl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresPrimitives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/functionCancellationAndObserver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/abstractApiWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/fileUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/semVerVersion.h
)

set(itomCommonQt_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/addInGrabber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/addInInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/abstractAddInDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/abstractAddInConfigDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/helperCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/pluginThreadCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedFunctionsQt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedStructuresPrimitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/sharedStructuresQt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/functionCancellationAndObserver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/abstractApiWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/fileUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/semVerVersion.cpp
)

if(MSVC)
    list(APPEND itomCommonQt_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations END STEP 1

add_library(${target_name} ${LIBRARY_TYPE} ${itomCommonQt_SOURCES} ${itomCommonQt_HEADERS} ${itomCommonQt_HEADERS_MOC} ${EXISTING_TRANSLATION_FILES})

target_link_libraries(${target_name} ${QT_LIBRARIES} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${QT5_LIBRARIES} dataobject itomCommonLib itomShapeLib)



#translation
set(FILES_TO_TRANSLATE ${itomCommonQt_SOURCES} ${itomCommonQt_HEADERS})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/commonQtVersion.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/common)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/${target_name}_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)