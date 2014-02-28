SET (target_name itomWidgets)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${ITOM_SDK_DIR})

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMWIDGETS_SHARED -D_ITOMWIDGETS_SHARED)
    ADD_DEFINITIONS(-DITOMWIDGETS_DLL -D_ITOMWIDGETS_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (QT5_FOUND FALSE)
if (CMAKE_VERSION VERSION_GREATER 2.8.7)
    if (POLICY CMP0020)
        cmake_policy(SET CMP0020 NEW)
    endif (POLICY CMP0020)
    
    find_package(Qt5 COMPONENTS Widgets UiTools PrintSupport Network Sql Xml LinguistTools QUIET)

    if (NOT ${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
        set(QT5_FOUND TRUE)
        set(CMAKE_AUTOMOC ON)
    endif (NOT ${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
endif (CMAKE_VERSION VERSION_GREATER 2.8.7)
if (NOT QT5_FOUND)
    find_package(Qt4 REQUIRED)
endif (NOT QT5_FOUND)

find_package(VisualLeakDetector QUIET)

include("${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake")

ADD_DEFINITIONS(${QT_DEFINITIONS})
IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)
ADD_DEFINITIONS(-DCMAKE -DITOMSHAREDDESIGNER)

IF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)
    ADD_DEFINITIONS(-DVISUAL_LEAK_DETECTOR_CMAKE)
ENDIF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
)


if (NOT QT5_FOUND)
    INCLUDE(${QT_USE_FILE})
endif (NOT QT5_FOUND)

set(itomWidgets_Factory_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/sliderWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pathLineEditFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/valueProxy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/proxyStyle.h
)

# the following header files are finally copied into SDK/itomWidgets
set(itomWidgets_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonWidgets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeSlider.h
    ${CMAKE_CURRENT_SOURCE_DIR}/collapsibleGroupBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeSlider.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSlider.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidget.h
)

#this macro only generates the moc-file but does not compile it, since it is included in another source file.
#this comes from the ctkCommon project
macro(QT4_GENERATE_MOCS)
  foreach(file ${ARGN})
    set(moc_file moc_${file})
    QT4_GENERATE_MOC(${file} ${moc_file})

    get_filename_component(source_name ${file} NAME_WE)
    get_filename_component(source_ext ${file} EXT)
    if(${source_ext} MATCHES "\\.[hH]")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cpp)
        set(source_ext .cpp)
      elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cxx)
        set(source_ext .cxx)
      endif()
    endif()
    set_property(SOURCE ${source_name}${source_ext} APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${moc_file})
  endforeach()
endmacro()

QT4_GENERATE_MOCS(
    pathLineEdit.h
)

if (NOT QT5_FOUND)
    QT4_WRAP_CPP_ITOM(itomWidgets_Factory_HEADERS_MOC ${itomWidgets_Factory_HEADERS})
    QT4_WRAP_CPP_ITOM(itomWidgets_HEADERS_MOC ${itomWidgets_HEADERS})
endif (NOT QT5_FOUND)

set(itomWidgets_HEADERS ${itomWidets_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.h)

set(itomWidgets_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgets.qrc
)

set(itomWidgets_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/sliderWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pathLineEditFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/collapsibleGroupBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valueProxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proxyStyle.cpp
)

#If you want to use automatical metadata for dlls under windows use the following if-case.
if(MSVC)
    list(APPEND itomWidgets_SOURCES version.rc)
endif(MSVC)

#handle translations BEGIN STEP 1
set (FILES_TO_TRANSLATE ${itomWidgets_SOURCES} ${itomWidgets_HEADERS} ${itomWidgets_Factory_HEADERS} ${designer_ui})
set (TRANSLATION_OUTPUT_FILES)
set (TRANSLATIONS_FILES)

file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations END STEP 1

if (QT5_FOUND)
    QT5_ADD_RESOURCES(itomWidgets_rcc_MOC ${itomWidgets_rcc})
else (QT5_FOUND)
    QT4_ADD_RESOURCES(itomWidgets_rcc_MOC ${itomWidgets_rcc})
endif (QT5_FOUND)

#add filters to visual studio
ADD_SOURCE_GROUP(factories)
#end add filters to visual studio

ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${itomWidgets_SOURCES} ${itomWidgets_HEADERS} ${itomWidgets_HEADERS_MOC} ${itomWidgets_Factory_HEADERS} ${itomWidgets_Factory_HEADERS_MOC} ${itomWidgets_ui_MOC} ${itomWidgets_rcc_MOC} ${EXISTING_TRANSLATION_FILES})

if (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${VISUALLEAKDETECTOR_LIBRARIES} itomCommonLib)
    qt5_use_modules(${target_name} UiTools Widgets Svg Designer)
else (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTDESIGNER_LIBRARY} ${QT_QTSVG_LIBRARY} ${VISUALLEAKDETECTOR_LIBRARIES} itomCommonLib)
endif (QT5_FOUND)

add_dependencies(${target_name} itomWidgets_SPECIAL_MOC)

#handle translations BEGIN STEP 2
if (UPDATE_TRANSLATIONS)
    if (QT5_FOUND)
        QT5_CREATE_TRANSLATION(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
    else (QT5_FOUND)
        QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
    endif (QT5_FOUND)
    
    add_custom_target (_${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
    add_dependencies(${target_name} _${target_name}_translation)
    
    if (QT5_FOUND)
        QT5_ADD_TRANSLATION(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${TRANSLATIONS_FILES})
    else (QT5_FOUND)
        QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${TRANSLATIONS_FILES})
    endif (QT5_FOUND)
else (UPDATE_TRANSLATIONS)
    if (QT5_FOUND)
        QT5_ADD_TRANSLATION(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${EXISTING_TRANSLATION_FILES})
    else (QT5_FOUND)
        QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${EXISTING_TRANSLATION_FILES})
    endif (QT5_FOUND)
endif (UPDATE_TRANSLATIONS)
#handle translations END STEP 2

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_DESIGNERLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)
#ADD_DESIGNER_QM_FILES_TO_COPY_LIST(QM_FILES COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

FOREACH(headerFile ${itomWidgets_HEADERS}) 
    FILE(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itomWidgets FILES_MATCHING PATTERN "*.h")
ENDFOREACH(headerFile)
