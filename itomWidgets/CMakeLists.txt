SET (target_name itomWidgets)
         
project(${target_name})

cmake_minimum_required(VERSION 2.8.9)

#CMAKE Policies
IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()
 
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/.. ${ITOM_SDK_DIR})

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMWIDGETS_DLL -D_ITOMWIDGETS_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE_QT(OFF UiTools Widgets Svg Xml Designer LinguistTools) #no automoc for Qt5
find_package(VisualLeakDetector QUIET)
find_package(OpenCV COMPONENTS core REQUIRED)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/..  
    ${QT_QTCORE_INCLUDE_DIR}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Designer_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}   
)

LINK_DIRECTORIES(
    ${OpenCV_DIR}/lib
)

set(itomWidgets_Factory_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/colorPickerButtonFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/sliderWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pathLineEditFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/popupWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/checkableComboBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/comboBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/searchBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/treeComboBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/menuComboBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoDObjectFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoMarkerFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoPickerFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoShapesFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/motorAxisControllerFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/popupWidget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/basePopupWidget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/menuComboBox_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoDObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoMarker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoPicker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoShapes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/valueProxy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/proxyStyle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/checkableModelHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/completer.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/popupWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/basePopupWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoDObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoMarker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoPicker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoShapes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/colorPickerButton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/colorDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/checkableComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/comboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/searchBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/treeComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/menuComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/motorAxisController.h
)


if (QT5_FOUND)
    QT5_WRAP_CPP(itomWidgets_HEADERS_MOC ${itomWidgets_HEADERS})
    QT5_WRAP_CPP(itomWidgets_Factory_HEADERS_MOC ${itomWidgets_Factory_HEADERS})
    #these header files need to be moc'ed, but the moc file should not be added to the project
    #since it is included in another source file that actually compiles the moc'ed file.
    QT5_GENERATE_MOCS(
        pathLineEdit.h)
else (QT5_FOUND)
    QT4_WRAP_CPP_ITOM(itomWidgets_Factory_HEADERS_MOC ${itomWidgets_Factory_HEADERS})
    QT4_WRAP_CPP_ITOM(itomWidgets_HEADERS_MOC ${itomWidgets_HEADERS})
    
    #these header files need to be moc'ed, but the moc file should not be added to the project
    #since it is included in another source file that actually compiles the moc'ed file.
    QT4_GENERATE_MOCS(
        pathLineEdit.h
    )
endif (QT5_FOUND)

#append the separately moc'ed header to the list of headers
LIST(APPEND itomWidgets_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.h")

set(itomWidgets_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgets.qrc
)

set(itomWidgets_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/colorPickerButtonFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/sliderWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pathLineEditFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/popupWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/checkableComboBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/comboBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/searchBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/treeComboBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoDObjectFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoMarkerFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoPickerFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/plotInfoShapesFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/menuComboBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/motorAxisControllerFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/checkableModelHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/completer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorPickerButton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/collapsibleGroupBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/popupWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basePopupWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/checkableComboBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/comboBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/searchBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/treeComboBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/menuComboBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valueProxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proxyStyle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoDObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoMarker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoPicker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoShapes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/motorAxisController.cpp
)

#If you want to use automatical metadata for dlls under windows use the following if-case.
if(MSVC)
    list(APPEND itomWidgets_SOURCES version.rc)
endif(MSVC)


set(itomWidgets_UI
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/rangeWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/doubleRangeWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/sliderWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/motorAxisController.ui
)

IF (QT5_FOUND)
    QT5_WRAP_UI(itomWidgets_UI_MOC ${itomWidgets_UI})
ELSE (QT5_FOUND)
    QT4_WRAP_UI_ITOM(itomWidgets_UI_MOC ${itomWidgets_UI})
ENDIF (QT5_FOUND)

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
ADD_SOURCE_GROUP(helper)
#end add filters to visual studio

ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${itomWidgets_SOURCES} ${itomWidgets_HEADERS} ${itomWidgets_HEADERS_MOC} ${itomWidgets_Factory_HEADERS} ${itomWidgets_Factory_HEADERS_MOC} ${itomWidgets_UI_MOC} ${itomWidgets_rcc_MOC} ${EXISTING_TRANSLATION_FILES})

if (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${VISUALLEAKDETECTOR_LIBRARIES} ${OpenCV_LIBS} itomCommonLib itomShapeLib itomCommonQtLib)
else (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTDESIGNER_LIBRARY} ${QT_QTSVG_LIBRARY} ${VISUALLEAKDETECTOR_LIBRARIES} ${OpenCV_LIBS} itomCommonLib itomShapeLib itomCommonQtLib)
endif (QT5_FOUND)

IF (QT5_FOUND)
    IF(CMAKE_VERSION VERSION_GREATER 3.0.0)
        #https://bugreports.qt.io/browse/QTBUG-39457
        cmake_policy(SET CMP0043 OLD)
    ENDIF()
    qt5_use_modules(${target_name} ${QT_COMPONENTS})
ENDIF (QT5_FOUND)

#translation
set (FILES_TO_TRANSLATE ${itomWidgets_SOURCES} ${itomWidgets_HEADERS} ${itomWidgets_UI})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
ADD_DESIGNERLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

FOREACH(headerFile ${itomWidgets_HEADERS}) 
    FILE(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itomWidgets FILES_MATCHING PATTERN "*.h")
ENDFOREACH(headerFile)