set(target_name itomWidgets)
            
project(${target_name})
    
cmake_minimum_required(VERSION 3.0.2)
 
option(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
option(BUILD_SHARED_LIBS "Build shared library." ON)
option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)
option(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
set(ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")


include("${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake")
INIT_ITOM_LIBRARY() #apply general project settings, CMake configurations, policies... for itom and its libraries

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/.. ${ITOM_SDK_DIR})

if(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE SHARED)
    
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    add_definitions(-DITOMWIDGETS_DLL -D_ITOMWIDGETS_DLL)
else(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif(UNIX)
endif(BUILD_ITOMLIBS_SHARED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE_QT(OFF UiTools Widgets Svg Xml Designer PrintSupport LinguistTools) #no automoc for Qt5
find_package(VisualLeakDetector QUIET)
find_package(OpenCV COMPONENTS core REQUIRED)


add_definitions(${QT_DEFINITIONS})
if(BUILD_UNICODE)
    add_definitions(-DUNICODE -D_UNICODE)
endif(BUILD_UNICODE)
add_definitions(-DITOMSHAREDDESIGNER)

if(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)
    add_definitions(-DVISUAL_LEAK_DETECTOR_CMAKE)
endif(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser
    ${CMAKE_CURRENT_SOURCE_DIR}/..  
    ${QT_QTCORE_INCLUDE_DIR}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Designer_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}   
    ${OpenCV_DIR}/include
)

link_directories(
    ${OpenCV_DIR}/lib
)

set(paramPropertyBrowser_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtbuttonpropertybrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtbuttonpropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qteditorfactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qteditorfactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtgroupboxpropertybrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtgroupboxpropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowserutils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowserutils_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertymanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertymanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qttreepropertybrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qttreepropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtvariantproperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtvariantproperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramIntWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramIntWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramDoubleWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramDoubleWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramCharWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramCharWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramStringWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramStringWidget.cpp
)

set(paramPropertyBrowser_MOC_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowserutils_p.h
)

set(itomWidgets_Factory_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/brushCreatorButtonFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/colorPickerButtonFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/fontButtonFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/sliderWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pathLineEditFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/penCreatorButtonFactory.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pythonLogWidgetFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/motorAxisControllerFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/statusLedFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/paramEditorFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/brushCreatorDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/penCreatorDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/popupWidget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/basePopupWidget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/menuComboBox_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoDObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoMarker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoPicker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotInfoShapes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pythonLogWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramEditorWidget.h
    #${CMAKE_CURRENT_SOURCE_DIR}/paramEditorModel.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/brushCreatorButton.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/penCreatorButton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/colorPickerButton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fontButton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/colorDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/checkableComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/comboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/searchBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/treeComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/menuComboBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/motorAxisController.h
    ${CMAKE_CURRENT_SOURCE_DIR}/statusLed.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramEditorWidget.h
)

# List of header that should go though moc
set(paramPropertyBrowser_HEADERS_MOC_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtbuttonpropertybrowser.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qteditorfactory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtgroupboxpropertybrowser.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowser.h
  #${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowserutils_p.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertymanager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qttreepropertybrowser.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtvariantproperty.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamFactory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramIntWidget.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramCharWidget.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramDoubleWidget.h
  ${CMAKE_CURRENT_SOURCE_DIR}/paramStringWidget.h
)

# List of cpp files that should go though moc
set(paramPropertyBrowser_CPP_MOC_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qteditorfactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertymanager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/itomParamFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qttreepropertybrowser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramIntWidget.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramCharWidget.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramDoubleWidget.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paramStringWidget.cpp
)


if(QT5_FOUND)
    QT5_WRAP_CPP(paramPropertyBrowser_MOC ${paramPropertyBrowser_MOC_SOURCES})
    QT5_WRAP_CPP(itomWidgets_HEADERS_MOC ${itomWidgets_HEADERS})
    QT5_WRAP_CPP(itomWidgets_Factory_HEADERS_MOC ${itomWidgets_Factory_HEADERS})
    #these header files need to be moc'ed, but the moc file should not be added to the project
    #since it is included in another source file that actually compiles the moc'ed file.
    QT5_GENERATE_MOCS(
        pathLineEdit.h)
        
    set(paramPropertyBrowser_MOC_CPP)
    foreach(file ${paramPropertyBrowser_HEADERS_MOC_SOURCES})
      get_filename_component(filename_we ${file} NAME_WE)
      set(output_file moc_${filename_we}.cpp)
      QT5_GENERATE_MOC(${file} ${output_file})
      LIST(APPEND paramPropertyBrowser_MOC_CPP ${output_file})
    endforeach()
    
    foreach(file ${paramPropertyBrowser_CPP_MOC_SOURCES})
      get_filename_component(filename_we ${file} NAME_WE)
      set(output_file ${filename_we}.moc)
      QT5_GENERATE_MOC(${file} ${output_file})
      LIST(APPEND paramPropertyBrowser_MOC_CPP ${output_file})
    endforeach()
else()
    message(SEND_ERROR "Only Qt5 is currently supported")
endif()

#append the separately moc'ed header to the list of headers
LIST(APPEND itomWidgets_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.h")

set(itomWidgets_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgets.qrc
)

set(itomWidgets_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/itomWidgetsFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/rangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/brushCreatorButtonFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/collapsibleGroupBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/colorPickerButtonFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/penCreatorButtonFactory.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleRangeSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSpinBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/doubleSliderFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/fontButtonFactory.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/pythonLogWidgetFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/menuComboBoxFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/motorAxisControllerFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/statusLedFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/paramEditorFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/checkableModelHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/completer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/brushCreatorButton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/brushCreatorDialog.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/colorPickerButton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fontButton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/collapsibleGroupBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSlider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleRangeWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/penCreatorButton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/penCreatorDialog.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/paramEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pythonLogWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/motorAxisController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statusLed.cpp
)

#If you want to use automatical metadata for dlls under windows use the following if-case.
if(MSVC)
    list(APPEND itomWidgets_SOURCES version.rc)
endif()


set(itomWidgets_UI
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/rangeWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/doubleRangeWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/sliderWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/motorAxisController.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/penCreatorDialog.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/brushCreatorDialog.ui
)



if(QT5_FOUND)
    QT5_WRAP_UI(itomWidgets_UI_MOC ${itomWidgets_UI})
endif()

file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")

if(QT5_FOUND)
    QT5_ADD_RESOURCES(itomWidgets_rcc_MOC ${itomWidgets_rcc})
endif()

#add filters to visual studio
ADD_SOURCE_GROUP(factories)
ADD_SOURCE_GROUP(helper)
ADD_SOURCE_GROUP(paramPropertyBrowser)
#end add filters to visual studio

add_library(${target_name} ${LIBRARY_TYPE} 
    ${itomWidgets_SOURCES} 
    ${itomWidgets_HEADERS} 
    ${itomWidgets_HEADERS_MOC} 
    ${itomWidgets_Factory_HEADERS} 
    ${itomWidgets_Factory_HEADERS_MOC} 
    ${itomWidgets_UI_MOC} 
    ${itomWidgets_rcc_MOC} 
    ${paramPropertyBrowser_SOURCES}
    ${paramPropertyBrowser_MOC}
    ${EXISTING_TRANSLATION_FILES})

if(QT5_FOUND)
    target_link_libraries(${target_name} ${VISUALLEAKDETECTOR_LIBRARIES} ${OpenCV_LIBS} ${QT5_LIBRARIES} itomCommonLib itomShapeLib itomCommonQtLib)
else()
    target_link_libraries(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTDESIGNER_LIBRARY} ${QT_QTSVG_LIBRARY} ${VISUALLEAKDETECTOR_LIBRARIES} ${OpenCV_LIBS} itomCommonLib itomShapeLib itomCommonQtLib)
endif()



#translation
set(FILES_TO_TRANSLATE ${itomWidgets_SOURCES} ${itomWidgets_HEADERS} ${itomWidgets_UI})
PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

#this is a dummy project to force some moc-files to be generated before they are directly included in some cpp files (qtpropertybrowser only)
add_custom_target(${target_name}GenerateMoc
  DEPENDS ${paramPropertyBrowser_MOC_CPP}
  )
add_dependencies(${target_name} ${target_name}GenerateMoc)

# COPY SECTION
#set(COPY_SOURCES "")
#set(COPY_DESTINATIONS "")
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/${target_name}_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)

if(WIN32)
    #For Windows, the resulting library is copied to the designer-folder, such that the widgets are listed in the widget library of QtDesigner
    ADD_DESIGNERLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)
else(WIN32)
   #Under linux, copying the library (like under Windows) will result in a strange double-linked-list error, since two libraries with the
   #same content will then be loaded by itom. Therefore, a symbolic link to the original library is created in the designer-folder.
    file(MAKE_DIRECTORY "${ITOM_APP_DIR}/designer")
    add_custom_command(TARGET ${target_name} POST_BUILD                                                # Adds a post-build event to this target
            COMMAND ${CMAKE_COMMAND} -E create_symlink                                                                    # which executes "cmake - E create_symlink..."
                "$<TARGET_LINKER_FILE:${target_name}>"                                                                                      # <--this is the old file
                "${ITOM_APP_DIR}/designer/$<TARGET_LINKER_FILE_NAME:${target_name}>"     # <--this is the new file
        )
endif(WIN32)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

foreach(headerFile ${itomWidgets_HEADERS}) 
    file(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itomWidgets FILES_MATCHING PATTERN "*.h")
endforeach(headerFile)