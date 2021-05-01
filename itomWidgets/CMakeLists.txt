# - itom software
# URL: http://www.uni-stuttgart.de/ito
# Copyright (C) 2020, Institut fuer Technische Optik (ITO),
# Universitaet Stuttgart, Germany
#
# This file is part of itom and its software development toolkit (SDK).
#
# itom is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public Licence as published by
# the Free Software Foundation; either version 2 of the Licence, or (at
# your option) any later version.
#
# In addition, as a special exception, the Institut fuer Technische
# Optik (ITO) gives you certain additional rights.
# These rights are described in the ITO LGPL Exception version 1.0,
# which can be found in the file LGPL_EXCEPTION.txt in this package.
#
# itom is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
# General Public Licence for more details.
#
#
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.1)
  
set(target_name itomWidgets)
project(${target_name})

message(STATUS "--------------- LIBRARY ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add 3rd party packages
itom_find_package_qt(OFF #no automoc for Qt5, aber unten wird er angeschalten?! ?!?!?
        UiTools 
        Widgets 
        Svg 
        Xml 
        Designer 
        PrintSupport 
        LinguistTools
        ) 
        
find_package(VisualLeakDetector QUIET)
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DITOMWIDGETS_DLL -D_ITOMWIDGETS_DLL)
add_definitions(-DITOMSHAREDDESIGNER)

#check https://blog.initiali.se/posts/building-qt-designer-plugins-for-your-custom-qt-widgets/
# if not included like this an "undefined interface error will be risen from Qt (moc?!)
INCLUDE_DIRECTORIES(
    ${Qt5Designer_INCLUDE_DIRS}
    ${ITOM_SDK_INCLUDE_DIR}
)

# kann ein target_include_directories werden
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${Qt5Widgets_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
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

# List of header that should go through moc
set(paramPropertyBrowser_HEADERS_MOC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtbuttonpropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qteditorfactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtgroupboxpropertybrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramPropertyBrowser/qtpropertybrowser.h
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

# List of cpp files that should go through moc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/dataObjectTableFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/dataObjectMetaTableFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectDelegate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectModel.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectTable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectMetaTable.h
)

QT5_WRAP_CPP(paramPropertyBrowser_MOC ${paramPropertyBrowser_MOC_SOURCES})
QT5_WRAP_CPP(itomWidgets_HEADERS_MOC ${itomWidgets_HEADERS})
QT5_WRAP_CPP(itomWidgets_Factory_HEADERS_MOC ${itomWidgets_Factory_HEADERS})

#these header files need to be moc'ed, but the moc file should not be added to the project
#since it is included in another source file that actually compiles the moc'ed file.
itom_qt_generate_mocs(
    pathLineEdit.h
    )

# some moc files are directly #included in other cpp files.
# Therefore, the exact filename of the moc file need to be known,
# which can only be done by manually processing them by QT5_GENERATE_MOC.
set(paramPropertyBrowser_MOC_CPP)

foreach(filename ${paramPropertyBrowser_HEADERS_MOC_SOURCES})
    get_filename_component(filename_we ${filename} NAME_WE)
    set(output_file moc_${filename_we}.cpp) #HEEEERE CHECK THIS LINE!!!
    QT5_GENERATE_MOC(${filename} ${output_file})
    list(APPEND paramPropertyBrowser_MOC_CPP ${output_file})
endforeach()

foreach(filename ${paramPropertyBrowser_CPP_MOC_SOURCES})
    get_filename_component(filename_we ${filename} NAME_WE)
    set(output_file ${filename_we}.moc)#it differs from the above one
    QT5_GENERATE_MOC(${filename} ${output_file})
    list(APPEND paramPropertyBrowser_MOC_CPP ${output_file})
endforeach()


#append the separately moc'ed header to the list of headers
list(APPEND itomWidgets_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/pathLineEdit.h")

set(itomWidgets_RCC
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
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/dataObjectTableFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factories/dataObjectMetaTableFactory.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectDelegate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectTable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectMetaTable.cpp
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

QT5_WRAP_UI(itomWidgets_UI_MOC ${itomWidgets_UI})
QT5_ADD_RESOURCES(itomWidgets_RCC_MOC ${itomWidgets_RCC})

#add filters to visual studio
itom_add_source_group(factories)
itom_add_source_group(helper)
itom_add_source_group(paramPropertyBrowser)
#end add filters to visual studio

add_library(${target_name} SHARED
    ${itomWidgets_SOURCES} 
    ${itomWidgets_HEADERS} 
    ${itomWidgets_HEADERS_MOC} 
    ${itomWidgets_Factory_HEADERS} 
    ${itomWidgets_Factory_HEADERS_MOC} 
    ${itomWidgets_UI_MOC} 
    ${itomWidgets_RCC_MOC} 
    ${paramPropertyBrowser_SOURCES}
    ${paramPropertyBrowser_MOC})

target_link_libraries(${target_name} 
    ${VISUALLEAKDETECTOR_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${QT5_LIBRARIES} 
    itomCommonLib 
    itomShapeLib 
    itomCommonQtLib
    )

# Qt: enable all automoc, autouic and autorcc. 
# Autouic file will be disabled for all files that are 
# processed manually by itom_qt_wrap_ui.
set_target_properties(${target_name} PROPERTIES 
        AUTOMOC OFF 
        AUTORCC OFF 
        AUTOUIC OFF
        )

#translation
itom_library_translation(QM_FILES TARGET ${target_name} 
    FILES_TO_TRANSLATE 
    ${itomWidgets_SOURCES} 
    ${itomWidgets_HEADERS} 
    ${itomWidgets_UI}
    )

# this is a dummy project to force some moc-files to be generated before 
# they are directly included in some cpp files (qtpropertybrowser only)
add_custom_target(${target_name}GenerateMoc
    DEPENDS ${paramPropertyBrowser_MOC_CPP}
)

add_dependencies(${target_name} ${target_name}GenerateMoc)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")

#copy all qm files to the translations subfolder of the itom root directory
foreach(_qm ${QM_FILES})
    set(COPY_SOURCES ${COPY_SOURCES} ${_qm})
    set(COPY_DESTINATIONS ${COPY_DESTINATIONS} "${CMAKE_CURRENT_BINARY_DIR}/../translation")
endforeach()

itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)

if(WIN32)
    #For Windows, the resulting library is copied to the designer-folder, 
    # such that the widgets are listed in the widget library of QtDesigner
    itom_add_designerlibrary_to_copy_list(${target_name} COPY_SOURCES COPY_DESTINATIONS)
else()
   #Under linux, copying the library (like under Windows) will result in a strange 
   # double-linked-list error, since two libraries with the
   #same content will then be loaded by itom. Therefore, a symbolic link to the 
   # original library is created in the designer-folder.
    file(MAKE_DIRECTORY "${ITOM_APP_DIR}/designer")
    add_custom_command(TARGET ${target_name} POST_BUILD                                  # Adds a post-build event to this target
            COMMAND ${CMAKE_COMMAND} -E create_symlink                                   # which executes "cmake - E create_symlink..."
                "$<TARGET_LINKER_FILE:${target_name}>"                                   # <--this is the old file
                "${ITOM_APP_DIR}/designer/$<TARGET_LINKER_FILE_NAME:${target_name}>"     # <--this is the new file
        )
endif()

itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)

foreach(headerFile ${itomWidgets_HEADERS}) 
    file(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itomWidgets FILES_MATCHING PATTERN "*.h")
endforeach(headerFile)