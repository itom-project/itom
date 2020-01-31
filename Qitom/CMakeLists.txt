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
# itom is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
# General Public Licence for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.1)

set(target_name qitom) 
project(${target_name})

message(STATUS "--------------- EXECUTABLE ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

option(BUILD_WITH_PYMATLAB "Build itom with integrated python module 'matlab' to control and use Matlab within itom." OFF)
option(BUILD_WITH_HELPVIEWER "Build itom with help viewer for the itom documentation (only >= Qt 5.6.0)." ON)
option(BUILD_WITH_GIMMICKS "Build itom with gimmicks." ON)
option(CONSIDER_GIT_SVN "If ON GIT version history will be considered." ON)
option(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
set(ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(${BUILD_WITH_HELPVIEWER})
    #HELPVIEWER is only available for Qt >= 5.6.0
    #at first, get the available Qt version
    itom_find_package_qt(ON Core)

    if((NOT ${Qt5_FOUND}) OR (Qt5_VERSION VERSION_LESS 5.6.0))
        message(WARNING "BUILD_WITH_HELPVIEWER is set to FALSE, since it is only possible for Qt >= 5.6.0")
        set(BUILD_WITH_HELPVIEWER FALSE)
    endif()
endif()

if(${BUILD_WITH_HELPVIEWER})
    #at first, get the available Qt version
    itom_find_package_qt(ON Core)
    #requires Qt5 and Qt5WebEngine in order to built a built-in help viewer, since QtAssistant only has a limited set of rendering options since Qt5.6.
    itom_find_package_qt(ON Core Widgets UiTools PrintSupport Network Sql Xml OpenGL Svg WebEngineWidgets Help Designer LinguistTools)
    
    if(Qt5WebEngineWidgets_FOUND)
        add_definitions(-DUSEHELPVIEWER -D_USEHELPVIEWER)
    else()
        message(FATAL "Your Qt installation does not provide the component WebEngineWidgets. Disable the option BUILD_WITH_HELPVIEWER in CMAKE to proceed without the integrated help viewer")
    endif()
else()
    #uses the old QtAssistant as help viewer.
    itom_find_package_qt(ON Core Widgets UiTools PrintSupport Network Sql Xml OpenGL Svg Designer LinguistTools)
endif()

find_package(PythonLibs 3.0 REQUIRED) 

# can be used from 3.12 onwards
# find_package(Python 3.0 COMPONENTS Interpreter Development REQUIRED )
# FIND_PACKAGE (Python3 COMPONENTS Interpreter Development)

find_package(OpenCV COMPONENTS core REQUIRED)
if(BUILD_WITH_PCL)
    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
    # actually we need this only to configure the itom.bat including vtk, otherwise 
    # eventually pcl-tools plugin will not be loaded
    find_package(VTK QUIET)    
endif()

find_package(NumPy REQUIRED)
find_package(VisualLeakDetector QUIET)
find_package(HTMLHelp)
find_package(LATEX QUIET) 

message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Numpy include directory: ${NUMPY_INCLUDE_DIRS}")

# set some preprocessor directives
add_definitions(${Qt5Widgets_DEFINITIONS})
add_definitions(-DUNICODE -D_UNICODE)
add_definitions(-DITOM_CORE -DITOMSHAREDDESIGNER)

if(BUILD_WITH_PCL)
    add_definitions(-DUSEPCL -D_USEPCL)
endif()

if(BUILD_WITH_PYMATLAB)
    add_definitions(-DUSEPYMATLAB -D_USEPYMATLAB)
endif()

if(BUILD_WITH_GIMMICKS)
    add_definitions(-DUSEGIMMICKS -D_USEGIMMICKS)
endif()

if(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)
    add_definitions(-DVISUAL_LEAK_DETECTOR_CMAKE)
endif()

# check files and directories from pymatlab, if flag BUILD_WITH_PYMATLAB is set
if(BUILD_WITH_PYMATLAB)
    if(WIN32)
        set(MATLAB_ROOT_PATH
            ${MATLAB_ROOT_PATH}
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.12;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.14;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\8.0;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\8.1;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2015a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2015b;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2016a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2016b;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2017a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2017b;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2018a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2018b;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2019a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2019b;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2020a;MATLABROOT]"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\R2020b;MATLABROOT]"
        )
        
        if(BUILD_TARGET64)
            set(MATLAB_LIB_SUFFIXES
                "lib/win64"
                "lib/win64/microsoft"
                "extern/lib/win64"
                "extern/lib/win64/microsoft"
            )
        else()
            set(MATLAB_LIB_SUFFIXES
                "lib/win32"
                "lib/win32/microsoft"
                "extern/lib/win32"
                "extern/lib/win32/microsoft"
            )
        endif()
    else()
        set(MATLAB_LIB_SUFFIXES)
    endif()
  
    find_path(PYMATLAB_INCLUDE_DIR "engine.h" PATHS ${MATLAB_ROOT_PATH}/extern/include)
    if(NOT PYMATLAB_INCLUDE_DIR)
        message(FATAL_ERROR "BUILD_WITH_PYMATLAB is checked, but PYMATLAB_INCLUDE_DIR could not be found")
    endif()
else()
    set(PYMATLAB_INCLUDE_DIR "")
endif()


include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5WebEngineWidgets_INCLUDE_DIRS}
    ${Qt5Help_INCLUDE_DIRS}
    ${Qt5Network_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PYMATLAB_INCLUDE_DIR}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
    ../DataObject/
    ../PointCloud/
    ../AddInManager/
    ../itomWidgets
)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${CMAKE_CURRENT_BINARY_DIR}/../PointCloud
    ${CMAKE_CURRENT_BINARY_DIR}/../AddInManager
    
)
 
set(qitom_HEADERS_NOMOC
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructures.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresPrimitives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/typeDefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/numeric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/color.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../AddInManager/pluginModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../AddInManager/addInManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../AddInManager/algoInterfaceValidator.h
    
)

set(qitom_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../AddInManager/0.h   # this file must be first in list for automoc
    ${CMAKE_CURRENT_SOURCE_DIR}/main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.h
    ${CMAKE_CURRENT_SOURCE_DIR}/global.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/fileDownloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/guiHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/sharedPointerHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/sleeper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpViewer/helpViewer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpViewer/qtHelpUrlSchemeHandler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/qpropertyHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/bookmarkModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/classNavigatorItem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/itomFileSystemModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/leafFilterProxyModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pipManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userInteractionWatcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonAutoInterval.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonDataObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngineInc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFont.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItom.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItomMetaObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonJedi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonMatlab.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNone.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNpDataObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNumeric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPCL.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlotItem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProgressObserver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProxy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRegion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRgba.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonShape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonSharedPointerGuard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonStream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythontParamConversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUiTimer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonWorkspace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/qDebugStream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractFilterDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractPropertyPageWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogCloseItom.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenNewGui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogProperties.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSnapshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogTimerManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/itomSpinBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGotoAssignment.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorScripts.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralPlotSettings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralStyles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPalettes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsAlgorithms.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsActuators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropWorkspaceUnpack.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractCodeEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/bookmarkDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/figureWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/helpDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/itomQWidgets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/lastCommandDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorPrinter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjPCLFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractItomDesignerPlugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractNode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/codeEditor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/delayJobRunner.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/mode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/textDecoration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/textBlockUserData.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/foldDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/charBasedFoldDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/indentFoldDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/modesManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/panelsManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/textDecorationsManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/autoindent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/caretLineHighlight.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/errorLineHighlight.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/lineBackgroundMarker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/indenter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/occurrences.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyAutoIndent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/symbolMatcherMode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyCalltips.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyGotoAssignment.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/codeCompletion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/wordclick.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/checkerBookmarkPanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/breakpointPanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/foldingPanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/lineNumber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/pythonSyntaxHighlighter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/syntaxHighlighterBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/codeEditorStyle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/utils/utils.h
)

set(qitom_ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogCloseItom.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenNewGui.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSnapshot.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogTimerManager.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputDialog.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGotoAssignment.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorScripts.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralPlotSettings.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralStyles.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPalettes.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsAlgorithms.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsActuators.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropWorkspaceUnpack.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiDebugViewer.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiMainWindow.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.ui
)

if(QT5_FOUND)
    QT5_WRAP_UI(qitom_ui_MOC ${qitom_ui})
endif()

set(qitom_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/qitom.qrc
)

if(QT5_FOUND)
    QT5_ADD_RESOURCES(qitom_rcc_MOC ${qitom_rcc})
else()
    message(SEND_ERROR "Currently only Qt5 is supported")
endif()

set(qitom_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/global.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/fileDownloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/guiHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/qpropertyHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpViewer/helpViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpViewer/qtHelpUrlSchemeHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/bookmarkModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/classNavigatorItem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/itomFileSystemModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/leafFilterProxyModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pipManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userInteractionWatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonAutoInterval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonDataObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFont.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonMatlab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNpDataObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNumeric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPCL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlotItem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProgressObserver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRegion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRgba.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonShape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonSharedPointerGuard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonStream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythontParamConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUiTimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonWorkspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/qDebugStream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractFilterDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogCloseItom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenNewGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogProperties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSnapshot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogTimerManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGotoAssignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorScripts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralPlotSettings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralStyles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPalettes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsAlgorithms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPluginsActuators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropWorkspaceUnpack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractCodeEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/bookmarkDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/figureWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/helpDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/lastCommandDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorPrinter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/codeEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/delayJobRunner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/mode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/textDecoration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/textBlockUserData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/foldDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/charBasedFoldDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/foldDetector/indentFoldDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/modesManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/panelsManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/managers/textDecorationsManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/autoindent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/caretLineHighlight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/errorLineHighlight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/lineBackgroundMarker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/indenter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/occurrences.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyAutoIndent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/symbolMatcherMode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyCalltips.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/pyGotoAssignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/codeCompletion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/modes/wordclick.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/checkerBookmarkPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/breakpointPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/foldingPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/panels/lineNumber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/pythonSyntaxHighlighter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/syntaxHighlighterBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/syntaxHighlighter/codeEditorStyle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditor/utils/utils.cpp
)

if(MSVC)
    list(APPEND qitom_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
endif()

#handle translations
# see e.g. http://stackoverflow.com/questions/5630155/how-to-integrate-qt-internationalization-to-cmake
#set(qitom_ts
#    ${CMAKE_CURRENT_SOURCE_DIR}/../translation/qitom_de.ts
#)

#add filters to visual studio
itom_add_source_group(api)
itom_add_source_group(helper)
itom_add_source_group(models)
itom_add_source_group(organizer)
itom_add_source_group(python)
itom_add_source_group(ui)
itom_add_source_group(widgets)
itom_add_source_group(helpViewer)
itom_add_source_group(codeEditor)
itom_add_source_group(codeEditor/foldDetector)
itom_add_source_group(codeEditor/managers)
itom_add_source_group(codeEditor/modes)
itom_add_source_group(codeEditor/panels)
itom_add_source_group(codeEditor/syntaxHighlighter)
itom_add_source_group(codeEditor/utils)
#end add filters to visual studio

if(MSVC)
    # Qt disables the native wchar_t type, do it too to avoid linking issues
    # well that changed in qt5 so go back to built-in type
    if(QT5_FOUND)
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zc:wchar_t" )
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../)

# First for the generic no-config case (e.g. with mingw)
set(ITOMOUTPUTDIR "../")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )

# Second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
endforeach()

# HANDLE VERSION (FROM GIT)
unset(GIT_FOUND CACHE)
find_package(Git)

if(CONSIDER_GIT_SVN AND GIT_FOUND)
    message(STATUS "Using GIT as repository information")
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/pyGit2Header.py.in ${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py )
    
    add_custom_target(
        ${target_name}_prebuild
        COMMAND    ${PYTHON_EXECUTABLE}
        "${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py"
    )
else()
    message(STATUS "Git versioning was not enabled (CONSIDER_GIT_SVN OFF) or git could not be found. Default version.h file is generated")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/version_nogit.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/version.h)
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/version_nogit.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)
    
    #empty target, necessary for translations (later). All prebuild-events are connected to this target.
    add_custom_target(
        ${target_name}_prebuild
    )
endif()
# END HANDLE VERSION

#handle translations BEGIN
set(FILES_TO_TRANSLATE ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_ui} ) #PARENT_SCOPE)
set(TRANSLATION_OUTPUT_FILES)
set(TRANSLATIONS_FILES)
file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")

if(UPDATE_TRANSLATIONS)
    if(QT5_FOUND)
        itom_qt5_create_translation(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE})
    else()
        message(SEND_ERROR "Currently only Qt5 is supported")
    endif()
    
    add_custom_target (${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
    add_dependencies(${target_name}_prebuild ${target_name}_translation)
    
    if(QT5_FOUND)
        itom_qt5_compile_translation(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${TRANSLATIONS_FILES})
    else()
        message(SEND_ERROR "Currently only Qt5 is supported")
    endif()
else()
    if(QT5_FOUND)
        itom_qt5_compile_translation(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${EXISTING_TRANSLATION_FILES})
    else()
        message(SEND_ERROR "Currently only Qt5 is supported")
    endif()
endif()
#handle translations END

#Find ITOM_VERION version by looking at global.h
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/global.h ITOM_VERSIONS_TMP REGEX "^#define ITOM_VERSION_[A-Z]+[ \t]+0x[0-9A-F]+$")
string(REGEX REPLACE ".*#define ITOM_VERSION_MAJOR[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_MAJOR ${ITOM_VERSIONS_TMP})
string(REGEX REPLACE ".*#define ITOM_VERSION_MINOR[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_MINOR ${ITOM_VERSIONS_TMP})
string(REGEX REPLACE ".*#define ITOM_VERSION_PATCH[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_PATCH ${ITOM_VERSIONS_TMP})
#set(ITOM_VERSION_STRING ${ITOM_VERSION_MAJOR}.${ITOM_VERSION_MINOR}.${ITOM_VERSION_PATCH} CACHE STRING "" FORCE)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/global.h ITOM_VERSIONS_TMP REGEX "^#define ITOM_VERSION_STR[ \t]+.*$")
string(REGEX REPLACE ".*#define ITOM_VERSION_STR[ \t]+.([0-9\\.]+).*" "\\1" ITOM_VERSION_STRING ${ITOM_VERSIONS_TMP})

if(APPLE)
    # Set meta data for OS X bundle
    set(MACOSX_BUNDLE TRUE)
    set(MACOSX_BUNDLE_BUNDLE_NAME "itom")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "ITO.itom")
    set(MACOSX_BUNDLE_INFO_STRING "itom - an Open Source Measurement, Automation and Evaluation Software\nhttp://itom.bitbucket.org")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${ITOM_VERSION_STRING}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${ITOM_VERSION_STRING}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION ${ITOM_VERSION_STRING})
    set(MACOSX_BUNDLE_COPYRIGHT "GNU LGPL with itom exception. Used plugins and libraries might differ. http://itom.bitbucket.org/latest/docs/01_introduction/introduction.html#licensing.")

    # Setting icon for OS X
    set(target_icon itomicon.icns)
    set(target_icon_path "${CMAKE_CURRENT_SOURCE_DIR}/appicon/osx/${target_icon}")
    set(MACOSX_BUNDLE_ICON_FILE "${target_icon}")
    set_source_files_properties(${target_icon_path} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    # Add executable for OS X
    add_executable(${target_name} MACOSX_BUNDLE ${target_icon_path} ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_HEADERS_MOC} ${qitom_ui_MOC} ${qitom_rcc_MOC} ${EXISTING_TRANSLATION_FILES})

    # Set binary file names for OS X
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG "itomd")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE "itom")

    # Copy files into app bundle after build
    itom_copy_type_to_bundle_nonrec(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../" "" ".dylib")
    itom_copy_type_to_bundle_nonrec(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../" "" ".py")
    itom_copy_to_bundle(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../itom-packages" "itom-packages")
    itom_copy_to_bundle(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../plugins" "plugins")
    itom_copy_to_bundle(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../designer" "designer")
    itom_copy_to_bundle(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../help" "help")
    itom_copy_to_bundle(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../demo" "../demo")
else(APPLE)
    # Executable for WINDOWS and LINUX
    add_executable(${target_name} WIN32 ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_HEADERS_MOC} ${qitom_ui_MOC} ${qitom_rcc_MOC} ${EXISTING_TRANSLATION_FILES} )

    # Set binary file names for  WINDOWS and LINUX
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG "qitomd")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE "qitom")
endif(APPLE)

add_dependencies(${target_name} ${target_name}_prebuild)
add_dependencies(${target_name} itomWidgets)

set( WINICONRES "" )

if(MSVC)
    set(CMAKE_CXX_STANDARD_LIBRARIES "user32.lib")
    if(CMAKE_CL_64)
        set( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon64.res )
    else()
        set( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon32.res )
    endif()
endif()

if(PYTHON_LIBRARY_DEBUG AND PYTHON_LIBRARY_RELEASE)
    set(PYTHON_LIBRARY_ITOM ${PYTHON_LIBRARY_RELEASE})
#only debug
elseif(PYTHON_LIBRARY_DEBUG)
    message(STATUS "Python is always linked as release version to itom, therefore indicate the release library of python")
#only release
elseif(PYTHON_LIBRARY_RELEASE)
    set(PYTHON_LIBRARY_ITOM ${PYTHON_LIBRARY_RELEASE})
elseif(PYTHON_LIBRARY)
    set(PYTHON_LIBRARY_ITOM ${PYTHON_LIBRARY})
#no library found
else()
    message(STATUS "Python could not be found")
endif()

if(BUILD_WITH_PCL)
    set(POINTCLOUD_LIBRARIES pointcloud)
else()
    set(POINTCLOUD_LIBRARIES "")
endif()


if(QT5_FOUND)
    if(WIN32)
        target_link_libraries(${target_name} ${WINICONRES} ${PYTHON_LIBRARY_ITOM} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} ${QT5_LIBRARIES} dataobject addinmanager qpropertyeditor itomCommonLib itomCommonQtLib itomWidgets itomCommonPlotLib ${PYMATLAB_LIBRARIES} shell32)
    else()
        target_link_libraries(${target_name} ${WINICONRES} ${PYTHON_LIBRARY_ITOM} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} ${QT5_LIBRARIES} dataobject addinmanager qpropertyeditor itomCommonLib itomCommonQtLib itomWidgets itomCommonPlotLib ${PYMATLAB_LIBRARIES})
    endif()
endif()
 
itom_build_parallel_linux(${target_name})

# for backward compatiblity: enum.py is a default python library from Python 3.4 on. In order to 
# also support earlier versoins, enum.py was also part of the itom-packages folder. However, this
# version is old and creates conflicts with newer versions required by further modern python packages.
# Therefore, enum.py was removed from itom-packages. If one uses Python 3.3 or earlier, the packages enum34
# has to be installed manually!
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../itom-packages/enum.py")
    message(WARNING "itom-packages/enum.py exists. This file will be deleted.")
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/../itom-packages/enum.py")
endif()

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/plotLegends")
    message(WARNING "SDK/include/plotLegends exists. This folder will be removed.")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/plotLegends")
endif()

# copy some folders and files during generate
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itom-packages                 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. PATTERN ".svn" EXCLUDE PATTERN "__pycache__" EXCLUDE)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../demo                          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. PATTERN ".svn" EXCLUDE PATTERN "__pycache__" EXCLUDE)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itomSettings/itomDefault.ini  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../itomSettings)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoDebugger.py                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoFunctions.py               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)

# - copy files to itom SDK folder in binary directory as post build step (after every build)
set(COPY_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.h)
set(COPY_DESTINATIONS ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include)
itom_post_build_copy_files(qitom COPY_SOURCES COPY_DESTINATIONS)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterfaceVersion.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_MAJOR [0-9]+$")
if(ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_MAJOR ([0-9]+)$" "\\1" ITOM_ADDININTERFACE_VERSION_MAJOR ${ITOM_VERSION_TMP})
else(ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_MAJOR in addInInterfaceVersion.h missing. Can not detect version of itom SDK")
    set(ITOM_ADDININTERFACE_VERSION_MAJOR "?")
endif(ITOM_VERSION_TMP)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterfaceVersion.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_MINOR [0-9]+$")
if(ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_MINOR ([0-9]+)$" "\\1" ITOM_ADDININTERFACE_VERSION_MINOR ${ITOM_VERSION_TMP})
else(ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_MINOR in addInInterfaceVersion.h missing. Can not detect version of itom SDK")
    set(ITOM_ADDININTERFACE_VERSION_MINOR "?")
endif(ITOM_VERSION_TMP)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterfaceVersion.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_PATCH [0-9]+$")
if(ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_PATCH ([0-9]+)$" "\\1" ITOM_ADDININTERFACE_VERSION_PATCH ${ITOM_VERSION_TMP})
else(ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_PATCH in addInInterfaceVersion.h missing. Can not detect version of itom SDK")
    set(ITOM_ADDININTERFACE_VERSION_PATCH "?")
endif(ITOM_VERSION_TMP)

set(ITOM_VERSION_STRING_WITH_API "ito.AddIn.InterfaceBase/${ITOM_ADDININTERFACE_VERSION_MAJOR}.${ITOM_ADDININTERFACE_VERSION_MINOR}.${ITOM_ADDININTERFACE_VERSION_PATCH}")

#find designer plugin interface version from plot/designerPluginInterfaceVersion.h
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../plot/designerPluginInterfaceVersion.h DesignerPluginInterfaceVersionFull LIMIT_COUNT 1 REGEX "^#define ITOM_DESIGNERPLUGININTERFACE_VERSION_STR.*$")
string(REGEX MATCH "([0-9]+),([0-9]+),([0-9]+)" DESIGNER_PLUGIN_INTERFACE_VERSION ${DesignerPluginInterfaceVersionFull}) #searches for Number.Number.Number[.Number] in DesignerPluginInterfaceVersionFull
string(REPLACE "," "." DESIGNER_PLUGIN_INTERFACE_VERSION ${DESIGNER_PLUGIN_INTERFACE_VERSION})

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/itom_sdk.h.in ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itom_sdk.h)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../pluginMetaData.json.in ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/pluginMetaData.json )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../translation )
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib )
 
#ADD symbols/lib directory in application directory where pdb files will be loaded for debugging (MSVC only)
if(MSVC)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../symbols/dll )
endif()

#ADD a batch file for itom-release 
if(MSVC)
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../itom.bat.in ${CMAKE_CURRENT_BINARY_DIR}/../itom.bat )
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../itomd.bat.in ${CMAKE_CURRENT_BINARY_DIR}/../itomd.bat )
endif()

#Configure a RC file for MS visual studio to add version information in explorer to exe file
if(MSVC)
    message(STATUS "Using rc-File for version information under windows")
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/version.rc.in ${CMAKE_CURRENT_BINARY_DIR}/version.rc )
    
endif()

# Copying scripts for OS X build process
if(APPLE)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../osx_install_dependencies.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
endif(APPLE)


