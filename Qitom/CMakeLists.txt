SET (target_name qitom)
         
project(${target_name})
 
cmake_minimum_required(VERSION 2.8.9)

#CMAKE Policies
IF(CMAKE_VERSION VERSION_GREATER 2.8.12)
    #https://bugreports.qt.io/browse/QTBUG-39457
    cmake_policy(SET CMP0043 OLD)
ENDIF()
  
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_WITH_PCL "Build itom with PointCloudLibrary support (pointCloud, polygonMesh, point...)" ON)
OPTION(BUILD_WITH_PYMATLAB "Build itom with integrated python module 'matlab' to control and use Matlab within itom." OFF)
OPTION(CONSIDER_GIT_SVN "If ON GIT version history will be considered." ON)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
include("../ItomBuildMacros.cmake")  
       
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE_QT(ON Core Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools Svg)
find_package(PythonLibs 3.0 REQUIRED)  
find_package(QScintilla REQUIRED)  
find_package(OpenCV COMPONENTS core REQUIRED)
IF (BUILD_WITH_PCL)
    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
ENDIF (BUILD_WITH_PCL)
find_package(NumPy REQUIRED)
find_package(VisualLeakDetector QUIET)
find_package(HTMLHelp)
find_package(LATEX QUIET) 
        
           
ADD_DEFINITIONS(${QT_DEFINITIONS})  
add_definitions(${Qt5Widgets_DEFINITIONS})

IF (BUILD_UNICODE) 
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

IF (BUILD_WITH_PCL)
    ADD_DEFINITIONS(-DUSEPCL -D_USEPCL)
ENDIF (BUILD_WITH_PCL)

IF (BUILD_WITH_PYMATLAB)
    ADD_DEFINITIONS(-DUSEPYMATLAB -D_USEPYMATLAB)
ENDIF (BUILD_WITH_PYMATLAB)

ADD_DEFINITIONS(-DCMAKE -DITOM_CORE -DITOMSHAREDDESIGNER)
 
IF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)
    ADD_DEFINITIONS(-DVISUAL_LEAK_DETECTOR_CMAKE)
ENDIF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)


# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

# check files and directories from pymatlab, if flag BUILD_WITH_PYMATLAB is set
IF (BUILD_WITH_PYMATLAB)
    FIND_PATH(PYMATLAB_INCLUDE_DIRS "engine.h")
    IF (NOT PYMATLAB_INCLUDE_DIRS)
        MESSAGE(FATAL_ERROR "BUILD_WITH_PYMATLAB is checked, but PYMATLAB_INCLUDE_DIRS could not be found")
    ENDIF()
    FIND_LIBRARY(PYMATLAB_LIB_ENG libeng PATHS ${MATLAB_LIBRARY_DIR})
    FIND_LIBRARY(PYMATLAB_LIB_MX libmx PATHS ${MATLAB_LIBRARY_DIR})
    IF (PYMATLAB_LIB_ENG AND PYMATLAB_LIB_MX)
        SET(PYMATLAB_LIBRARIES ${PYMATLAB_LIB_ENG} ${PYMATLAB_LIB_MX})
    ELSE()
        MESSAGE(FATAL_ERROR "The matlab libraries libeng or libmx could not be found. Indicate MATLAB_LIBRARY_DIR as directory where these libraries can be found")
        SET(PYMATLAB_LIBRARIES "")
    ENDIF()
ELSE (BUILD_WITH_PYMATLAB)
    SET(PYMATLAB_INCLUDE_DIRS "")
    SET(PYMATLAB_LIBRARIES "")
ENDIF (BUILD_WITH_PYMATLAB)


INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${Qt5Widgets_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIRS}
    ${QSCINTILLA_INCLUDE_DIR}
    ${OpenCV_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${PYMATLAB_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
    ../DataObject/
    ../PointCloud/
    ../itomWidgets
)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${CMAKE_CURRENT_BINARY_DIR}/../PointCloud
    ${OpenCV_DIR}/lib
)

set(qitom_HEADERS_NOMOC
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/apiFunctionsGraphInc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/apiFunctionsInc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructures.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresPrimitives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/typeDefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/color.h
    
)

set(qitom_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/aaa.h   # this file must be first in list for automoc
    ${CMAKE_CURRENT_SOURCE_DIR}/main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.h
    ${CMAKE_CURRENT_SOURCE_DIR}/global.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/fileDownloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/paramHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/sharedPointerHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/sleeper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/classNavigatorItem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/leafFilterProxyModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/PlugInModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pipManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/addInManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/algoInterfaceValidator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/qsciApiManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userOrganizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userInteractionWatcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonAutoInterval.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonDataObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngineInc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItom.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItomMetaObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonMatlab.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNpDataObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNumeric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPCL.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProxy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRegion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonSharedPointerGuard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonStream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythontParamConversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRgba.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFont.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlotItem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUiTimer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonWorkspace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/qDebugStream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractFilterDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractPropertyPageWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogProperties.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/itomSpinBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractPyScintillaWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/helpDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/itomQWidgets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/lastCommandDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/figureWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractNode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjPCLFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractItomDesignerPlugin.h
)

IF (QT5_FOUND)
#    QT5_WRAP_CPP(qitom_HEADERS_MOC ${qitom_HEADERS})
ELSE (QT5_FOUND)
    QT4_WRAP_CPP_ITOM(qitom_HEADERS_MOC ${qitom_HEADERS})
ENDIF (QT5_FOUND)

set(qitom_ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiDebugViewer.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiMainWindow.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.ui
)

IF (QT5_FOUND)
    QT5_WRAP_UI(qitom_ui_MOC ${qitom_ui})
ELSE (QT5_FOUND)
    QT4_WRAP_UI_ITOM(qitom_ui_MOC ${qitom_ui})
ENDIF (QT5_FOUND)


set(qitom_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/qitom.qrc
)

IF (QT5_FOUND)
    QT5_ADD_RESOURCES(qitom_rcc_MOC ${qitom_rcc})
ELSE (QT5_FOUND)
    QT4_ADD_RESOURCES(qitom_rcc_MOC ${qitom_rcc})
ENDIF (QT5_FOUND)

set(qitom_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/global.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/fileDownloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/paramHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/classNavigatorItem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/leafFilterProxyModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/PlugInModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pipManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/addInManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/algoInterfaceValidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/qsciApiManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/userInteractionWatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonAutoInterval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonDataObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNpDataObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNumeric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPCL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRegion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonSharedPointerGuard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonStream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythontParamConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonRgba.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFont.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlotItem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUiTimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonWorkspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/qDebugStream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractFilterDialog.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/ui/AbstractPropertyPageWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogIconBrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogProperties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPipManagerInstall.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/helpTreeDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleLastCommand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropHelpDock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractPyScintillaWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/helpDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/lastCommandDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/figureWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagementEdit.cpp    
)

if(MSVC)
    list(APPEND qitom_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
endif(MSVC)

#handle translations
# see e.g. http://stackoverflow.com/questions/5630155/how-to-integrate-qt-internationalization-to-cmake
#set(qitom_ts
#    ${CMAKE_CURRENT_SOURCE_DIR}/../translation/qitom_de.ts
#)

#add filters to visual studio
ADD_SOURCE_GROUP(api)
ADD_SOURCE_GROUP(helper)
ADD_SOURCE_GROUP(models)
ADD_SOURCE_GROUP(organizer)
ADD_SOURCE_GROUP(python)
ADD_SOURCE_GROUP(ui)
ADD_SOURCE_GROUP(widgets)
#end add filters to visual studio

if(MSVC)
    # Qt disables the native wchar_t type, do it too to avoid linking issues
    # well that changed in qt5 so go back to built-in type
    if (QT5_FOUND)
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zc:wchar_t" )
    else (QT5_FOUND)
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zc:wchar_t-" )
    endif (QT5_FOUND)
endif(MSVC)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../)

# First for the generic no-config case (e.g. with mingw)
SET(ITOMOUTPUTDIR "../")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ITOMOUTPUTDIR} )
# Second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ITOMOUTPUTDIR} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

# HANDLE VERSION (FROM GIT)
UNSET(GIT_FOUND CACHE)
find_package(Git)

IF(CONSIDER_GIT_SVN AND GIT_FOUND)
    MESSAGE(STATUS "Using GIT as repository information")
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/pyGit2Header.py.in ${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py )
    
    ADD_CUSTOM_TARGET (
        ${target_name}_prebuild
        COMMAND    ${PYTHON_EXECUTABLE}
        "${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py"
    )
ELSE()
    MESSAGE(STATUS "Git versioning was not enabled (CONSIDER_GIT_SVN OFF) or git could not be found. Default version.h file is generated")
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/version_nogit.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    FILE(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/version.h)
    FILE(RENAME ${CMAKE_CURRENT_BINARY_DIR}/version_nogit.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)
    
    #empty target, necessary for translations (later). All prebuild-events are connected to this target.
    ADD_CUSTOM_TARGET (
        ${target_name}_prebuild
    )
ENDIF(CONSIDER_GIT_SVN AND GIT_FOUND)
# END HANDLE VERSION

#handle translations BEGIN
set (FILES_TO_TRANSLATE ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_ui} ) #PARENT_SCOPE)
set (TRANSLATION_OUTPUT_FILES)
set (TRANSLATIONS_FILES)
file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
if (UPDATE_TRANSLATIONS)
    if (QT5_FOUND)
        QT5_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
    else (QT5_FOUND)
        QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
    endif (QT5_FOUND)
    
    add_custom_target (${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
    add_dependencies(${target_name}_prebuild ${target_name}_translation)
    
#    message(FATAL ${TRANSLATIONS_FILES} "---" ${EXISTING_TRANSLATION_FILES})
    
    if (QT5_FOUND)
        QT5_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${TRANSLATIONS_FILES})
    else (QT5_FOUND)
        QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${TRANSLATIONS_FILES})
    endif (QT5_FOUND)
else (UPDATE_TRANSLATIONS)
    if (QT5_FOUND)
        QT5_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${EXISTING_TRANSLATION_FILES})    
    else (QT5_FOUND)
        QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/../translation" ${target_name}_prebuild ${EXISTING_TRANSLATION_FILES})
    endif (QT5_FOUND)
endif (UPDATE_TRANSLATIONS)
#handle translations END

#Find ITOM_VERION version by looking at global.h
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/global.h ITOM_VERSIONS_TMP REGEX "^#define ITOM_VERSION_[A-Z]+[ \t]+0x[0-9A-F]+$")
string(REGEX REPLACE ".*#define ITOM_VERSION_MAJOR[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_MAJOR ${ITOM_VERSIONS_TMP})
string(REGEX REPLACE ".*#define ITOM_VERSION_MINOR[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_MINOR ${ITOM_VERSIONS_TMP})
string(REGEX REPLACE ".*#define ITOM_VERSION_PATCH[ \t]+(0x[0-9A-F]+).*" "\\1" ITOM_VERSION_PATCH ${ITOM_VERSIONS_TMP})
#set(ITOM_VERSION_STRING ${ITOM_VERSION_MAJOR}.${ITOM_VERSION_MINOR}.${ITOM_VERSION_PATCH} CACHE STRING "" FORCE)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/global.h ITOM_VERSIONS_TMP REGEX "^#define ITOM_VERSION_STR[ \t]+.*$")
string(REGEX REPLACE ".*#define ITOM_VERSION_STR[ \t]+.([0-9\\.]+).*" "\\1" ITOM_VERSION_STRING ${ITOM_VERSIONS_TMP})

IF(APPLE)
    # Set meta data for OS X bundle
    SET(MACOSX_BUNDLE TRUE)
    SET(MACOSX_BUNDLE_BUNDLE_NAME "itom")
    SET(MACOSX_BUNDLE_GUI_IDENTIFIER "ITO.itom")
    SET(MACOSX_BUNDLE_INFO_STRING "itom - an Open Source Measurement, Automation and Evaluation Software\nhttp://itom.bitbucket.org")
    SET(MACOSX_BUNDLE_LONG_VERSION_STRING "${ITOM_VERSION_STRING}")
    SET(MACOSX_BUNDLE_SHORT_VERSION_STRING "${ITOM_VERSION_STRING}")
    SET(MACOSX_BUNDLE_BUNDLE_VERSION ${ITOM_VERSION_STRING})
    SET(MACOSX_BUNDLE_COPYRIGHT "GNU LGPL with itom exception. Used plugins and libraries might differ. http://itom.bitbucket.org/latest/docs/01_introduction/introduction.html#licensing.")

    # Setting icon for OS X
    SET(target_icon itomicon.icns)
    SET(target_icon_path "${CMAKE_CURRENT_SOURCE_DIR}/appicon/osx/${target_icon}")
    SET(MACOSX_BUNDLE_ICON_FILE "${target_icon}")
    SET_SOURCE_FILES_PROPERTIES(${target_icon_path} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    # Add executable for OS X
    ADD_EXECUTABLE(${target_name} MACOSX_BUNDLE ${target_icon_path} ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_HEADERS_MOC} ${qitom_ui_MOC} ${qitom_rcc_MOC} ${EXISTING_TRANSLATION_FILES})

    # SET_SOURCE_FILES_PROPERTIES(
    #     ${CMAKE_BINARY_DIR}/../itoDebugger.py
    #     ${CMAKE_BINARY_DIR}/../itoFunctions.py
    #     PROPERTIES MACOSX_PACKAGE_LOCATION MacOSX
    # )

    # SET_SOURCE_FILES_PROPERTIES(
    #   apple1_res1
    #   apple1_res2
    #   PROPERTIES MACOSX_PACKAGE_LOCATION Resources
    #   )

    # Set binary file names for OS X
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG "itomd")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE "itom")

    # Copy files into app bundle after build
    COPY_TYPE_TO_BUNDLE_NONREC(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../" "" ".dylib")
    COPY_TYPE_TO_BUNDLE_NONREC(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../" "" ".py")
    COPY_TO_BUNDLE(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../itom-packages" "itom-packages")
    COPY_TO_BUNDLE(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../plugins" "plugins")
    COPY_TO_BUNDLE(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../designer" "designer")
    COPY_TO_BUNDLE(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../help" "help")
    COPY_TO_BUNDLE(${target_name} "${CMAKE_CURRENT_BINARY_DIR}/../demo" "../demo")
ELSE(APPLE)
    # Executable for WINDOWS and LINUX
    ADD_EXECUTABLE(${target_name} WIN32 ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_NOMOC} ${qitom_HEADERS_MOC} ${qitom_ui_MOC} ${qitom_rcc_MOC} ${EXISTING_TRANSLATION_FILES} )

    # Set binary file names for  WINDOWS and LINUX
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG "qitomd")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE "qitom")
ENDIF(APPLE)

ADD_DEPENDENCIES(${target_name} ${target_name}_prebuild)
ADD_DEPENDENCIES(${target_name} itomWidgets)

IF (QT5_FOUND)
    qt5_use_modules(${target_name} Core Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools)
ENDIF (QT5_FOUND)

SET( WINICONRES "" )

IF(MSVC)
    SET(CMAKE_CXX_STANDARD_LIBRARIES "user32.lib")
    IF(CMAKE_CL_64)
        SET( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon64.res )
    ELSE(CMAKE_CL_64)
        SET( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon32.res )
    ENDIF(CMAKE_CL_64)
ENDIF(MSVC)

if(PYTHON_LIBRARY_DEBUG AND PYTHON_LIBRARY_RELEASE)
    #set(PYTHON_LIBRARY_ITOM debug ${PYTHON_LIBRARY_DEBUG} optimized ${PYTHON_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
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

IF(BUILD_WITH_PCL)
    IF(BUILD_ITOMLIBS_SHARED)
        SET(POINTCLOUD_LIBRARIES pointcloud)
    ELSE(BUILD_ITOMLIBS_SHARED)
        SET(POINTCLOUD_LIBRARIES ${PCL_LIBRARIES} pointcloud)
    ENDIF(BUILD_ITOMLIBS_SHARED)
ELSE(BUILD_WITH_PCL)
    SET(POINTCLOUD_LIBRARIES "")
ENDIF(BUILD_WITH_PCL)

IF (QT5_FOUND)
    if (WIN32)
        TARGET_LINK_LIBRARIES(${target_name} ${WINICONRES} ${QSCINTILLA_LIBRARY} ${PYTHON_LIBRARY_ITOM} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} dataobject qpropertyeditor itomCommonLib itomCommonQtLib itomWidgets ${PYMATLAB_LIBRARIES} shell32)
    else (WIN32)
        TARGET_LINK_LIBRARIES(${target_name} ${WINICONRES} ${QSCINTILLA_LIBRARY} ${PYTHON_LIBRARY_ITOM} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} dataobject qpropertyeditor itomCommonLib itomCommonQtLib itomWidgets ${PYMATLAB_LIBRARIES})
    endif (WIN32)
ELSE (QT5_FOUND)
    TARGET_LINK_LIBRARIES(${target_name} ${WINICONRES} ${QT_LIBRARIES} ${QT_QTMAIN_LIBRARY} ${QT_QTSVG_LIBRARY} ${QT_QTXML_LIBRARY} ${QT_QTOPENGL_LIBRARY} ${QT_QTUITOOLS_LIBRARY}  ${QT_QTSQL_LIBRARY} ${QT_QTHELP_LIBRARY} ${QT_QTNETWORK_LIBRARY} ${QT_PRINTSUPPORT_LIBRARY} ${QT_UITOOLS_LIBRARY} ${QSCINTILLA_LIBRARY} ${PYTHON_LIBRARY_ITOM} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} dataobject qpropertyeditor itomCommonLib itomCommonQtLib itomWidgets ${PYMATLAB_LIBRARIES})
ENDIF (QT5_FOUND)
 
BUILD_PARALLEL_LINUX(${target_name})

# copy some folders and files during generate
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itom-packages                 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. PATTERN ".svn" EXCLUDE PATTERN "__pycache__" EXCLUDE)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../demo                          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. PATTERN ".svn" EXCLUDE PATTERN "__pycache__" EXCLUDE)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itomSettings/itomDefault.ini  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../itomSettings)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoDebugger.py                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..) # FILES_MATCHING PATTERN "*.py")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoFunctions.py               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..) # FILES_MATCHING PATTERN "*.py")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../common                        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include FILES_MATCHING PATTERN "*.h")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../pluginTemplates               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/pluginTemplates PATTERN ".svn" EXCLUDE)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../plot                          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include FILES_MATCHING PATTERN "*.h")
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud                   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK FILES_MATCHING PATTERN "*.h")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindGLEW.cmake                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindGlut.cmake               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindFFTW.cmake                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindOpenCV.cmake              DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindVisualLeakDetector.cmake  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindLAPACKE.cmake             DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindXsd.cmake                 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindXerces.cmake              DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindLibUSB.cmake              DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindITOM_SDK.cmake            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../Find3DCONNEXION.cmake         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindWindowsSDK.cmake          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
IF(MSVC)
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../designerPluginLibraryVersion.rc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../pluginLibraryVersion.rc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
ENDIF(MSVC)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../docs/docsTmpl/_themes         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/docs)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../docs/docsTmpl/pluginDoc       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/docs)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../docs/docsTmpl/sphinxext       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/docs)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../docs/userDoc/source/ext       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/docs)

#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/itom_sdk.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_MAJOR [0-9]+$")
if (ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_MAJOR ([0-9]+)$" "\\1" ITOM_VERSION_MAJOR ${ITOM_VERSION_TMP})
else (ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_MAJOR in addInInterface.h missing. Can not detect version of itom SDK")
    set(ITOM_VERSION_MAJOR "?")
endif (ITOM_VERSION_TMP)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_MINOR [0-9]+$")
if (ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_MINOR ([0-9]+)$" "\\1" ITOM_VERSION_MINOR ${ITOM_VERSION_TMP})
else (ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_MINOR in addInInterface.h missing. Can not detect version of itom SDK")
    set(ITOM_VERSION_MINOR "?")
endif (ITOM_VERSION_TMP)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h ITOM_VERSION_TMP REGEX "^.*#define ITOM_ADDININTERFACE_PATCH [0-9]+$")
if (ITOM_VERSION_TMP)
    string(REGEX REPLACE "^.*#define ITOM_ADDININTERFACE_PATCH ([0-9]+)$" "\\1" ITOM_VERSION_PATCH ${ITOM_VERSION_TMP})
else (ITOM_VERSION_TMP)
    message(WARNING "#define ITOM_ADDININTERFACE_PATCH in addInInterface.h missing. Can not detect version of itom SDK")
    set(ITOM_VERSION_PATCH "?")
endif (ITOM_VERSION_TMP)

SET(ITOM_VERSION_STRING_WITH_API "ito.AddIn.InterfaceBase/${ITOM_VERSION_MAJOR}.${ITOM_VERSION_MINOR}.${ITOM_VERSION_PATCH}")

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/itom_sdk.h.in ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/itom_sdk.h)

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../translation )
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../translation           DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. FILES_MATCHING PATTERN "*.qm")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib )
 
#ADD symbols/lib directory in application directory where pdb files will be loaded for debugging (MSVC only)
IF(MSVC)
    FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../symbols/dll )
ENDIF(MSVC)

#ADD a batch file for itom-release 
IF(MSVC)
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/../itom.bat.in ${CMAKE_CURRENT_BINARY_DIR}/../itom.bat )
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/../itomd.bat.in ${CMAKE_CURRENT_BINARY_DIR}/../itomd.bat )
ENDIF (MSVC)

#ADD relevant and bug sensitive path to visual studio debug enviroment via qitom.vcxproj.user
#IF(MSVC)
#    IF(BUILD_TARGET64)
#        CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/qitom.vcxproj.user.x64 ${CMAKE_CURRENT_BINARY_DIR}/qitom.vcxproj.user )
#    ELSE(BUILD_TARGET64)
#        CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/qitom.vcxproj.user.win32 ${CMAKE_CURRENT_BINARY_DIR}/qitom.vcxproj.user )
#    ENDIF(BUILD_TARGET64)
#ENDIF (MSVC)

#Configure a RC file for MS visual studio to add version information in explorer to exe file
if(MSVC)
    MESSAGE(STATUS "Using rc-File for version information under windows")
    #CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/pyItomFileVersion.py.in ${CMAKE_CURRENT_BINARY_DIR}/pyItomFileVersion.py )
    #CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/version.rctmpl ${CMAKE_CURRENT_BINARY_DIR}/version.rc )
    #ADD_CUSTOM_COMMAND (
    #    TARGET qitom
    #    PRE_BUILD
    #    COMMAND    ${PYTHON_EXECUTABLE}
    #    ARGS       "${CMAKE_CURRENT_BINARY_DIR}/pyItomFileVersion.py"
    #)
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/version.rc.in ${CMAKE_CURRENT_BINARY_DIR}/version.rc )
    
ENDIF (MSVC)

# Copying scripts for OS X build process
IF(APPLE)
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../osx_install_dependencies.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
ENDIF(APPLE)


