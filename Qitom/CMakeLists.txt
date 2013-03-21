SET (target_name qitom)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(CONSIDER_GIT_SVN "If ON GIT or SVN will be considered." ON)
OPTION(PREFER_GIT "If ON GIT will be prefered, else use SVN." ON)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

find_package(Qt4 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(QScintilla REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
find_package(NumPy REQUIRED)
find_package(VisualLeakDetector QUIET)
find_package(HTMLHelp)

ADD_DEFINITIONS(${QT_DEFINITIONS})

IF (BUILD_UNICODE)
	ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

ADD_DEFINITIONS(-DCMAKE -DITOM_CORE)


IF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)
    ADD_DEFINITIONS(-DVISUAL_LEAK_DETECTOR_CMAKE)
ENDIF(VISUALLEAKDETECTOR_FOUND AND VISUALLEAKDETECTOR_ENABLED)

SET (QT_USE_QTOPENGL TRUE)
SET (QT_USE_QTXML TRUE)
SET (QT_USE_QTSVG TRUE)
SET (QT_USE_QTNETWORK TRUE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTSVG_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIRS}
    ${QSCINTILLA_INCLUDE_DIR}
    ${OpenCV_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
    ../DataObject/
	../PointCloud/
)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
	${CMAKE_CURRENT_BINARY_DIR}/../PointCloud
    ${OpenCV_DIR}/lib
)

INCLUDE(${QT_USE_FILE})

set(qitom_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.h
    ${CMAKE_CURRENT_SOURCE_DIR}/global.h
    ${CMAKE_CURRENT_SOURCE_DIR}/main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.h
	${CMAKE_CURRENT_SOURCE_DIR}/../common/apiFunctionsGraphInc.h
	${CMAKE_CURRENT_SOURCE_DIR}/../common/apiFunctionsInc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperActuator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperGrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructures.h
	${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/typeDefs.h
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractNode.h
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractFigure.h
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjFigure.h
	${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctions.h
	${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/helper/paramHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/sharedPointerHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/models/PlugInModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/addInManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/algoInterfaceValidator.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/qsciApiManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.h
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.h
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.h
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
	${CMAKE_CURRENT_SOURCE_DIR}/ui/itomSpinBox.h
	${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.h
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.h
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.h
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.h	
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractPyScintillaWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.h
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.h
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.h	
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.h
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/iconBrowserDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/QTabWidgetItom.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptTabWidget.h
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.h	
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.h
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.h
)

QT4_WRAP_CPP_ITOM(qitom_HEADERS_MOC ${qitom_HEADERS})

set(qitom_ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogLoadedPlugins.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogNewPluginInstance.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogOpenFileWithFilter.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogPluginPicker.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReloadModule.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogReplace.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSaveFileWithFilter.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogSelectUser.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogVariableDetail.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetInfoBox.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.ui
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.ui
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiDebugViewer.ui
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/uiMainWindow.ui
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.ui	
)

QT4_WRAP_UI_ITOM(qitom_ui_MOC ${qitom_ui})

set(qitom_rcc
    ${CMAKE_CURRENT_SOURCE_DIR}/qitom.qrc
)

QT4_ADD_RESOURCES(qitom_rcc_MOC ${qitom_rcc})

set(qitom_SOURCES 
	${CMAKE_CURRENT_SOURCE_DIR}/AppManagement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/global.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mainApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInGrabber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/addInInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperActuator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helperGrabber.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedFunctionsQt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructures.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresGraphics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sharedStructuresQt.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractNode.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractFigure.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../plot/AbstractDObjFigure.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctions.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/api/apiFunctionsGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/IOHelper.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/helper/paramHelper.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/helper/versionHelper.cpp	
    ${CMAKE_CURRENT_SOURCE_DIR}/models/breakPointModel.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/models/PlugInModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/UserModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/addInManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/algoInterfaceValidator.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/designerWidgetOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/helpSystem.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/paletteOrganizer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/organizer/processOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/qsciApiManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/scriptEditorOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/uiOrganizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/organizer/widgetWrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonDataObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonEngine.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/python/pythonItom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNpDataObject.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/python/pythonNumeric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPCL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonPlugins.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/python/pythonProxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonQtSignalMapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonSharedPointerGuard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonStream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythontParamConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUi.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/python/pythonUiTimer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/python/pythonWorkspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/qDebugStream.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/abstractFilterDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/AbstractPropertyPageWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogAbout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogEditBreakpoint.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/dialogGoto.cpp
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
	${CMAKE_CURRENT_SOURCE_DIR}/ui/paramInputParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetFindWord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropConsoleWrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAPI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorAutoCompletion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorCalltips.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorGeneral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropEditorStyles.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropFigurePlugins.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralApplication.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropGeneralLanguage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/widgetPropPythonStartup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/abstractPyScintillaWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/AIManagerWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/breakPointDockWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/callStackDockWidget.cpp	
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/consoleWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/fileSystemDockWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/iconBrowserDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/mainWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/scriptTabWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiDialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/userUiWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets/workspaceWidget.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/widgets/userManagement.cpp	
)

#handle translations
# see e.g. http://stackoverflow.com/questions/5630155/how-to-integrate-qt-internationalization-to-cmake
#set(qitom_ts
#    ${CMAKE_CURRENT_SOURCE_DIR}/../translation/qitom_de.ts
#)

#QT4_ADD_TRANSLATION(qitom_qm ${qitom_ts})

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
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zc:wchar_t-" )
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

#handle translations BEGIN STEP 1
set (FILES_TO_TRANSLATE ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_ui} ) #PARENT_SCOPE)

set (TRANSLATION_OUTPUT_FILES)
set (TRANSLATIONS_FILES)

file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations BEGIN END 1

ADD_EXECUTABLE(${target_name} WIN32 ${qitom_SOURCES} ${qitom_HEADERS} ${qitom_HEADERS_MOC} ${qitom_ui_MOC} ${qitom_rcc_MOC} ${EXISTING_TRANSLATION_FILES} )
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG "qitomd")
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE "qitom")

#handle translations BEGIN STEP 2
if (UPDATE_TRANSLATIONS)
	QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
	
	add_custom_target (${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
	add_dependencies(${target_name} ${target_name}_translation)
	
	QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${TRANSLATIONS_FILES})
else (UPDATE_TRANSLATIONS)
	QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${EXISTING_TRANSLATION_FILES})
endif (UPDATE_TRANSLATIONS)
#handle translations END STEP 2

SET( WINICONRES "" )

IF(MSVC)
    SET(CMAKE_CXX_STANDARD_LIBRARIES "user32.lib")
	IF(CMAKE_CL_64)
		SET( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon64.res )
	ELSE(CMAKE_CL_64)
		SET( WINICONRES ${CMAKE_CURRENT_SOURCE_DIR}/appicon/windows/itomicon32.res )
	ENDIF(CMAKE_CL_64)
ENDIF(MSVC)

TARGET_LINK_LIBRARIES(${target_name} ${WINICONRES} ${QT_LIBRARIES} ${QT_QTMAIN_LIBRARY} ${QT_QTSVG_LIBRARY} ${QT_QTXML_LIBRARY} ${QT_QTOPENGL_LIBRARY} ${QT_QTUITOOLS_LIBRARY} ${QT_QTHELP_LIBRARY} ${QT_QTNETWORK_LIBRARY} ${QSCINTILLA_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} dataobject pointcloud)

# copy some folders and files during generate
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itom-packages         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. PATTERN ".svn" EXCLUDE PATTERN "__pycache__" EXCLUDE)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itomSettings/itomDefault.ini          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../itomSettings)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoDebugger.py        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..) # FILES_MATCHING PATTERN "*.py")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../itoFunctions.py       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..) # FILES_MATCHING PATTERN "*.py")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../common                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include PATTERN ".svn" EXCLUDE)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../plot                  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include PATTERN ".svn" EXCLUDE)
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK FILES_MATCHING PATTERN "*.h")
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindGLEW.cmake        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindGlut.cmake        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindFFTW.cmake        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindNumPy.cmake       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindOpenCV.cmake      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
#FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindQScintilla.cmake  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindXsd.cmake         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindXerces.cmake         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../ItomBuildMacros.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../FindITOM_SDK.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/itom_sdk.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include)

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../translation )
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../translation           DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.. FILES_MATCHING PATTERN "*.qm")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib )

#ADD a batch file for itom-release
if(MSVC)
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/../itom.bat.in ${CMAKE_CURRENT_BINARY_DIR}/../itom.bat )
ENDIF (MSVC)

UNSET(GIT_FOUND CACHE)
find_package(Git)
UNSET(SUBWCREV_FOUND CACHE)
FIND_PROGRAM(SUBWCREV_FOUND "subwcrev") #change to git

IF(CONSIDER_GIT_SVN AND GIT_FOUND AND PREFER_GIT)
    MESSAGE(STATUS "Using GIT as repository information")
    CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/pyGit2Header.py.in ${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py )
    ADD_CUSTOM_COMMAND (
        TARGET qitom
        PRE_BUILD
        COMMAND    ${PYTHON_EXECUTABLE}
        ARGS       "${CMAKE_CURRENT_BINARY_DIR}/pyGit2Header.py"
    )
ELSEIF(CONSIDER_GIT_SVN AND SUBWCREV_FOUND AND NOT(PREFER_GIT))
    MESSAGE(STATUS "Using SVN as repository information")
    ADD_CUSTOM_COMMAND (
        TARGET qitom
        PRE_BUILD
        COMMAND    subwcrev
        ARGS       " \"${CMAKE_CURRENT_SOURCE_DIR}\" \"${CMAKE_CURRENT_SOURCE_DIR}/version.tmpl\" \"${CMAKE_CURRENT_BINARY_DIR}/version.h\" "
    )
ELSE(CONSIDER_GIT_SVN AND SUBWCREV_FOUND AND NOT(PREFER_GIT))
    MESSAGE(STATUS "subwcrev (subversion tool) could not be found. Default version.h file is generated")
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/version_nosvn.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    FILE(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/version.h)
    FILE(RENAME ${CMAKE_CURRENT_BINARY_DIR}/version_nosvn.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)
ENDIF(CONSIDER_GIT_SVN AND GIT_FOUND AND PREFER_GIT)

MARK_AS_ADVANCED(SUBWCREV_FOUND) #only show in cmake gui if Advanced-checkbox is marked
