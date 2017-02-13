#########################################################################
#set general things
#########################################################################
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
OPTION(BUILD_OPENCV_SHARED "Use the shared version of OpenCV (default: ON)." ON)
SET (BUILD_QTVERSION "auto" CACHE STRING "auto: automatically detects Qt4 or Qt5, else use Qt4 or Qt5")
OPTION(BUILD_OPENMP_ENABLE "Use OpenMP parallelization if available. If TRUE, the definition USEOPENMP is set. This is only the case if OpenMP is generally available and if the build is release." ON)

IF(BUILD_OPENCV_SHARED)
    SET(OpenCV_STATIC FALSE)
ELSE(BUILD_OPENCV_SHARED)
    SET(OpenCV_STATIC TRUE)
ENDIF(BUILD_OPENCV_SHARED)

if (BUILD_TARGET64)
    set(CMAKE_SIZEOF_VOID_P 8)
else (BUILD_TARGET64)
    set(CMAKE_SIZEOF_VOID_P 4)
endif (BUILD_TARGET64)

IF (${CMAKE_MAJOR_VERSION} LESS 3)
    SET(CMAKE_VERSION_GE_030000 "FALSE")
    IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} STRGREATER 2.7)
        IF (${CMAKE_PATCH_VERSION} GREATER 11)
            MESSAGE (STATUS "CMake > 2.8.11")
            SET(CMAKE_VERSION_GT_020811 "TRUE") #CMAKE <= 2.8.10 (changes in FindQt4)
        ELSE (${CMAKE_PATCH_VERSION} GREATER 11)
            MESSAGE (STATUS "CMake 2.8.0 - 2.8.11")
            SET(CMAKE_VERSION_GT_020811 "FALSE") #CMAKE <= 2.8.10 (changes in FindQt4)
        ENDIF (${CMAKE_PATCH_VERSION} GREATER 11)
    ELSE (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} STRGREATER 2.7)
        MESSAGE (STATUS "CMake < 2.8")
        SET(CMAKE_VERSION_GT_020811 "FALSE") #CMAKE <= 2.8.10 (changes in FindQt4)
    ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} STRGREATER 2.7)
ELSE (${CMAKE_MAJOR_VERSION} LESS 3)
    SET(CMAKE_VERSION_GE_030000 "TRUE")
    SET(CMAKE_VERSION_GT_020811 "TRUE")
    MESSAGE (STATUS "CMake >= 3.0")
ENDIF (${CMAKE_MAJOR_VERSION} LESS 3)

#These are the overall pre-compiler directives for itom and its plugins:
#
#Windows:
#  x86: WIN32, _WIN32
#  x64: WIN32, _WIN32, WIN64, _WIN64
#Linux:
#       linux, Linux
#Apple:
#       __APPLE__
IF(CMAKE_HOST_WIN32)
    IF(BUILD_TARGET64)
        SET(CMAKE_CXX_FLAGS "/DWIN32 /D_WIN32 /DWIN64 /D_WIN64" ${CMAKE_CXX_FLAGS})
    ELSE(BUILD_TARGET64)
        SET(CMAKE_CXX_FLAGS "/DWIN32 /D_WIN32" ${CMAKE_CXX_FLAGS})
    ENDIF(BUILD_TARGET64)
ELSEIF(CMAKE_HOST_APPLE)
    ADD_DEFINITIONS(-D__APPLE__)
ELSEIF(CMAKE_HOST_UNIX) #this also includes apple, which is however already handled above
    ADD_DEFINITIONS(-DLinux -Dlinux)
ENDIF()


if(MSVC)
    ADD_DEFINITIONS(/MP)

    # set some optimization compiler flags
    # i.e.:
    #   - Ox full optimization (replaces standard O2 set by cmake)
    #   - Oi enable intrinsic functions
    #   - Ot favor fast code
    #   - Oy omit frame pointers
    #   - GL whole program optimization
    #   - GT fibre safe optimization
    #   - openmp enable openmp support, isn't enabled globally here as it breaks opencv
    SET ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
    SET ( CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
    SET ( CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    
    IF (NOT BUILD_TARGET64)
        #Disable safe SEH for Visual Studio, 32bit (this is necessary since some 3rd party libraries are compiled without /SAFESEH)
        SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        SET(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        SET(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
        SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        SET(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        SET(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
        SET(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        SET(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        SET(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        SET(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
    ENDIF (NOT BUILD_TARGET64)
ENDIF (MSVC)

#try to enable OpenMP (e.g. not available with VS Express)
find_package(OpenMP QUIET)

IF (OPENMP_FOUND)
    IF (BUILD_OPENMP_ENABLE)
        message(STATUS "OpenMP found and enabled for release compilation")
        SET ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -DUSEOPENMP" )
        SET ( CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_C_FLAGS} -DUSEOPENMP" )
    ELSE (BUILD_OPENMP_ENABLE)
        message(STATUS "OpenMP found but not enabled for release compilation")
    ENDIF (BUILD_OPENMP_ENABLE)
ELSE(OPENMP_FOUND)
    message(STATUS "OpenMP not found.")
ENDIF(OPENMP_FOUND)

# Begin: Remove duplicates compilation flags
separate_arguments(CMAKE_CXX_FLAGS)
list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

separate_arguments(CMAKE_CXX_FLAGS_RELEASE)
list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS_RELEASE)
string(REPLACE ";" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

separate_arguments(CMAKE_C_FLAGS_RELEASE)
list(REMOVE_DUPLICATES CMAKE_C_FLAGS_RELEASE)
string(REPLACE ";" " " CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
# End: Remove duplicates compilation flags


IF (BUILD_ITOMLIBS_SHARED OR ITOM_SDK_SHARED_LIBS)
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
ENDIF (BUILD_ITOMLIBS_SHARED OR ITOM_SDK_SHARED_LIBS)

MACRO (BUILD_PARALLEL_LINUX targetName)
  if(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "GNUCXX pipe flag enabled")
        set_target_properties(${targetName} PROPERTIES COMPILE_FLAGS "-pipe")
  ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDMACRO (BUILD_PARALLEL_LINUX)


MACRO (FIND_PACKAGE_QT SET_AUTOMOC)
    #call this macro to find the qt4 package (either version qt4 or qt5).
    #
    # call example FIND_PACKAGE_QT(ON Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools Designer)
    #
    # this will detect Qt with all given packages (packages given as Qt5 package names, Qt4 is automatically
    # back-translated) and automoc for Qt5 is set to ON (ignored for Qt4)
    #
    # If the CMAKE Config variable BUILD_QTVERSION is 'auto', Qt5 is detected and if not found Qt4 is detected
    # Force to find a specific Qt-branch by setting BUILD_QTVERSION to either 'Qt4' or 'Qt5'
    #
    # For Qt5.0 a specific load mechanism is used, since find_package(Qt5 COMPONENTS...) is only available for Qt5 > 5.0
    #
    SET(Components ${ARGN}) #all arguments after SetAutomoc are components for Qt
    SET(QT_COMPONENTS ${ARGN})
    SET(QT5_LIBRARIES "")

    IF(${BUILD_QTVERSION} STREQUAL "Qt4")
        SET(DETECT_QT5 FALSE)
    ELSEIF(${BUILD_QTVERSION} STREQUAL "Qt5")
        if (CMAKE_VERSION VERSION_GREATER 2.8.7)
            if (POLICY CMP0020)
                cmake_policy(SET CMP0020 NEW)
            ENDIF (POLICY CMP0020)
        ELSE ()
            MESSAGE(SEND_ERROR "with cmake <= 2.8.7 Qt5 cannot be detected.")
        ENDIF ()
        SET(DETECT_QT5 TRUE)
    ELSEIF(${BUILD_QTVERSION} STREQUAL "auto")
        if (CMAKE_VERSION VERSION_GREATER 2.8.7)
            if (POLICY CMP0020)
                cmake_policy(SET CMP0020 NEW)
            ENDIF (POLICY CMP0020)
            SET(DETECT_QT5 TRUE)
        ELSE ()
            MESSAGE(STATUS "with cmake <= 2.8.7 no Qt4 auto-detection is possible. Search for Qt4")
            SET(DETECT_QT5 FALSE)
        ENDIF ()
    ELSE()
        MESSAGE(SEND_ERROR "wrong value for BUILD_QTVERSION. auto, Qt4 or Qt5 allowed")
    ENDIF()
    set (QT5_FOUND FALSE)
        
    IF (DETECT_QT5)
        #TRY TO FIND QT5
        find_package(Qt5 COMPONENTS Core QUIET)
        
        IF (${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
            #maybe Qt5.0 is installed that does not support the overall FindQt5 script
            find_package(Qt5Core QUIET)
            IF (NOT Qt5Core_FOUND)
                IF(${BUILD_QTVERSION} STREQUAL "auto")
                    SET(DETECT_QT5 FALSE)
                ELSE()
                    MESSAGE(SEND_ERROR "Qt5 could not be found on this computer")
                ENDIF()
            ELSE (NOT Qt5Core_FOUND)
                set(QT5_FOUND TRUE)
                
                IF (WIN32)
                    find_package(WindowsSDK REQUIRED)
                    set (CMAKE_PREFIX_PATH "${WINDOWSSDK_PREFERRED_DIR}/Lib/")
                    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${WINDOWSSDK_PREFERRED_DIR}/Lib/)
                ENDIF (WIN32)
                
                set(QT5_FOUND TRUE)
                
                IF (${SET_AUTOMOC})
                    set(CMAKE_AUTOMOC ON)
                ELSE (${SET_AUTOMOC})
                    set(CMAKE_AUTOMOC OFF)
                ENDIF (${SET_AUTOMOC})
                
                FOREACH (comp ${Components})
                    message(STATUS "FIND_PACKAGE FOR COMPONENT ${comp}")
                    find_package(Qt5${comp} REQUIRED)
                    IF (${comp} STREQUAL "Widgets")
                        add_definitions(${Qt5Widgets_DEFINITIONS})
                        SET(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                    ELSEIF (${comp} STREQUAL "LinguistTools")
                        #it is not possible to link Qt5::LinguistTools since it does not exist
                    ELSE ()
                        SET(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                    ENDIF ()
                ENDFOREACH (comp)
            ENDIF (NOT Qt5Core_FOUND)
            
        ELSE (${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
            #QT5 could be found with component based find_package command
            IF (WIN32)
              find_package(WindowsSDK REQUIRED)
              set (CMAKE_PREFIX_PATH "${WINDOWSSDK_PREFERRED_DIR}/Lib/")
              set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${WINDOWSSDK_PREFERRED_DIR}/Lib/)
            ENDIF (WIN32)
            
            find_package(Qt5 COMPONENTS ${Components} REQUIRED)
            set(QT5_FOUND TRUE)
            
            IF (${SET_AUTOMOC})
                set(CMAKE_AUTOMOC ON)
            ELSE (${SET_AUTOMOC})
                set(CMAKE_AUTOMOC OFF)
            ENDIF (${SET_AUTOMOC})
            
            FOREACH (comp ${Components})
                IF (${comp} STREQUAL "Widgets")
                    add_definitions(${Qt5Widgets_DEFINITIONS})
                    SET(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                ELSEIF (${comp} STREQUAL "LinguistTools")
                    #it is not possible to link Qt5::LinguistTools since it does not exist
                ELSE ()
                    SET(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                ENDIF ()
                
            ENDFOREACH (comp)
        ENDIF (${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
        
        IF (Qt5Core_FOUND)
            # These variables are not defined with Qt5 CMake modules
            SET(QT_BINARY_DIR "${_qt5Core_install_prefix}/bin")
            SET(QT_LIBRARY_DIR "${_qt5Core_install_prefix}/lib")
        ENDIF (Qt5Core_FOUND)
        
    ENDIF (DETECT_QT5)

    IF (NOT DETECT_QT5)
        #TRY TO FIND QT4
        SET(QT5_FOUND FALSE)
        find_package(Qt4 REQUIRED)
        SET (QT_USE_CORE TRUE)
        
        FOREACH (comp ${Components})
            IF (${comp} STREQUAL "OpenGL")
                SET (QT_USE_QTOPENGL TRUE)
            ELSEIF (${comp} STREQUAL "Core")
                SET (QT_USE_QTCORE TRUE)
            ELSEIF (${comp} STREQUAL "Designer")
                SET (QT_USE_QTDESIGNER TRUE)
            ELSEIF (${comp} STREQUAL "Xml")
                SET (QT_USE_QTXML TRUE)
            ELSEIF (${comp} STREQUAL "Svg")
                SET (QT_USE_QTSVG TRUE)
            ELSEIF (${comp} STREQUAL "Sql")
                SET (QT_USE_QTSQL TRUE)
            ELSEIF (${comp} STREQUAL "Network")
                SET (QT_USE_QTNETWORK TRUE)
            ELSEIF (${comp} STREQUAL "UiTools")
                SET (QT_USE_QTUITOOLS TRUE)
            ELSEIF (${comp} STREQUAL "Widgets")
                SET (QT_USE_QTGUI TRUE)
            ELSEIF (${comp} STREQUAL "PrintSupport")
            ELSEIF (${comp} STREQUAL "LinguistTools")
            ELSEIF (${comp} STREQUAL "WebEngine")
            ELSEIF (${comp} STREQUAL "WebEngineWidgets")
            ELSEIF (${comp} STREQUAL "Concurrent")
            ELSE ()
                message (SEND_ERROR "Qt component ${comp} unknown")
            ENDIF ()
        ENDFOREACH (comp)
        
        INCLUDE(${QT_USE_FILE})
    ENDIF (NOT DETECT_QT5)
    
    ADD_DEFINITIONS(${QT_DEFINITIONS})
ENDMACRO (FIND_PACKAGE_QT)


#use this macro in order to generate and/or reconfigure the translation of any plugin or designer plugin.
#
# example:
# set (FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_ui}) #adds all files to the list of files that are searched for strings to translate
# PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")
#
# Hereby, ITOM_LANGUAGES is a semicolon-separeted string with different languages, e.g. "de;fr"
# EXISTING_TRANSLATION_FILES is an option (ON/OFF) that decides whether the qm-file should only be build from the existing ts-file or IF the ts-file
# is reconfigured with respect to the given files in FILES_TO_TRANSLATE.
#
# Please note, that you need to add the resulting QM_FILES to the copy-list using the macro
# ADD_QM_FILES_TO_COPY_LIST or ADD_DESIGNER_QM_FILES_TO_COPY_LIST
#
MACRO (PLUGIN_TRANSLATION qm_files target force_translation_update existing_translation_files languages files_to_translate)
    SET(TRANSLATIONS_FILES)
    SET(TRANSLATION_OUTPUT_FILES)
    SET(QMFILES)

    IF (${force_translation_update})
        IF (QT5_FOUND)
            QT5_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target} ${languages} ${files_to_translate})
        ELSE (QT5_FOUND)
            QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target} ${languages} ${files_to_translate})
        ENDIF (QT5_FOUND)
        
        add_custom_target (_${target}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
        add_dependencies(${target} _${target}_translation)
        
        IF (QT5_FOUND)
            QT5_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${TRANSLATIONS_FILES})
        ELSE (QT5_FOUND)
            QT4_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${TRANSLATIONS_FILES})
        ENDIF (QT5_FOUND)
    ELSE (${force_translation_update})
        IF (QT5_FOUND)
            QT5_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${existing_translation_files})
        ELSE (QT5_FOUND)
            QT4_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${existing_translation_files})
        ENDIF (QT5_FOUND)
    ENDIF (${force_translation_update})
    
    SET(${qm_files} ${${qm_files}} ${QMFILES})
    
ENDMACRO (PLUGIN_TRANSLATION)


###########################################################################
# useful macros
###########################################################################

# using custom macro for qtCreator compatibility, i.e. put ui files into GeneratedFiles/ folder
# This macro is copied and adapted from Qt4Macros.cmake (Copyright Kitware, Inc.).
MACRO (QT4_WRAP_UI_ITOM outfiles)
    
    IF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
        QT4_EXTRACT_OPTIONS(ui_files ui_options ui_target ${ARGN})
    ELSE((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
        QT4_EXTRACT_OPTIONS(ui_files ui_options ${ARGN})
    ENDIF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles)

    FOREACH (it ${ui_files})
        GET_FILENAME_COMPONENT(outfile ${it} NAME_WE)
        GET_FILENAME_COMPONENT(infile ${it} ABSOLUTE)
        SET(outfile ${CMAKE_CURRENT_BINARY_DIR}/ui_${outfile}.h) # Here we set output
        ADD_CUSTOM_COMMAND(OUTPUT ${outfile}
            COMMAND ${QT_UIC_EXECUTABLE}
            ARGS ${ui_options} -o ${outfile} ${infile}
            MAIN_DEPENDENCY ${infile})
        SET(${outfiles} ${${outfiles}} ${outfile})
    ENDFOREACH (it)
    SOURCE_GROUP("GeneratedFiles" FILES ${${outfiles}})
ENDMACRO (QT4_WRAP_UI_ITOM)


# This macro is copied and adapted from Qt4Macros.cmake (Copyright Kitware, Inc.).
MACRO (QT4_WRAP_CPP_ITOM outfiles )
    # get include dirs
    QT4_GET_MOC_FLAGS(moc_flags)
    
    IF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
        QT4_EXTRACT_OPTIONS(moc_files moc_options moc_target ${ARGN})
    ELSE((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
        QT4_EXTRACT_OPTIONS(moc_files moc_options ${ARGN})
    ENDIF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))

    foreach (it ${moc_files})
        GET_FILENAME_COMPONENT(it ${it} ABSOLUTE)
        QT4_MAKE_OUTPUT_FILE(${it} moc_ cxx outfile)
        
        IF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
            QT4_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags}" "${moc_options}" "${moc_target}")
        ELSE((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))
            QT4_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags}" "${moc_options}")
        ENDIF((${CMAKE_VERSION_GT_020811} STREQUAL "TRUE"))

        set(${outfiles} ${${outfiles}} ${outfile})
    endforeach()

    SOURCE_GROUP("GeneratedFiles" FILES ${${outfiles}})
ENDMACRO ()


# This macro is copied and adapted from Qt4Macros.cmake (Copyright Kitware, Inc.).
MACRO(QT4_CREATE_TRANSLATION_ITOM outputFiles tsFiles target languages)
    IF(${CMAKE_VERSION_GT_020811})
        QT4_EXTRACT_OPTIONS(_lupdate_files _lupdate_options _lupdate_target ${ARGN})
    ELSE(${CMAKE_VERSION_GT_020811})
        QT4_EXTRACT_OPTIONS(_lupdate_files _lupdate_options ${ARGN})
    ENDIF(${CMAKE_VERSION_GT_020811})
    
    set(_my_sources)
    set(_my_dirs)
    set(_my_tsfiles)
    set(_ts_pro)

    #reset tsFiles
    set(${tsFiles} "")

    foreach (_file ${_lupdate_files})
        get_filename_component(_ext ${_file} EXT)
        get_filename_component(_abs_FILE ${_file} ABSOLUTE)
        IF(_ext MATCHES "ts")
            list(APPEND _my_tsfiles ${_abs_FILE})
        ELSE()
            IF(NOT _ext)
                list(APPEND _my_dirs ${_abs_FILE})
            ELSE()
                list(APPEND _my_sources ${_abs_FILE})
            ENDIF()
        ENDIF()
    endforeach()

    foreach( _lang ${${languages}})
        set(_tsFile ${CMAKE_CURRENT_SOURCE_DIR}/translation/${target}_${_lang}.ts)
        get_filename_component(_ext ${_tsFile} EXT)
        get_filename_component(_abs_FILE ${_tsFile} ABSOLUTE)
        IF(EXISTS ${_abs_FILE})
            list(APPEND _my_tsfiles ${_abs_FILE})
        ELSE()
            #create new ts file
            add_custom_command(OUTPUT ${_abs_FILE}_new
                COMMAND ${QT_LUPDATE_EXECUTABLE}
                ARGS ${_lupdate_options} ${_my_dirs} -locations relative -no-ui-lines -target-language ${_lang} -ts ${_abs_FILE}
                DEPENDS ${_my_sources} VERBATIM)
            list(APPEND _my_tsfiles ${_abs_FILE})
            set(${outputFiles} ${${outputFiles}} ${_abs_FILE}_new) #add output file for custom command to outputFiles list
        ENDIF()
    endforeach()

    set(${tsFiles} ${${tsFiles}} ${_my_tsfiles}) #add translation files (*.ts) to tsFiles list

    foreach(_ts_file ${_my_tsfiles})
        IF(_my_sources)
            # make a .pro file to call lupdate on, so we don't make our commands too
            # long for some systems
            get_filename_component(_ts_name ${_ts_file} NAME_WE)
            set(_ts_pro ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_ts_name}_lupdate.pro)
            set(_pro_srcs)
            foreach(_pro_src ${_my_sources})
                set(_pro_srcs "${_pro_srcs} \"${_pro_src}\"")
            endforeach()
            set(_pro_includes)
            get_directory_property(_inc_DIRS INCLUDE_DIRECTORIES)
            foreach(_pro_include ${_inc_DIRS})
                get_filename_component(_abs_include "${_pro_include}" ABSOLUTE)
                set(_pro_includes "${_pro_includes} \"${_abs_include}\"")
            endforeach()
            file(WRITE ${_ts_pro} "SOURCES = ${_pro_srcs}\nINCLUDEPATH = ${_pro_includes}\n")
        ENDIF()
        add_custom_command(OUTPUT ${_ts_file}_update
            COMMAND ${QT_LUPDATE_EXECUTABLE}
            ARGS ${_lupdate_options} ${_ts_pro} ${_my_dirs} -locations relative -no-ui-lines -ts ${_ts_file}
            DEPENDS ${_my_sources} ${_ts_pro} VERBATIM)
        set(${outputFiles} ${${outputFiles}} ${_ts_file}_update) #add output file for custom command to outputFiles list
    endforeach()
ENDMACRO()


MACRO(QT5_CREATE_TRANSLATION_ITOM outputFiles tsFiles target languages)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)

    cmake_parse_arguments(_LUPDATE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(_lupdate_files ${_LUPDATE_UNPARSED_ARGUMENTS})
    set(_lupdate_options ${_LUPDATE_OPTIONS})

    set(_my_sources)
    set(_my_tsfiles)

    #reset tsFiles
    set(${tsFiles} "")

    foreach(_file ${_lupdate_files})
        get_filename_component(_ext ${_file} EXT)
        get_filename_component(_abs_FILE ${_file} ABSOLUTE)
        if(_ext MATCHES "ts")
            list(APPEND _my_tsfiles ${_abs_FILE})
        else()
            list(APPEND _my_sources ${_abs_FILE})
        endif()
    endforeach()
    
    foreach( _lang ${${languages}})
        set(_tsFile ${CMAKE_CURRENT_SOURCE_DIR}/translation/${target}_${_lang}.ts)
        get_filename_component(_ext ${_tsFile} EXT)
        get_filename_component(_abs_FILE ${_tsFile} ABSOLUTE)
        
        
        IF(EXISTS ${_abs_FILE})
            list(APPEND _my_tsfiles ${_abs_FILE})
        ELSE()
            #create new ts file
            add_custom_command(OUTPUT ${_abs_FILE}_new
                COMMAND ${Qt5_LUPDATE_EXECUTABLE}
                ARGS ${_lupdate_options} ${_my_dirs} -locations relative -no-ui-lines -target-language ${_lang} -ts ${_abs_FILE}
                DEPENDS ${_my_sources} VERBATIM)
            list(APPEND _my_tsfiles ${_abs_FILE})
            set(${outputFiles} ${${outputFiles}} ${_abs_FILE}_new) #add output file for custom command to outputFiles list
        ENDIF()
    endforeach()
    
    set(${tsFiles} ${${tsFiles}} ${_my_tsfiles}) #add translation files (*.ts) to tsFiles list
    
    
    foreach(_ts_file ${_my_tsfiles})
        if(_my_sources)
            # make a list file to call lupdate on, so we don't make our commands too
            # long for some systems
            get_filename_component(_ts_name ${_ts_file} NAME_WE)
            set(_ts_lst_file "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_ts_name}_lst_file")
            set(_lst_file_srcs)
            foreach(_lst_file_src ${_my_sources})
                set(_lst_file_srcs "${_lst_file_src}\n${_lst_file_srcs}")
            endforeach()

            get_directory_property(_inc_DIRS INCLUDE_DIRECTORIES)
            foreach(_pro_include ${_inc_DIRS})
                # some include pathes somehow disturb lupdate, such that it requires a long time to finish.
                # Therefore, they are excluded from the lupdate include list
                string(FIND ${_pro_include} "boost" pos)
                if(pos LESS 0)
                    get_filename_component(_abs_include "${_pro_include}" ABSOLUTE)
                    set(_lst_file_srcs "-I${_pro_include}\n${_lst_file_srcs}")
                endif()
            endforeach()

            file(WRITE ${_ts_lst_file} "${_lst_file_srcs}")
        endif()
        add_custom_command(OUTPUT ${_ts_file}_update
            COMMAND ${Qt5_LUPDATE_EXECUTABLE}
            ARGS ${_lupdate_options} "@${_ts_lst_file}" -ts "${_ts_file}"
            DEPENDS ${_my_sources} ${_ts_lst_file} VERBATIM)
        set(${outputFiles} ${${outputFiles}} ${_ts_file}_update) #add output file for custom command to outputFiles list
    endforeach()
#    QT5_ADD_TRANSLATION_ITOM(${_qm_files} ${_my_tsfiles})
#    set(${_qm_files} ${${_qm_files}} PARENT_SCOPE)
ENDMACRO()


MACRO(QT4_ADD_TRANSLATION_ITOM _qm_files output_location target)
    foreach (_current_FILE ${ARGN})
        get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)
        get_filename_component(qm ${_abs_FILE} NAME_WE)

        file(MAKE_DIRECTORY "${output_location}")
        set(qm "${output_location}/${qm}.qm")

        add_custom_command(TARGET ${target}
            PRE_BUILD
            COMMAND ${QT_LRELEASE_EXECUTABLE}
            ARGS ${_abs_FILE} -qm ${qm}
            VERBATIM
            )

        set(${_qm_files} ${${_qm_files}} ${qm})
    endforeach ()
ENDMACRO()


MACRO(QT5_ADD_TRANSLATION_ITOM _qm_files output_location target)
    foreach (_current_FILE ${ARGN})
        get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)
        get_filename_component(qm ${_abs_FILE} NAME_WE)

        file(MAKE_DIRECTORY "${output_location}")
        set(qm "${output_location}/${qm}.qm")
        add_custom_command(TARGET ${target}
            PRE_BUILD
            COMMAND ${Qt5_LRELEASE_EXECUTABLE}
            ARGS ${_abs_FILE} -qm ${qm}
            VERBATIM
            )

        set(${_qm_files} ${${_qm_files}} ${qm})
    endforeach ()
ENDMACRO()


#this macro only generates the moc-file but does not compile it, since it is included in another source file.
#this comes from the ctkCommon project
#Creates a rule to run moc on infile and create outfile. Use this IF for some reason QT5_WRAP_CPP() 
#isn't appropriate, e.g. because you need a custom filename for the moc file or something similar.
macro(QT4_GENERATE_MOCS)
    foreach(file ${ARGN})
        set(moc_file moc_${file})
        QT4_GENERATE_MOC(${file} ${moc_file})

        get_filename_component(source_name ${file} NAME_WE)
        get_filename_component(source_ext ${file} EXT)
        IF(${source_ext} MATCHES "\\.[hH]")
            IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cpp)
                set(source_ext .cpp)
            ELSEIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cxx)
                set(source_ext .cxx)
            ENDIF()
        ENDIF()
        set_property(SOURCE ${source_name}${source_ext} APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${moc_file})
    endforeach()
endmacro()


#this macro only generates the moc-file but does not compile it, since it is included in another source file.
#this comes from the ctkCommon project
#Creates a rule to run moc on infile and create outfile. Use this IF for some reason QT5_WRAP_CPP() 
#isn't appropriate, e.g. because you need a custom filename for the moc file or something similar.
macro(QT5_GENERATE_MOCS)
    foreach(file ${ARGN})
        set(moc_file moc_${file})
        QT5_GENERATE_MOC(${file} ${moc_file})

        get_filename_component(source_name ${file} NAME_WE)
        get_filename_component(source_ext ${file} EXT)
        IF(${source_ext} MATCHES "\\.[hH]")
            IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cpp)
                set(source_ext .cpp)
            ELSEIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_name}.cxx)
                set(source_ext .cxx)
            ENDIF()
        ENDIF()
        set_property(SOURCE ${source_name}${source_ext} APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${moc_file})
    endforeach()
endmacro()


#use this macro in order to append to the sources and destinations
#list the library file for your designer plugin, that is finally
#copied to the designer folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# ADD_DESIGNERLIBRARY_TO_COPY_LIST(targetNameOfYourDesignerPlugin COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call POST_BUILD_COPY_FILES-macro, to initialize the copying.
#
MACRO (ADD_DESIGNERLIBRARY_TO_COPY_LIST target sources destinations)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    LIST(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    LIST(APPEND ${destinations} ${ITOM_APP_DIR}/designer)
    
    LIST(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
    LIST(APPEND ${destinations} ${ITOM_APP_DIR}/designer)    
ENDMACRO (ADD_DESIGNERLIBRARY_TO_COPY_LIST target sources destinations)


#use this macro in order to append to the sources and destinations
#list the header files for your designer plugin, that is finally
#copied to the designer/{$target} folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# ADD_DESIGNERLIBRARY_TO_COPY_LIST(targetNameOfYourDesignerPlugin COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call POST_BUILD_COPY_FILES-macro, to initialize the copying.
#
MACRO (ADD_DESIGNERHEADER_TO_COPY_LIST target headerfiles sources destinations)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    
    foreach(_hfile ${${headerfiles}})
        LIST(APPEND ${sources} ${_hfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        LIST(APPEND ${destinations} ${ITOM_APP_DIR}/designer/${target})
    endforeach()
ENDMACRO (ADD_DESIGNERHEADER_TO_COPY_LIST)


#use this macro in order to append to the sources and destinations
#list the library file for your itom plugin, that is finally
#copied to the plugin/target folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# ADD_PLUGINLIBRARY_TO_COPY_LIST(targetNameOfYourPlugin COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call POST_BUILD_COPY_FILES-macro, to initialize the copying.
#
MACRO (ADD_PLUGINLIBRARY_TO_COPY_LIST target sources destinations)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    
    #GET_TARGET_PROPERTY(VAR_LOCATION ${target} LOCATION)
    #STRING(REGEX REPLACE "\\(Configuration\\)" "<CONFIGURATION>" VAR_LOCATION ${VAR_LOCATION})
    #SET(VAR_LOCATION "$<TARGET_FILE:${target}>")
    LIST(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    
    LIST(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target})
ENDMACRO (ADD_PLUGINLIBRARY_TO_COPY_LIST)


MACRO (ADD_QM_FILES_TO_COPY_LIST target qm_files sources destinations)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    
    foreach(_qmfile ${${qm_files}})
        LIST(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        LIST(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target}/translation)
    endforeach()
ENDMACRO (ADD_QM_FILES_TO_COPY_LIST)


MACRO (ADD_DESIGNER_QM_FILES_TO_COPY_LIST qm_files sources destinations)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    
    foreach(_qmfile ${${qm_files}})
        LIST(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        LIST(APPEND ${destinations} ${ITOM_APP_DIR}/designer/translation)
    endforeach()
ENDMACRO (ADD_DESIGNER_QM_FILES_TO_COPY_LIST)


MACRO (ADD_OUTPUTLIBRARY_TO_SDK_LIB target sources destinations)
    
    IF(${ITOM_SDK_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_SDK_DIR is not indicated")
    ENDIF()
    
    IF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
      SET(SDK_PLATFORM "x86")
    ELSE ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
      SET(SDK_PLATFORM "x64")
    ENDIF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    
    IF(MSVC12)
        SET(SDK_COMPILER "vc12")
    ELSEIF(MSVC11)
        SET(SDK_COMPILER "vc11")
    ELSEIF(MSVC10)
        SET(SDK_COMPILER "vc10")
    ELSEIF(MSVC9)
        SET(SDK_COMPILER "vc9")
    ELSEIF(MSVC8)
        SET(SDK_COMPILER "vc8")
    ELSEIF(MSVC)
        SET(SDK_COMPILER "vc${MSVC_VERSION}")
    ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
        SET(SDK_COMPILER "gnucxx")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        SET(SDK_COMPILER "clang")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(SDK_COMPILER "gnucxx")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        SET(SDK_COMPILER "intel")
    ELSEIF(APPLE)
        SET(SDK_COMPILER "osx_default")
    ELSE(MSVC12)
        SET(SDK_COMPILER "unknown")
    ENDIF(MSVC12)
    
    SET( destination "${ITOM_SDK_DIR}/lib/${SDK_COMPILER}_${SDK_PLATFORM}" )
    
    LIST(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    LIST(APPEND ${destinations} ${destination}) 
ENDMACRO (ADD_OUTPUTLIBRARY_TO_SDK_LIB target sources destinations)


MACRO (ADD_LIBRARY_TO_APPDIR_AND_SDK target sources destinations)

    IF(${ITOM_SDK_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_SDK_DIR is not indicated")
    ENDIF()

    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    ENDIF()

    IF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(SDK_PLATFORM "x86")
    ELSE ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(SDK_PLATFORM "x64")
    ENDIF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )

    IF(MSVC12)
        SET(SDK_COMPILER "vc12")
    ELSEIF(MSVC11)
        SET(SDK_COMPILER "vc11")
    ELSEIF(MSVC10)
        SET(SDK_COMPILER "vc10")
    ELSEIF(MSVC9)
        SET(SDK_COMPILER "vc9")
    ELSEIF(MSVC8)
        SET(SDK_COMPILER "vc8")
    ELSEIF(MSVC)
        SET(SDK_COMPILER "vc${MSVC_VERSION}")
    ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
        SET(SDK_COMPILER "gnucxx")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        SET(SDK_COMPILER "clang")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(SDK_COMPILER "gnucxx")
    ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        SET(SDK_COMPILER "intel")
    ELSEIF(APPLE)
        SET(SDK_COMPILER "osx_default")
    ELSE(MSVC12)
        SET(SDK_COMPILER "unknown")
    ENDIF(MSVC12)

    SET( sdk_destination "${ITOM_SDK_DIR}/lib/${SDK_COMPILER}_${SDK_PLATFORM}" )

    IF(BUILD_ITOMLIBS_SHARED)
        #copy library (dll) to app-directory and linker library (lib) to sdk_destination
        LIST(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
        LIST(APPEND ${destinations} ${sdk_destination})

        LIST(APPEND ${sources} "$<TARGET_FILE:${target}>")
        LIST(APPEND ${destinations} ${ITOM_APP_DIR})
    ELSE(BUILD_ITOMLIBS_SHARED)
        #copy linker library (lib) to sdk_destination
        LIST(APPEND ${sources} "$<TARGET_FILE:${target}>")
        LIST(APPEND ${destinations} ${sdk_destination})
    ENDIF(BUILD_ITOMLIBS_SHARED)
ENDMACRO (ADD_LIBRARY_TO_APPDIR_AND_SDK target sources destinations)


MACRO (POST_BUILD_COPY_FILES target sources destinations)
    list(LENGTH ${sources} temp)
    math(EXPR len1 "${temp} - 1")
    list(LENGTH ${destinations} temp)
    math(EXPR len2 "${temp} - 1")
    #message(STATUS "sources LEN: ${len1}")
    #message(STATUS "destinations LEN: ${len2}")

    IF( NOT len1 EQUAL len2 )
        message(SEND_ERROR "POST_BUILD_COPY_FILES len(sources) must be equal to len(destinations)")
    ENDIF( NOT len1 EQUAL len2 )
    
    SET (destPathes "")
    foreach(dest ${${destinations}})
        #IF dest is a full name to a file:
        #GET_FILENAME_COMPONENT(destPath ${dest} PATH)
        #LIST(APPEND destPathes ${destPath})
        LIST(APPEND destPathes ${dest})
    endforeach(dest ${${destinations}})
    LIST(REMOVE_DUPLICATES destPathes)
    
#    message(STATUS "destPathes: ${destPathes}")
    
    #try to create all pathes
    foreach(destPath ${destPathes})
        #first try to create the directory
        ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${destPath}"
        )
    endforeach(destPath ${destPathes})
    
    foreach(val RANGE ${len1})
        list(GET ${sources} ${val} val1)
        list(GET ${destinations} ${val} val2)
#        message(STATUS "POST_BUILD: COPY ${val1} TO ${val2}")
        
        ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD                 # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different                 # which executes "cmake - E copy_if_different..."
                "${val1}"                                                 # <--this is in-file
                "${val2}"                                                # <--this is out-file path
        )
    endforeach()
ENDMACRO (POST_BUILD_COPY_FILES target sources destinations)


MACRO (POST_BUILD_COPY_FILE_TO_LIB_FOLDER target sources)
    IF(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    ENDIF()
    
    list(LENGTH ${sources} temp)
    math(EXPR len1 "${temp} - 1")
    
    #message(STATUS "sources LEN: ${len1}")
    #message(STATUS "destinations LEN: ${len2}")
    
    #create lib folder (for safety only, IF it does not exist some cmake versions do not copy the files in the 
    #desired way using copy_if_different below
    ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${ITOM_APP_DIR}/lib"
    )

    foreach(val RANGE ${len1})
        list(GET ${sources} ${val} val1)
        #message(STATUS "POST_BUILD: COPY ${val1} TO ${ITOM_APP_DIR}/lib")
        
        ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD                 # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different                 # which executes "cmake - E copy_if_different..."
                "${val1}"                                                 # <--this is in-file
                "${ITOM_APP_DIR}/lib"                                                # <--this is out-file path
        )
    endforeach()
ENDMACRO (POST_BUILD_COPY_FILE_TO_LIB_FOLDER target sources)


MACRO (ADD_SOURCE_GROUP subfolder)
    FILE(GLOB GROUP_FILES_H
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.ui
    )
    SOURCE_GROUP("Header Files\\${subfolder}" FILES ${GROUP_FILES_H})
    
    FILE(GLOB GROUP_FILES_S
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.cpp
    )
    SOURCE_GROUP("Source Files\\${subfolder}" FILES ${GROUP_FILES_S})
ENDMACRO (ADD_SOURCE_GROUP subfolder)


#some unused macros

MACRO(COPY_FILE_IF_CHANGED in_file out_file target)
#  MESSAGE(STATUS "copy command: " ${in_file} " " ${out_file} " " ${target})
    IF(${in_file} IS_NEWER_THAN ${out_file})    
  #    message("COpying file: ${in_file} to: ${out_file}")
        ADD_CUSTOM_COMMAND (
    #    OUTPUT     ${out_file}
            TARGET ${target}
            POST_BUILD
            COMMAND    ${CMAKE_COMMAND}
            ARGS       -E copy ${in_file} ${out_file}
    #    DEPENDS     qitom
    #    DEPENDS    ${in_file}
    #    MAIN_DEPENDENCY ${in_file}
        )
    ENDIF(${in_file} IS_NEWER_THAN ${out_file})
ENDMACRO(COPY_FILE_IF_CHANGED)


MACRO(COPY_FILE_INTO_DIRECTORY_IF_CHANGED in_file out_dir target)
    GET_FILENAME_COMPONENT(file_name ${in_file} NAME) 
    COPY_FILE_IF_CHANGED(${in_file} ${out_dir}/${file_name} ${target})
ENDMACRO(COPY_FILE_INTO_DIRECTORY_IF_CHANGED)


#Copies all the files from in_file_list into the out_dir. 
# sub-trees are ignored (files are stored in same out_dir)
MACRO(COPY_FILES_INTO_DIRECTORY_IF_CHANGED in_file_list out_dir target)
    FOREACH(in_file ${in_file_list})
        COPY_FILE_INTO_DIRECTORY_IF_CHANGED(${in_file} ${out_dir} ${target})
    ENDFOREACH(in_file)     
ENDMACRO(COPY_FILES_INTO_DIRECTORY_IF_CHANGED)


#Copy all files and directories in in_dir to out_dir. 
# Subtrees remain intact.
MACRO(COPY_DIRECTORY_IF_CHANGED in_dir out_dir target pattern recurse)
    #message("Copying directory ${in_dir}")
    FILE(${recurse} in_file_list ${in_dir}/${pattern})
    FOREACH(in_file ${in_file_list})
        IF(NOT ${in_file} MATCHES ".*svn.*")
            STRING(REGEX REPLACE ${in_dir} ${out_dir} out_file ${in_file}) 
            COPY_FILE_IF_CHANGED(${in_file} ${out_file} ${target})
        ENDIF(NOT ${in_file} MATCHES ".*svn.*")
    ENDFOREACH(in_file)     
ENDMACRO(COPY_DIRECTORY_IF_CHANGED)


MACRO(PLUGIN_DOCUMENTATION target main_document) #main_document without .rst at the end
    SET(PLUGIN_NAME ${target})
    SET(PLUGIN_DOC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    SET(PLUGIN_DOC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/build)
    SET(PLUGIN_DOC_INSTALL_DIR ${ITOM_APP_DIR}/plugins/${target}/docs)
    SET(PLUGIN_DOC_MAIN ${main_document})
    configure_file(${ITOM_SDK_DIR}/docs/pluginDoc/plugin_doc_config.cfg.in ${CMAKE_CURRENT_BINARY_DIR}/docs/plugin_doc_config.cfg)
ENDMACRO(PLUGIN_DOCUMENTATION)


# OSX ONLY: Copy files from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS.
IF(APPLE)
    MACRO(COPY_TO_BUNDLE target srcDir destDir)
        FILE(GLOB_RECURSE templateFiles RELATIVE ${srcDir} ${srcDir}/*)
        FOREACH(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            IF(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            ENDIF(NOT IS_DIRECTORY ${srcTemplatePath})
        ENDFOREACH(templateFile)
    ENDMACRO(COPY_TO_BUNDLE)
ENDIF(APPLE)


# OSX ONLY: Copy files from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS.
IF(APPLE)
    MACRO(COPY_TO_BUNDLE_NONREC target srcDir destDir)
        FILE(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
        FOREACH(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            IF(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            ENDIF(NOT IS_DIRECTORY ${srcTemplatePath})
        ENDFOREACH(templateFile)
    ENDMACRO(COPY_TO_BUNDLE_NONREC)
ENDIF(APPLE)


# OSX ONLY: Copy files of certain type from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS
IF(APPLE)
    MACRO(COPY_TYPE_TO_BUNDLE target srcDir destDir type)
        FILE(GLOB_RECURSE templateFiles RELATIVE ${srcDir} ${srcDir}/*${type})
        FOREACH(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            IF(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            ENDIF(NOT IS_DIRECTORY ${srcTemplatePath})
        ENDFOREACH(templateFile)
    ENDMACRO(COPY_TYPE_TO_BUNDLE)
ENDIF(APPLE)


# OSX ONLY: Copy files of certain type from source directory to destination directory in app bundle, substituting any
# variables (NON-RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS
IF(APPLE)
    MACRO(COPY_TYPE_TO_BUNDLE_NONREC target srcDir destDir type)
        FILE(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*${type})
        FOREACH(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            IF(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            ENDIF(NOT IS_DIRECTORY ${srcTemplatePath})
        ENDFOREACH(templateFile)
    ENDMACRO(COPY_TYPE_TO_BUNDLE_NONREC)
ENDIF(APPLE)