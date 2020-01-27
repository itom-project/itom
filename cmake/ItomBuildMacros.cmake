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
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

#########################################################################
#set general things
#########################################################################
cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
set(BUILD_QTVERSION "auto" CACHE STRING "currently only Qt5 is supported. Set this value to 'auto' in order to auto-detect the correct Qt version or set it to 'Qt5' to hardly select Qt5.")
option(BUILD_OPENMP_ENABLE "Use OpenMP parallelization if available. If TRUE, the definition USEOPENMP is set. This is only the case if OpenMP is generally available and if the build is release." ON)

#These are the overall pre-compiler directives for itom and its plugins:
#
#Windows:
#  x86: WIN32, _WIN32
#  x64: WIN32, _WIN32, WIN64, _WIN64
#Linux:
#       linux, Linux
#Apple:
#       __APPLE__
if(CMAKE_HOST_WIN32)
    # check for EHa flag, as CMake does not set it by default we use this to determine if want
    # to add the itom compiler flags
    string(FIND ${CMAKE_CXX_FLAGS} "/EHa" _index)
    if(NOT ${_index} GREATER -1)
        # enable catching of machine exceptions, e.g. access violations
        string(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

        # check some if some other flags are set, and if not set them
        string(FIND ${CMAKE_CXX_FLAGS} "/DWIN32" _index)
        if(NOT ${_index} GREATER -1)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32")
        endif()
        
        string(FIND ${CMAKE_CXX_FLAGS} "/D_WIN32" _index)
        if(NOT ${_index} GREATER -1)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32")
        endif()

        # some more additional flags for win64
        if(BUILD_TARGET64)
            string(FIND ${CMAKE_CXX_FLAGS} "/DWIN64" _index)
            if(NOT ${_index} GREATER -1)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN64")
            endif()
            string(FIND ${CMAKE_CXX_FLAGS} "/D_WIN64" _index)
            if(NOT ${_index} GREATER -1)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN64")
            endif()
        endif()       
        
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "common C++ build flags" FORCE)
    endif()    
elseif(CMAKE_HOST_APPLE)
    add_definitions(-D__APPLE__)
elseif(CMAKE_HOST_UNIX) #this also includes apple, which is however already handled above
    add_definitions(-DLinux -Dlinux)
endif()


if(MSVC)
    # ck 15/11/2017 changed, as adding /MP to definitions breaks cuda builds with e.g. enable_language(CUDA), i.e. better
    # msvs integration of cuda build
    #add_definitions(/MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "/MP")

    # set some optimization compiler flags
    # i.e.:
    #   - Ox full optimization (replaces standard O2 set by cmake)
    #   - Oi enable intrinsic functions
    #   - Ot favor fast code
    #   - Oy omit frame pointers
    #   - GL whole program optimization
    #   - GT fibre safe optimization
    #   - openmp enable openmp support, isn't enabled globally here as it breaks opencv
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    
    if(NOT BUILD_TARGET64)
        #Disable safe SEH for Visual Studio, 32bit (this is necessary since some 3rd party libraries are compiled without /SAFESEH)
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
        set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
        set(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
        set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
        set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
    endif()
endif()

#try to enable OpenMP (e.g. not available with VS Express)
find_package(OpenMP QUIET)

if(OPENMP_FOUND)
    if(BUILD_OPENMP_ENABLE)
        message(STATUS "OpenMP found and enabled for release compilation")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -DUSEOPENMP" )
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_C_FLAGS} -DUSEOPENMP" )
    else(BUILD_OPENMP_ENABLE)
        message(STATUS "OpenMP found but not enabled for release compilation")
    endif(BUILD_OPENMP_ENABLE)
else(OPENMP_FOUND)
    message(STATUS "OpenMP not found.")
endif(OPENMP_FOUND)

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

add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)

include(CMakeParseArguments)

# - enables a linux compiler to start the build with multiple cores.
macro(itom_build_parallel_linux target)
  if(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "GNUCXX pipe flag enabled")
        set_target_properties(${target} PROPERTIES COMPILE_FLAGS "-pipe")
  endif(CMAKE_COMPILER_IS_GNUCXX)
endmacro()

# - initializes the default CMake policy to 3.12. Call this at the start of a plugin'safe
# or designer plugin's CMakeLists.txt to have a common CMake policy behaviour.
# 
# If you have a CMake version <= 3.12, all existing policies will be considered to be NEW.
# For CMake versions > 3.12, all policies, that have been introduced after this CMake version will
# be handled as OLD, all other policies as NEW.
#
# example:
# 
# set(target_name yourTargetName)
# set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk folder")
# 
# include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")
# 
# itom_init_cmake_policy()
# itom_init_plugin_library(${target_name})
# .
macro(itom_init_cmake_policy)
    cmake_minimum_required(VERSION 3.1...3.15)

    if(${CMAKE_VERSION} VERSION_LESS 3.12)
        cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
    endif()
endmacro()

# - This macro set common initializing things for an itom plugin. Call this macro after having
# included this file at the beginning of the CMakeLists.txt of the plugin.
#
# example:
# 
# set(target_name yourTargetName)
# set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk folder")
# 
# include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")
# 
# itom_init_cmake_policy()
# itom_init_plugin_library(${target_name})
# .
macro(itom_init_plugin_library target)
    message(STATUS "\n<--- PLUGIN ${target} --->")
endmacro()

# - This macro set common initializing things for an itom designer plugin. Call this macro after having
# included this file at the beginning of the CMakeLists.txt of the designer plugin.
#
# example:
# 
# set(target_name yourTargetName)
# set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk folder")
# 
# include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")
# 
# itom_init_cmake_policy()
# itom_init_designerplugin_library(${target_name})
# .
macro(itom_init_designerplugin_library target)
    message(STATUS "\n<--- DESIGNERPLUGIN ${target} --->")
endmacro()

# - call this macro to find one of the supported Qt packages (currently only Qt5 is supported, the support
# of Qt4 has been removed.
# 
# example:
# 
# itom_find_package_qt(ON Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools Designer)
#
# this will detect Qt with all given packages (packages given as Qt5 package names) 
# and automoc for Qt5 is set to ON.
#
# If the CMAKE Config variable BUILD_QTVERSION is 'auto', Qt5 is detected (support for Qt4 has been removed).
# Force to find a specific Qt-branch by setting BUILD_QTVERSION to either 'Qt5'
#
# For Qt5.0 a specific load mechanism is used, since find_package(Qt5 COMPONENTS...) is only available for Qt5 > 5.0.
macro(itom_find_package_qt SET_AUTOMOC)
    
    set(Components ${ARGN}) #all arguments after SET_AUTOMOC are components for Qt
    set(QT_COMPONENTS ${ARGN})
    set(QT5_LIBRARIES "")

    if(${BUILD_QTVERSION} STREQUAL "Qt4")
        message(SEND_ERROR "The support for Qt4 has been removed for itom > 3.2.1")
    elseif(${BUILD_QTVERSION} STREQUAL "Qt5")
        if(POLICY CMP0020)
            cmake_policy(SET CMP0020 NEW)
        endif(POLICY CMP0020)
        set(DETECT_QT5 TRUE)
    elseif(${BUILD_QTVERSION} STREQUAL "auto")
        if(POLICY CMP0020)
            cmake_policy(SET CMP0020 NEW)
        endif(POLICY CMP0020)
        set(DETECT_QT5 TRUE)
    else()
        message(SEND_ERROR "wrong value for BUILD_QTVERSION. auto, Qt5 allowed")
    endif()
    set(QT5_FOUND FALSE)
        
    if(DETECT_QT5)
        #TRY TO FIND QT5
        find_package(Qt5 COMPONENTS Core QUIET)
        
        if(${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
            #maybe Qt5.0 is installed that does not support the overall FindQt5 script
            find_package(Qt5Core QUIET)
            if(NOT Qt5Core_FOUND)
                if(${BUILD_QTVERSION} STREQUAL "auto")
                    set(DETECT_QT5 FALSE)
                else()
                    message(SEND_ERROR "Qt5 could not be found on this computer")
                endif()
            else(NOT Qt5Core_FOUND)
                set(QT5_FOUND TRUE)
                
                if(WIN32)
                    find_package(WindowsSDK REQUIRED)
                    set(CMAKE_PREFIX_PATH "${WINDOWSSDK_PREFERRED_DIR}/Lib/")
                    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${WINDOWSSDK_PREFERRED_DIR}/Lib/)
                endif(WIN32)
                
                set(QT5_FOUND TRUE)
                
                if(${SET_AUTOMOC})
                    set(CMAKE_AUTOMOC ON)
                else(${SET_AUTOMOC})
                    set(CMAKE_AUTOMOC OFF)
                endif(${SET_AUTOMOC})
                
                foreach(comp ${Components})
                    message(STATUS "FIND_PACKAGE FOR COMPONENT ${comp}")
                    find_package(Qt5${comp} REQUIRED)
                    if(${comp} STREQUAL "Widgets")
                        add_definitions(${Qt5Widgets_DEFINITIONS})
                        set(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                    elseif(${comp} STREQUAL "LinguistTools")
                        #it is not possible to link Qt5::LinguistTools since it does not exist
                    else()
                        set(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                    endif()
                endforeach(comp)
            endif(NOT Qt5Core_FOUND)
            
        else(${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
            #QT5 could be found with component based find_package command
            if(WIN32)
              find_package(WindowsSDK REQUIRED)
              set(CMAKE_PREFIX_PATH "${WINDOWSSDK_PREFERRED_DIR}/Lib/")
              set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${WINDOWSSDK_PREFERRED_DIR}/Lib/)
            endif(WIN32)
            
            find_package(Qt5 COMPONENTS ${Components} REQUIRED)
            set(QT5_FOUND TRUE)
            
            if(${SET_AUTOMOC})
                set(CMAKE_AUTOMOC ON)
            else(${SET_AUTOMOC})
                set(CMAKE_AUTOMOC OFF)
            endif(${SET_AUTOMOC})
            
            foreach(comp ${Components})
                if(${comp} STREQUAL "Widgets")
                    add_definitions(${Qt5Widgets_DEFINITIONS})
                    set(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                elseif(${comp} STREQUAL "LinguistTools")
                    #it is not possible to link Qt5::LinguistTools since it does not exist
                else()
                    set(QT5_LIBRARIES ${QT5_LIBRARIES} Qt5::${comp})
                endif()  
            endforeach(comp)
            
        endif(${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND") 
        
        if(Qt5Core_FOUND)
            # These variables are not defined with Qt5 CMake modules
            set(QT_BINARY_DIR "${_qt5Core_install_prefix}/bin")
            set(QT_LIBRARY_DIR "${_qt5Core_install_prefix}/lib")
        endif()
        
    endif(DETECT_QT5)
    
    add_definitions(${QT_DEFINITIONS})
endmacro()


#use this macro in order to generate and/or reconfigure the translation of any plugin or designer plugin.
#
# example:
#
# #1. scan for existing translation files (*.ts)
# file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
# #2. give all source files that should be checked for strings to be translated
# set(FILES_TO_TRANSLATE ${PLUGIN_SOURCES} ${PLUGIN_HEADERS} ${PLUGIN_UIC}) #adds all files to the list of files that are searched for strings to translate
# itom_library_translation(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")
#
# Hereby, ITOM_LANGUAGES is a semicolon-separated string with different languages, e.g. "de;fr"
# UPDATE_TRANSLATIONS is an option (ON/OFF) that decides whether the qm-file should only be build from the existing ts-file or if the ts-file
# is reconfigured with respect to the given files in FILES_TO_TRANSLATE.
# 
# Please note, that you need to add the resulting QM_FILES to the copy-list using the macro
# itom_add_plugin_qm_files_to_copy_list or itom_add_designer_qm_files_to_copy_list (for plugins or designer plugins)
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# # e.g. add further entries to COPY_SOURCES and COPY_DESTINATIONS
# itom_add_designer_qm_files_to_copy_list(QM_FILES COPY_SOURCES COPY_DESTINATIONS)
# itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)
#
# This macro automatically adds all translation files (*.ts) as source files to the given target.
# .
macro(itom_library_translation qm_files target force_translation_update existing_translation_files languages files_to_translate)

    if(NOT QT5_FOUND)
        message(SEND_ERROR "Currently only Qt5 is supported")
    endif()
    
    if(${force_translation_update})
        set(TRANSLATIONS_FILES) #list with all ts files
        set(TRANSLATION_OUTPUT_FILES)
        itom_qt5_create_translation(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target} ${languages} ${files_to_translate})
        add_custom_target (_${target}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
        add_dependencies(${target} _${target}_translation)
    else()
        set(TRANSLATIONS_FILES ${existing_translation_files})
    endif()
    
    set(QMFILES)
    itom_qt5_compile_translation(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${TRANSLATIONS_FILES})
    set(${qm_files} ${${qm_files}} ${QMFILES})
    
    #add the translation files to the solution
    target_sources(${target}
        PRIVATE
        ${TRANSLATIONS_FILES}
    )
endmacro()

# Parses all given source file for Qt translation strings and create one ts file per
# desired language using Qt's tool lupdate.
# 
# The call is
# itom_qt5_create_translation(outputFiles tsFiles target languages srcfile1 srcfile2...)
# 
# .
macro(itom_qt5_create_translation outputFiles tsFiles target languages)
    message(STATUS "--------------------------------------------------------------------")
    message(STATUS "itom_qt5_create_translation: Create ts files for target ${target}")
    message(STATUS "--------------------------------------------------------------------")
    
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
        
        
        if(EXISTS ${_abs_FILE})
            message(STATUS "- Consider existing ts-file: ${_abs_FILE}")
            list(APPEND _my_tsfiles ${_abs_FILE})
        else()
            #create new ts file
            add_custom_command(OUTPUT ${_abs_FILE}_new
                COMMAND ${Qt5_LUPDATE_EXECUTABLE}
                ARGS ${_lupdate_options} ${_my_dirs} -locations relative -no-ui-lines -target-language ${_lang} -ts ${_abs_FILE}
                DEPENDS ${_my_sources} VERBATIM)
            list(APPEND _my_tsfiles ${_abs_FILE})
            set(${outputFiles} ${${outputFiles}} ${_abs_FILE}_new) #add output file for custom command to outputFiles list
            message(STATUS "- Create new ts-file (lupdate process): ${_abs_FILE}")
        endif()
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
        message(STATUS "- Update (existing) ts-file (lupdate process): ${_ts_file}")
    endforeach()
    
    message(STATUS "--------------------------------------------------------------------")
endmacro()

# - compiles all source translation files (*.ts), given after the target argument,
# using Qt's lrelease tool and outputs their binary representation (*.qm), that is stored
# in the output_location. The list of _qm_files, that might already contain values before
# calling this macro is extended by the newly compiled qm files.
# 
# This step is added as custom command to the given target. Usually a new target is used
# for this and the target, that originally contains the source files should have a dependency
# to this target.
#
# The call is
# itom_qt5_compile_translation(qm_files output_location target tsfile1 tsfile2...)
#
# example:
# set(QM_FILES "")
# itom_qt5_add_transation(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" "build_translation_target" "file1.ts file2.ts file3.ts")
# .
macro(itom_qt5_compile_translation _qm_files output_location target)
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
endmacro()


# this macro only generates the moc-file but does not compile it, since it is included in another source file.
# this comes from the ctkCommon project
# Creates a rule to run moc on infile and create outfile. Use this IF for some reason QT5_WRAP_CPP() 
# isn't appropriate, e.g. because you need a custom filename for the moc file or something similar.
# 
# Pass a list of files, that should be processed by Qt's moc tool. The moc files will then be located
# in the binary directory.
#
# example:
# itom_qt_generate_mocs(
#     pathLineEdit.h
#     checkBoxSpecial.h
# )
macro(itom_qt_generate_mocs)
    foreach(file ${ARGN})
        set(moc_file moc_${file})
        
        if(${QT5_FOUND})
            QT5_GENERATE_MOC(${file} ${moc_file})
        else()
            message(SEND_ERROR "Qt5 must be present to generate mocs")
        endif()

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


# - itom_qt_wrap_ui(outfiles target ui_file1 ui_file2 ... )
#
# This is a wrapper for qt5_wrap_ui (if Qt5 is available) that processes the given ui files
# by Qt's uic process, automatically adds the processed files (ui_...h) to the target's sources
# and returns the processed files as outfiles.
#
# Usually Qt's AUTOUIC would do the same, however it might also be required
# to translate these processed files. Then, the access to the list of parsed filenames
# must be available, such that it can be passed to the translation macro.
# Therefore it is recommended to use this method.
#
# Hint: it is no problem to enable AUTOUIC though, since qt5_wrap_ui will skip the autouic
# for every single file that is passed to this function.
function(itom_qt_wrap_ui outfiles target)
    if(QT5_FOUND)
        #parse all *.ui files by Qt's uic process and get the parsed source files
        qt5_wrap_ui(temp_output ${ARGN})
        #add the output files to the target
        target_sources(${target} PRIVATE ${temp_output})
        list(APPEND ${outfiles} ${temp_output})
        set(${outfiles} ${${outfiles}} PARENT_SCOPE)
    else()
        message(SEND_ERROR "Currently only Qt5 is supported")
    endif()
endfunction()


# - use this macro in order to append to the sources and destinations
# list the library file for your designer plugin, that is finally
# copied to the designer folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# itom_add_designerlibrary_to_copy_list(targetNameOfYourDesignerPlugin COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call itom_post_build_copy_files-macro, to initialize the copying.
#
macro(itom_add_designerlibrary_to_copy_list target sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    list(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    list(APPEND ${destinations} ${ITOM_APP_DIR}/designer)
    
    list(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
    list(APPEND ${destinations} ${ITOM_APP_DIR}/designer)
endmacro()


# use this macro in order to append to the sources and destinations
# list the header files for your designer plugin, that is finally
# copied to the designer/{$target} folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# itom_add_designerplugin_headers_to_copy_list(targetNameOfYourDesignerPlugin list_of_header_files COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call itom_post_build_copy_files-macro, to initialize the copying.
#
macro(itom_add_designerplugin_headers_to_copy_list target headerfiles sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    foreach(_hfile ${${headerfiles}})
        list(APPEND ${sources} ${_hfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/designer/${target})
    endforeach()
endmacro()


# use this macro in order to append to the sources and destinations
# list the library file for your itom plugin, that is finally
# copied to the plugin/target folder of itom.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# itom_add_pluginlibrary_to_copy_list(targetNameOfYourPlugin COPY_SOURCES COPY_DESTINATIONS)
#
# Now the length of COPY_SOURCES and COPY_DESTINATIONS is 1. You can append further entries
# and finally call itom_post_build_copy_files-macro, to initialize the copying.
#
macro(itom_add_pluginlibrary_to_copy_list target sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    list(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    
    list(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target})
endmacro()

# - appends the list of binary translation files (qm_files) to be copied from their source
# directory to the 'plugins/${target}/translation' subfolder of the qitom root directory. This is
# done by adding one or multiple filepathes and folders to the given lists 'sources' and
# 'destinations'. The copy operation from every entry in sources to destinations
# can then be triggered by calling 'itom_post_build_copy_files'.
#
# qm_files are usually obtained by calling 'itom_library_translation'.
macro(itom_add_plugin_qm_files_to_copy_list target qm_files sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    foreach(_qmfile ${${qm_files}})
        list(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target}/translation)
    endforeach()
endmacro()

# - appends the list of binary translation files (qm_files) to be copied from their source
# directory to the 'designer/translation' subfolder of the qitom root directory. This is
# done by adding one or multiple filepathes and folders to the given lists 'sources' and
# 'destinations'. The copy operation from every entry in sources to destinations
# can then be triggered by calling 'itom_post_build_copy_files'.
#
# qm_files are usually obtained by calling 'itom_library_translation'.
macro(itom_add_designer_qm_files_to_copy_list qm_files sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    foreach(_qmfile ${${qm_files}})
        list(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/designer/translation)
    endforeach()
endmacro()

# - adds a post build step to the given target that iteratively
# adds a copy operation from every filepath in 'sources' to the directory in 'destinations'.
# The arguments 'sources' and 'destinations' must therefore contain the same number of elements.
macro(itom_post_build_copy_files target sources destinations)
    list(LENGTH ${sources} temp)
    math(EXPR len1 "${temp} - 1")
    list(LENGTH ${destinations} temp)
    math(EXPR len2 "${temp} - 1")

    if( NOT len1 EQUAL len2 )
        message(SEND_ERROR "itom_post_build_copy_files: len(sources) must be equal to len(destinations)")
    endif( NOT len1 EQUAL len2 )
    
    set(destPathes "")
    
    foreach(dest ${${destinations}})
        list(APPEND destPathes ${dest})
    endforeach()
    
    list(REMOVE_DUPLICATES destPathes)
    
    #try to create all pathes
    foreach(destPath ${destPathes})
        #first try to create the directory
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${destPath}"
        )
    endforeach()
    
    foreach(val RANGE ${len1})
        list(GET ${sources} ${val} val1)
        list(GET ${destinations} ${val} val2)
        message(STATUS "-- POST_BUILD: copy '${val1}' to '${val2}'")
        
        add_custom_command(TARGET ${target} POST_BUILD                 # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different                 # which executes "cmake - E copy_if_different..."
                "${val1}"                                                 # <--this is in-file
                "${val2}"                                                # <--this is out-file path
        )
    endforeach()
endmacro()

# - adds a post_build step to the given target, that copies all given source files (sources)
# to the 'lib' subfolder of the itom root directory. This subfolder is added to the local PATH environment
# variable of the itom application before plugins or designer plugins are loaded.
#
# Therefore it is possible to copy further 3rd party libraries, that an itom plugin depends on,
# into this lib directory.
macro (itom_post_build_copy_files_to_lib_folder target sources)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    list(LENGTH ${sources} temp)
    math(EXPR len1 "${temp} - 1")
    
    if(${len1} GREATER "-1")
        #message(STATUS "sources LEN: ${len1}")
        #message(STATUS "destinations LEN: ${len2}")
        
        #create lib folder (for safety only, IF it does not exist some cmake versions do not copy the files in the 
        #desired way using copy_if_different below
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${ITOM_APP_DIR}/lib"
        )

        foreach(val RANGE ${len1})
            list(GET ${sources} ${val} val1)
            #message(STATUS "POST_BUILD: COPY ${val1} TO ${ITOM_APP_DIR}/lib")
            
            add_custom_command(TARGET ${target} POST_BUILD                 # Adds a post-build event to MyTest
                COMMAND ${CMAKE_COMMAND} -E copy_if_different                 # which executes "cmake - E copy_if_different..."
                    "${val1}"                                                 # <--this is in-file
                    "${ITOM_APP_DIR}/lib"                                                # <--this is out-file path
            )
        endforeach()
    else(${len1} GREATER "-1")
        message(STATUS "No files to copy to lib folder for target ${target}")
    endif(${len1} GREATER "-1")
endmacro()

# - groups all .h and .ui files in a "Header Files" group and .cpp files in a "Source Files" group in Visual Studio.
# Pass the name of a subfolder. All files within this subfolder will then be scanned and
# all header, ui and source files are distributed into filters or subfilters (MSVC only).
# If you want to pass a nested subfolder, use a slash, not a backslash.
# 
# Example: 
# 
# itom_add_source_group(codeEditor/syntaxHighlighter)
# .
macro(itom_add_source_group subfolder)
    string(REPLACE "/" "\\" subfolder_backslash "${subfolder}")
    
    set(REG_EXT_HEADERS "[^/]*([.]ui|[.]h|[.]hpp)$")
    set(REG_EXT_SOURCES "[^/]*([.]cpp|[.]c)$")

    source_group("Header Files\\${subfolder_backslash}" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/${REG_EXT_HEADERS}")
    source_group("Source Files\\${subfolder_backslash}" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/${REG_EXT_SOURCES}")
endmacro()



# - Create the configuration file for parsing and integrating the rst documentation of a plugin
# into the overall itom documentation.
#
# Call this macro by the end of the CMakeLists.txt of a plugin, such that the 
# plugin_doc_config.cfg file is generated in the 'docs' subfolder of the output folder of the plugin.
# This config file can then be parsed by the script 'create_plugin_doc.py' in the docs subfolder of the itom
# root directory.
#
# example if the rst main document of the plugin is called myPluginDoc.rst (and located in the docs subfolder
# of the plugin sources):
# itom_configure_plugin_documentation(${target_name} myPluginDoc)
# .
macro(itom_configure_plugin_documentation target main_document) #main_document without .rst at the end
    set(PLUGIN_NAME ${target})
    set(PLUGIN_DOC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(PLUGIN_DOC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/build)
    set(PLUGIN_DOC_INSTALL_DIR ${ITOM_APP_DIR}/plugins/${target}/docs)
    set(PLUGIN_DOC_MAIN ${main_document})
    configure_file(${ITOM_SDK_DIR}/docs/pluginDoc/plugin_doc_config.cfg.in ${CMAKE_CURRENT_BINARY_DIR}/docs/plugin_doc_config.cfg)
endmacro()

macro(itom_copy_file_if_changed in_file out_file target)
    IF(${in_file} IS_NEWER_THAN ${out_file})    
  #    message("COpying file: ${in_file} to: ${out_file}")
        add_custom_command (
    #    OUTPUT     ${out_file}
            TARGET ${target}
            POST_BUILD
            COMMAND    ${CMAKE_COMMAND}
            ARGS       -E copy ${in_file} ${out_file}
    #    DEPENDS     qitom
    #    DEPENDS    ${in_file}
    #    MAIN_DEPENDENCY ${in_file}
        )
    endif()
endmacro()

# Copy all files and directories in in_dir to out_dir. 
# Subtrees remain intact.
macro(itom_copy_directory_if_changed in_dir out_dir target pattern recurse)
    file(${recurse} in_file_list ${in_dir}/${pattern})
    foreach(in_file ${in_file_list})
        if(NOT ${in_file} MATCHES ".*svn.*")
            string(REGEX REPLACE ${in_dir} ${out_dir} out_file ${in_file}) 
            itom_copy_file_if_changed(${in_file} ${out_file} ${target})
        endif()
    endforeach()
endmacro()



# Deprecated macros added with itom 4.0 (January 2020). They will be removed in the future.
# These macros are only redirects to renamed macros.
macro(ADD_SOURCE_GROUP)
    message(WARNING "Deprecated call to 'ADD_SOURCE_GROUP'. Call 'itom_add_source_group' instead.")
    itom_add_source_group(${ARGV})
endmacro()

macro(QT5_GENERATE_MOCS)
    message(WARNING "Deprecated call to 'QT5_GENERATE_MOCS'. Call 'itom_qt_generate_mocs' instead.")
    itom_qt_generate_mocs(${ARGV})
endmacro()

macro(FIND_PACKAGE_QT)
    message(WARNING "Deprecated call to 'FIND_PACKAGE_QT'. Call 'itom_find_package_qt' instead.")
    itom_find_package_qt(${ARGV})
endmacro()

macro(POST_BUILD_COPY_FILES)
    message(WARNING "Deprecated call to 'POST_BUILD_COPY_FILES'. Call 'itom_post_build_copy_files' instead.")
    itom_post_build_copy_files(${ARGV})
endmacro()

macro(ADD_DESIGNERLIBRARY_TO_COPY_LIST)
    message(WARNING "Deprecated call to 'ADD_DESIGNERLIBRARY_TO_COPY_LIST'. Call 'itom_add_designerlibrary_to_copy_list' instead.")
    itom_add_designerlibrary_to_copy_list(${ARGV})
endmacro()

macro(ADD_DESIGNER_QM_FILES_TO_COPY_LIST)
    message(WARNING "Deprecated call to 'ADD_DESIGNER_QM_FILES_TO_COPY_LIST'. Call 'itom_add_designer_qm_files_to_copy_list' instead.")
    itom_add_designer_qm_files_to_copy_list(${ARGV})
endmacro()

macro(ADD_QM_FILES_TO_COPY_LIST)
    message(WARNING "Deprecated call to 'ADD_QM_FILES_TO_COPY_LIST'. Call 'itom_add_plugin_qm_files_to_copy_list' instead.")
    itom_add_plugin_qm_files_to_copy_list(${ARGV})
endmacro()

macro(PLUGIN_TRANSLATION)
    message(WARNING "Deprecated call to 'PLUGIN_TRANSLATION'. Call 'itom_library_translation' instead.")
    itom_library_translation(${ARGV})
endmacro()

macro(BUILD_PARALLEL_LINUX targetName)
    message(WARNING "Deprecated call to 'BUILD_PARALLEL_LINUX'. Call 'itom_build_parallel_linux' instead.")
    itom_build_parallel_linux(${ARGV})
endmacro()

macro(PLUGIN_DOCUMENTATION)
    message(WARNING "Deprecated call to 'PLUGIN_DOCUMENTATION'. Call 'itom_configure_plugin_documentation' instead.")
    itom_configure_plugin_documentation(${ARGV})
endmacro()

macro(POST_BUILD_COPY_FILE_TO_LIB_FOLDER)
    message(WARNING "Deprecated call to 'POST_BUILD_COPY_FILE_TO_LIB_FOLDER'. Call 'itom_post_build_copy_files_to_lib_folder' instead.")
    itom_post_build_copy_files_to_lib_folder(${ARGV})
endmacro()

macro(ADD_PLUGINLIBRARY_TO_COPY_LIST)
    message(WARNING "Deprecated call to 'ADD_PLUGINLIBRARY_TO_COPY_LIST'. Call 'itom_add_pluginlibrary_to_copy_list' instead.")
    itom_add_pluginlibrary_to_copy_list(${ARGV})
endmacro()

macro(ADD_DESIGNERHEADER_TO_COPY_LIST)
    message(WARNING "Deprecated call to 'ADD_DESIGNERHEADER_TO_COPY_LIST'. Call 'itom_add_designerplugin_headers_to_copy_list' instead.")
    itom_add_designerplugin_headers_to_copy_list(${ARGV})
endmacro()

MACRO(COPY_DIRECTORY_IF_CHANGED)
    message(WARNING "Deprecated call to 'COPY_DIRECTORY_IF_CHANGED'. Call 'itom_copy_directory_if_changed' instead.")
    itom_copy_directory_if_changed(${ARGV})
endmacro()