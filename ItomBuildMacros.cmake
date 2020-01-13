#########################################################################
#set general things
#########################################################################
cmake_minimum_required(VERSION 3.0.2)

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
set(BUILD_QTVERSION "auto" CACHE string "currently only Qt5 is supported. Set this value to 'auto' in order to auto-detect the correct Qt version or set it to 'Qt5' to hardly select Qt5.")
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE string "common C++ build flags" FORCE)
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


if(BUILD_ITOMLIBS_SHARED OR ITOM_SDK_SHARED_LIBS)
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
endif()

MACRO (BUILD_PARALLEL_LINUX targetName)
  if(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "GNUCXX pipe flag enabled")
        set_target_properties(${targetName} PROPERTIES COMPILE_FLAGS "-pipe")
  endif(CMAKE_COMPILER_IS_GNUCXX)
ENDMACRO (BUILD_PARALLEL_LINUX)

MACRO (INIT_ITOM_LIBRARY)
    # put any configurations here, that should hold for all libraries, plugins, designer plugins etc. of itom
    
    # step 1: set global CMake policies

    if(POLICY CMP0028)
        cmake_policy(SET CMP0028 NEW) #raise an CMake error if an imported target, containing ::, could not be found (CMake >= 3.0)
    endif()

    if(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
        if(POLICY CMP0042)
            cmake_policy(SET CMP0042 OLD) # (CMake >= 3.0)
        endif(POLICY CMP0042)
    endif()

    if(POLICY CMP0053)
        cmake_policy(SET CMP0053 NEW) #Simplify variable reference and escape sequence evaluation. (CMake >= 3.1)
    endif()
    
    if(POLICY CMP0071)
        cmake_policy(SET CMP0071 NEW) #Let AUTOMOC and AUTOUIC process GENERATED files.
    endif()

    if(POLICY CMP0074)
        cmake_policy(SET CMP0074 NEW) #find_package() uses <PackageName>_ROOT variables.. (CMake >= 3.12)
    endif()
ENDMACRO (INIT_ITOM_LIBRARY)

MACRO (FIND_PACKAGE_QT SET_AUTOMOC)
    # call this macro to find one of the supported Qt packages (currently only Qt5 is supported, the support
    # of Qt4 has been removed.
    #
    # call example FIND_PACKAGE_QT(ON Widgets UiTools PrintSupport Network Sql Xml OpenGL LinguistTools Designer)
    #
    # this will detect Qt with all given packages (packages given as Qt5 package names) 
    # and automoc for Qt5 is set to ON
    #
    # If the CMAKE Config variable BUILD_QTVERSION is 'auto', Qt5 is detected (support for Qt4 has been removed).
    # Force to find a specific Qt-branch by setting BUILD_QTVERSION to either 'Qt5'
    #
    # For Qt5.0 a specific load mechanism is used, since find_package(Qt5 COMPONENTS...) is only available for Qt5 > 5.0
    #
    set(Components ${ARGN}) #all arguments after SetAutomoc are components for Qt
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
ENDMACRO (FIND_PACKAGE_QT)


#use this macro in order to generate and/or reconfigure the translation of any plugin or designer plugin.
#
# example:
# set(FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_ui}) #adds all files to the list of files that are searched for strings to translate
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
    set(TRANSLATIONS_FILES)
    set(TRANSLATION_OUTPUT_FILES)
    set(QMFILES)

    if(${force_translation_update})
        if(QT5_FOUND)
            QT5_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target} ${languages} ${files_to_translate})
        else(QT5_FOUND)
            message(SEND_ERROR "Currently only Qt5 is supported")
        endif(QT5_FOUND)
        
        add_custom_target (_${target}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
        add_dependencies(${target} _${target}_translation)
        
        if(QT5_FOUND)
            QT5_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${TRANSLATIONS_FILES})
        else(QT5_FOUND)
            message(SEND_ERROR "Currently only Qt5 is supported")
        endif(QT5_FOUND)
    else(${force_translation_update})
        if(QT5_FOUND)
            QT5_ADD_TRANSLATION_ITOM(QMFILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target} ${existing_translation_files})
        else(QT5_FOUND)
            message(SEND_ERROR "Currently only Qt5 is supported")
        endif(QT5_FOUND)
    endif(${force_translation_update})
    
    set(${qm_files} ${${qm_files}} ${QMFILES})
    
ENDMACRO (PLUGIN_TRANSLATION)


###########################################################################
# useful macros
###########################################################################

MACRO(QT5_CREATE_TRANSLATION_ITOM outputFiles tsFiles target languages)
    message(STATUS "--------------------------------------------------------------------\nQT5_CREATE_TRANSLATION_ITOM: Create ts files for target ${target}\n--------------------------------------------------------------------")
    
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
#    QT5_ADD_TRANSLATION_ITOM(${_qm_files} ${_my_tsfiles})
#    set(${_qm_files} ${${_qm_files}} PARENT_SCOPE)
    message(STATUS "--------------------------------------------------------------------")
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
macro(QT5_GENERATE_MOCS)
    foreach(file ${ARGN})
        set(moc_file moc_${file})
        QT5_GENERATE_MOC(${file} ${moc_file})

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
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    endif()
    list(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    list(APPEND ${destinations} ${ITOM_APP_DIR}/designer)
    
    list(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
    list(APPEND ${destinations} ${ITOM_APP_DIR}/designer)    
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
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    endif()
    
    foreach(_hfile ${${headerfiles}})
        list(APPEND ${sources} ${_hfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/designer/${target})
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
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    endif()
    
    #GET_TARGET_PROPERTY(VAR_LOCATION ${target} LOCATION)
    #string(REGEX REPLACE "\\(Configuration\\)" "<CONFIGURATION>" VAR_LOCATION ${VAR_LOCATION})
    #set(VAR_LOCATION "$<TARGET_FILE:${target}>")
    list(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    
    list(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target})
ENDMACRO (ADD_PLUGINLIBRARY_TO_COPY_LIST)


MACRO (ADD_QM_FILES_TO_COPY_LIST target qm_files sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    endif()
    
    foreach(_qmfile ${${qm_files}})
        list(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/plugins/${target}/translation)
    endforeach()
ENDMACRO (ADD_QM_FILES_TO_COPY_LIST)


MACRO (ADD_DESIGNER_QM_FILES_TO_COPY_LIST qm_files sources destinations)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
    endif()
    
    foreach(_qmfile ${${qm_files}})
        list(APPEND ${sources} ${_qmfile}) #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
        list(APPEND ${destinations} ${ITOM_APP_DIR}/designer/translation)
    endforeach()
ENDMACRO (ADD_DESIGNER_QM_FILES_TO_COPY_LIST)


MACRO (ADD_OUTPUTLIBRARY_TO_SDK_LIB target sources destinations)
    
    if(${ITOM_SDK_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_SDK_DIR is not indicated")
    endif()
    
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      set(SDK_PLATFORM "x86")
    else()
      set(SDK_PLATFORM "x64")
    endif()
    
    # The following list has to be consistent with FindITOM_SDK.cmake! 
    # From VS higher than 1900, the default case vc${MSVC_VERSION} is used.
    if(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "vc14")
    elseif(MSVC_VERSION EQUAL 1800)
        set(SDK_COMPILER "vc12")
    elseif(MSVC_VERSION EQUAL 1700)
        set(SDK_COMPILER "vc11")
    elseif(MSVC_VERSION EQUAL 1600)
        set(SDK_COMPILER "vc10")
    elseif(MSVC_VERSION EQUAL 1500)
        set(SDK_COMPILER "vc9")
    elseif(MSVC_VERSION EQUAL 1400)
        set(SDK_COMPILER "vc8")
    elseif(MSVC)
        set(SDK_COMPILER "vc${MSVC_VERSION}")
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SDK_COMPILER "clang")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(SDK_COMPILER "intel")
    elseif(APPLE)
        set(SDK_COMPILER "osx_default")
    else(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "unknown")
    endif(MSVC_VERSION EQUAL 1900)
    
    set( destination "${ITOM_SDK_DIR}/lib/${SDK_COMPILER}_${SDK_PLATFORM}" )
    
    list(APPEND ${sources} "$<TARGET_FILE:${target}>") #adds the complete source path including filename of the dll (configuration-dependent) to the list 'sources'
    list(APPEND ${destinations} ${destination}) 
ENDMACRO (ADD_OUTPUTLIBRARY_TO_SDK_LIB target sources destinations)


MACRO (ADD_LIBRARY_TO_APPDIR_AND_SDK target sources destinations)

    if(${ITOM_SDK_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_SDK_DIR is not indicated")
    endif()

    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        set(SDK_PLATFORM "x86")
    else( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        set(SDK_PLATFORM "x64")
    endif( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    
    # The following list has to be consistent with FindITOM_SDK.cmake! 
    # From VS higher than 1900, the default case vc${MSVC_VERSION} is used.
    if(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "vc14")
    elseif(MSVC_VERSION EQUAL 1800)
        set(SDK_COMPILER "vc12")
    elseif(MSVC_VERSION EQUAL 1700)
        set(SDK_COMPILER "vc11")
    elseif(MSVC_VERSION EQUAL 1600)
        set(SDK_COMPILER "vc10")
    elseif(MSVC_VERSION EQUAL 1500)
        set(SDK_COMPILER "vc9")
    elseif(MSVC_VERSION EQUAL 1400)
        set(SDK_COMPILER "vc8")
    elseif(MSVC)
        set(SDK_COMPILER "vc${MSVC_VERSION}")
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SDK_COMPILER "clang")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(SDK_COMPILER "intel")
    elseif(APPLE)
        set(SDK_COMPILER "osx_default")
    else(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "unknown")
    endif(MSVC_VERSION EQUAL 1900)

    set( sdk_destination "${ITOM_SDK_DIR}/lib/${SDK_COMPILER}_${SDK_PLATFORM}" )

    if(BUILD_ITOMLIBS_SHARED)
        #copy library (dll) to app-directory and linker library (lib) to sdk_destination
        list(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
        list(APPEND ${destinations} ${sdk_destination})

        list(APPEND ${sources} "$<TARGET_FILE:${target}>")
        list(APPEND ${destinations} ${ITOM_APP_DIR})
    else()
        #copy linker library (lib) to sdk_destination
        list(APPEND ${sources} "$<TARGET_FILE:${target}>")
        list(APPEND ${destinations} ${sdk_destination})
    endif()
ENDMACRO (ADD_LIBRARY_TO_APPDIR_AND_SDK target sources destinations)


MACRO (POST_BUILD_COPY_FILES target sources destinations)
    list(LENGTH ${sources} temp)
    math(EXPR len1 "${temp} - 1")
    list(LENGTH ${destinations} temp)
    math(EXPR len2 "${temp} - 1")
    #message(STATUS "sources LEN: ${len1}")
    #message(STATUS "destinations LEN: ${len2}")

    if( NOT len1 EQUAL len2 )
        message(SEND_ERROR "POST_BUILD_COPY_FILES len(sources) must be equal to len(destinations)")
    endif( NOT len1 EQUAL len2 )
    
    set(destPathes "")
    foreach(dest ${${destinations}})
        #IF dest is a full name to a file:
        #get_filename_component(destPath ${dest} PATH)
        #list(APPEND destPathes ${destPath})
        list(APPEND destPathes ${dest})
    endforeach(dest ${${destinations}})
    list(REMOVE_DUPLICATES destPathes)
    
#    message(STATUS "destPathes: ${destPathes}")
    
    #try to create all pathes
    foreach(destPath ${destPathes})
        #first try to create the directory
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${destPath}"
        )
    endforeach(destPath ${destPathes})
    
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
ENDMACRO (POST_BUILD_COPY_FILES target sources destinations)


MACRO (POST_BUILD_COPY_FILE_TO_LIB_FOLDER target sources)
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_DIR is not indicated")
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
ENDMACRO (POST_BUILD_COPY_FILE_TO_LIB_FOLDER target sources)


MACRO (ADD_SOURCE_GROUP subfolder)
    #pass a subfolder. Its directory is scanned an all header, ui and sources files
    #are distributed into filters or subfilters (MSVC only)
    #if you want to pass a nested subfolder, use the 'slash' (not the 'backslash')
    string(REPLACE "/" "\\" subfolder_backslash "${subfolder}")
    file(GLOB GROUP_FILES_H
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.ui
    )
    source_group("Header Files\\${subfolder_backslash}" FILES ${GROUP_FILES_H})
    
    file(GLOB GROUP_FILES_S
        ${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.cpp
    )
    source_group("Source Files\\${subfolder_backslash}" FILES ${GROUP_FILES_S})
ENDMACRO (ADD_SOURCE_GROUP subfolder)


#some unused macros

MACRO(COPY_FILE_IF_CHANGED in_file out_file target)
#  message(STATUS "copy command: " ${in_file} " " ${out_file} " " ${target})
    if(${in_file} IS_NEWER_THAN ${out_file})    
  #    message("COpying file: ${in_file} to: ${out_file}")
        add_custom_command(
    #    OUTPUT     ${out_file}
            TARGET ${target}
            POST_BUILD
            COMMAND    ${CMAKE_COMMAND}
            ARGS       -E copy ${in_file} ${out_file}
    #    DEPENDS     qitom
    #    DEPENDS    ${in_file}
    #    MAIN_DEPENDENCY ${in_file}
        )
    endif(${in_file} IS_NEWER_THAN ${out_file})
ENDMACRO(COPY_FILE_IF_CHANGED)


MACRO(COPY_FILE_INTO_DIRECTORY_IF_CHANGED in_file out_dir target)
    get_filename_component(file_name ${in_file} NAME) 
    COPY_FILE_IF_CHANGED(${in_file} ${out_dir}/${file_name} ${target})
ENDMACRO(COPY_FILE_INTO_DIRECTORY_IF_CHANGED)


#Copies all the files from in_file_list into the out_dir. 
# sub-trees are ignored (files are stored in same out_dir)
MACRO(COPY_FILES_INTO_DIRECTORY_IF_CHANGED in_file_list out_dir target)
    foreach(in_file ${in_file_list})
        COPY_FILE_INTO_DIRECTORY_IF_CHANGED(${in_file} ${out_dir} ${target})
    endforeach(in_file)     
ENDMACRO(COPY_FILES_INTO_DIRECTORY_IF_CHANGED)


#Copy all files and directories in in_dir to out_dir. 
# Subtrees remain intact.
MACRO(COPY_DIRECTORY_IF_CHANGED in_dir out_dir target pattern recurse)
    #message("Copying directory ${in_dir}")
    file(${recurse} in_file_list ${in_dir}/${pattern})
    foreach(in_file ${in_file_list})
        if(NOT ${in_file} MATCHES ".*svn.*")
            string(REGEX REPLACE ${in_dir} ${out_dir} out_file ${in_file}) 
            COPY_FILE_IF_CHANGED(${in_file} ${out_file} ${target})
        endif(NOT ${in_file} MATCHES ".*svn.*")
    endforeach(in_file)     
ENDMACRO(COPY_DIRECTORY_IF_CHANGED)


MACRO(PLUGIN_DOCUMENTATION target main_document) #main_document without .rst at the end
    set(PLUGIN_NAME ${target})
    set(PLUGIN_DOC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(PLUGIN_DOC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/build)
    set(PLUGIN_DOC_INSTALL_DIR ${ITOM_APP_DIR}/plugins/${target}/docs)
    set(PLUGIN_DOC_MAIN ${main_document})
    configure_file(${ITOM_SDK_DIR}/docs/pluginDoc/plugin_doc_config.cfg.in ${CMAKE_CURRENT_BINARY_DIR}/docs/plugin_doc_config.cfg)
ENDMACRO(PLUGIN_DOCUMENTATION)


# OSX ONLY: Copy files from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS.
if(APPLE)
    MACRO(COPY_TO_BUNDLE target srcDir destDir)
        file(GLOB_RECURSE templateFiles RELATIVE ${srcDir} ${srcDir}/*)
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    ENDMACRO(COPY_TO_BUNDLE)
endif(APPLE)


# OSX ONLY: Copy files from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS.
if(APPLE)
    MACRO(COPY_TO_BUNDLE_NONREC target srcDir destDir)
        file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    ENDMACRO(COPY_TO_BUNDLE_NONREC)
endif(APPLE)


# OSX ONLY: Copy files of certain type from source directory to destination directory in app bundle, substituting any
# variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS
if(APPLE)
    MACRO(COPY_TYPE_TO_BUNDLE target srcDir destDir type)
        file(GLOB_RECURSE templateFiles RELATIVE ${srcDir} ${srcDir}/*${type})
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    ENDMACRO(COPY_TYPE_TO_BUNDLE)
endif(APPLE)


# OSX ONLY: Copy files of certain type from source directory to destination directory in app bundle, substituting any
# variables (NON-RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS
if(APPLE)
    MACRO(COPY_TYPE_TO_BUNDLE_NONREC target srcDir destDir type)
        file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*${type})
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    ENDMACRO(COPY_TYPE_TO_BUNDLE_NONREC)
endif(APPLE)