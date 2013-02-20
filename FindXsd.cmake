# 
# Attempt to find the xsd application in various places. If found, the full
# path will be in XSD_EXECUTABLE. Look in the usual locations, as well as in
# the 'bin' directory in the path given in the XSD_ROOT environment variable.
# 

IF (XSD_INCLUDE_DIR AND XSD_EXECUTABLE)
# in cache already
SET(XSD_FIND_QUIETLY TRUE)
ENDIF (XSD_INCLUDE_DIR AND XSD_EXECUTABLE)

SET (XSD_POSSIBLE_ROOT_DIRS
  "$ENV{XSDDIR}"
  "$ENV{XSDDIR}"
  /usr/local
  /usr
  "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3"
  ${CMAKE_SOURCE_DIR}/../xsd/libxsd
 "$ENV{PATH}"
  )

 FIND_PATH(XSD_ROOT_DIR 
  NAMES 
  include/xsd/cxx/parser/elements.hxx     
  PATHS ${XSD_POSSIBLE_ROOT_DIRS}
  )



if (WIN32)
  set (XSD_EXE_NAME xsd)
endif (WIN32)
if (UNIX)
  set (XSD_EXE_NAME xsdcxx)
endif (UNIX)
if (APPLE)
  set (XSD_EXE_NAME xsd)
endif (APPLE)

FIND_PATH(XSD_INCLUDE_DIR xsd/cxx/parser/elements.hxx
  PATHS "[HKEY_CURRENT_USER\\software\\xsd\\include]"
  "[HKEY_CURRENT_USER]\\xsd\\include]"
  "$ENV{XSDDIR}/include"
  "$ENV{XSDDIR}/libxsd"
  /usr/local/include
  /usr/include
  "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/include"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/include"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/include"
  ${CMAKE_SOURCE_DIR}/../xsd/libxsd
  "${XSD_ROOT_DIR}/include"
  "${XSD_ROOT_DIR}/libxsd"
  "${XSD_ROOT_DIR}"
)

FIND_PROGRAM(XSD_EXECUTABLE 
  NAMES xsdcxx xsd
  PATHS "[HKEY_CURRENT_USER\\xsd\\bin]" $ENV{XSDDIR}/bin
  "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/bin"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/bin"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/bin"
  "$ENV{PATH}"
  "${XSD_ROOT_DIR}"
  "${XSD_ROOT_DIR}/bin"
)

IF (NOT XSD_INCLUDE_DIR)
 SET(XSD_VERSION "0")
ELSE(NOT XSD_INCLUDE_DIR)
 FILE(READ ${XSD_INCLUDE_DIR}/xsd/cxx/version.hxx XVERHXX)
 STRING(REGEX MATCHALL "\n *#define XSD_INT_VERSION +[0-9]+" XVERINT ${XVERHXX})
 STRING(REGEX REPLACE "\n *#define XSD_INT_VERSION +" "" XVERINT ${XVERINT})
 STRING(REGEX REPLACE "....$" "" XVERINT ${XVERINT})
 STRING(REGEX MATCHALL "..$" XVERMIN ${XVERINT})
 STRING(REGEX REPLACE "..$" "" XVERMAJ ${XVERINT})

 SET(XSD_VERMAJ ${XVERMAJ})
 SET(XSD_VERMIN ${XVERMIN})
ENDIF (NOT XSD_INCLUDE_DIR)

if (NOT XSD_INCLUDE_DIR)
  message (FATAL_ERROR "Unable to find xsd include files (xsd/cxx/parser/elements.hxx)")
else (NOT XSD_INCLUDE_DIR)
  if(NOT XSD_FIND_QUIETLY)
    message (STATUS "Found xsd: " ${XSD_INCLUDE_DIR})
    message (STATUS "         : " ${XSD_EXECUTABLE})
    message (STATUS "xsd Ver. : " ${XSD_VERMAJ} "." ${XSD_VERMIN})
  endif (NOT XSD_FIND_QUIETLY)
endif (NOT XSD_INCLUDE_DIR)

if (NOT XSD_EXECUTABLE)
  message (FATAL_ERROR "Unable to find xsd or xsdcxx executable")
  SET(XSD_EXECUTABLE)
  UNSET(XSD_EXECUTABLE CACHE)
endif (NOT XSD_EXECUTABLE)

IF (NOT ((XSD_VERMAJ GREATER 2) AND (XSD_VERMIN GREATER 2)))
  SET(XSD_INCLUDE_DIR )
  UNSET(XSD_INCLUDE_DIR CACHE)
  message (FATAL_ERROR "XSD version number mismatch")
ENDIF (NOT ((XSD_VERMAJ GREATER 2) AND (XSD_VERMIN GREATER 2)))

#
# General CMake package configuration.
#
INCLUDE( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( XSD DEFAULT_MSG XSD_EXECUTABLE XSD_INCLUDE_DIR )

MARK_AS_ADVANCED( XSD_INCLUDE_DIR XSD_EXECUTABLE )

# 
# Macro that attempts to generate C++ files from an XML schema. The NAME
# argument is the name of the CMake variable to use to store paths to the
# derived C++ source file. The FILE argument is the path of the schema file to
# process. Additional arguments should be XSD command-line options.
#
# Example:
#
# XSD_SCHEMA( FOO_SRCS Foo.xsd --root-element-first --generate-serialization )
#
# On return, FOO_SRCS will contain Foo.cxx.
#
MACRO( XSD_SCHEMA NAME FILE )
  #
  # Make a full path from the soource directory
  #
  SET( xs_SRC "${FILE}" )

  # 
  # XSD will generate two or three C++ files (*.cxx,*.hxx,*.ixx). Get the
  # destination file path sans any extension and then build paths to the
  # generated files.
  #
  GET_FILENAME_COMPONENT( xs_FILE "${FILE}" NAME_WE )
  SET( xs_CXX "${CMAKE_CURRENT_BINARY_DIR}/${xs_FILE}.cxx" )
  SET( xs_HXX "${CMAKE_CURRENT_BINARY_DIR}/${xs_FILE}.hxx" )
#  SET( xs_IXX "${CMAKE_CURRENT_BINARY_DIR}/${xs_FILE}.ixx" )


  #
  # Add the source files to the NAME variable, which presumably will be used to
  # define the source of another target.
  #
  LIST( APPEND ${NAME} ${xs_CXX} )

  #
  # Set up a generator for the output files from the given schema file using
  # the XSD cxx-tree command.
  #
  ADD_CUSTOM_TARGET( genxmlxsd ALL)
  ADD_CUSTOM_COMMAND( TARGET genxmlxsd PRE_BUILD
			COMMAND ${XSD_EXECUTABLE}
			ARGS "cxx-tree" ${ARGN} ${xs_SRC}
			DEPENDS ${xs_SRC} )

  #
  # Don't fail if a generated file does not exist.
  #
  SET_SOURCE_FILES_PROPERTIES( "${xs_CXX}" "${xs_HXX}" "${xs_IXX}"
  							   PROPERTIES GENERATED TRUE )

ENDMACRO( XSD_SCHEMA )