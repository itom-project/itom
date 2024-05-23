cmake_minimum_required(VERSION 3.12...3.29)

if(WIN32 AND ((NOT DEFINED ${gtest_force_shared_crt}) OR (NOT ${gtest_force_shared_crt})))
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib." FORCE)
    message(WARNING "For properly running the unittests, the CMAKE variable gtest_force_shared_crt has been forced to ON.")
endif()

add_subdirectory(unittest_dataobject)
add_subdirectory(unittest_commonlib)
add_subdirectory(unittest_commonqtlib)
add_subdirectory(unittest_addinmanager)
add_subdirectory(unittest_qpropertyeditor)
