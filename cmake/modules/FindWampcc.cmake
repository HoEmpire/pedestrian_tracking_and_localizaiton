find_package(PkgConfig)
pkg_check_modules(PC_WAMPCC QUIET Wampcc)

find_path(WAMPCC_INCLUDE_DIR
    NAMES wampcc.h
    PATHS ${PC_WAMPCC_INCLUDE_DIRS}
    PATH_SUFFIXES wampcc
    )
find_library(WAMPCC_LIBRARY
    NAMES wampcc
    PATHS ${PC_WAMPCC_LIBRARY_DIRS}
    )
find_library(WAMPCC_JSON_LIBRARY
    NAMES wampcc_json
    PATHS ${PC_WAMPCC_LIBRARY_DIRS}
    )

set(WAMPCC_VERSION ${PC_WAMPCC_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(wampcc
    FOUND_VAR WAMPCC_FOUND
    REQUIRED_VARS
    WAMPCC_LIBRARY
    WAMPCC_INCLUDE_DIR
    VERSION_VAR WAMPCC_VERSION
    )

if (WAMPCC_FOUND)
    list(APPEND WAMPCC_ALL_LIBRARY ${WAMPCC_LIBRARY} ${WAMPCC_JSON_LIBRARY})
    set(WAMPCC_LIBRARIES ${WAMPCC_ALL_LIBRARY})
    set(WAMPCC_INCLUDE_DIRS ${WAMPCC_INCLUDE_DIR})
    set(WAMPCC_DEFINITIONS ${PC_WAMPCC_CFLAGS_OTHER})
    message(STATUS "Found wampcc (include: ${WAMPCC_INCLUDE_DIRS}, library: ${WAMPCC_LIBRARIES})")
    mark_as_advanced(
        WAMPCC_INCLUDE_DIR
        WAMPCC_LIBRARY
    )
endif ()

