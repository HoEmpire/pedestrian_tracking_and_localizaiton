macro(add_tests_in_dir dirname)
    file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/test/${dirname}/*.cc")
    foreach (file ${files})
        get_filename_component(name ${file} NAME_WE)
        add_simple_test(${dirname} ${name})
    endforeach ()
endmacro()

# add executables with project library
macro(add_simple_test dirname name)
    add_executable(${dirname}_${name} ${CMAKE_CURRENT_SOURCE_DIR}/test/${dirname}/${name}.cc)
    target_link_libraries(${dirname}_${name} ${PROJECT_NAME} gtest_main)
    add_test(NAME ${dirname}_${name} COMMAND ${dirname}_${name})
    #    install(TARGETS ${name} DESTINATION bin)
endmacro()

macro(add_simple_apps)
    file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/app/*.cc")
    foreach (file ${files})
        get_filename_component(name ${file} NAME_WE)
        add_simple_app(${name})
    endforeach ()
endmacro()

# add executables with project library
macro(add_simple_app name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/app/${name}.cc)
    target_link_libraries(${name} ${PROJECT_NAME})
    install(TARGETS ${name}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
endmacro()

# setup third-party libraries
macro(setup_thirdparty_package name)
    # Download and unpack googletest at configure time
    configure_file(thirdparty/${name}/CMakeLists.txt.in ${name}-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}-download)
    if (result)
        message(FATAL_ERROR "CMake step for ${name} failed: ${result}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}-download)
    if (result)
        message(FATAL_ERROR "Build step for ${name} failed: ${result}")
    endif ()

    if ("${name}" STREQUAL "googletest")
        # Prevent overriding the parent project's compiler/linker
        # settings on Windows
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif ()

    # Add library directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${name}-src
        ${CMAKE_CURRENT_BINARY_DIR}/${name}-build
        EXCLUDE_FROM_ALL)

    if ("${name}" STREQUAL "googletest")
        # The gtest/gtest_main targets carry header search path
        # dependencies automatically when using CMake 2.8.11 or
        # later. Otherwise we have to add them here ourselves.
        if (CMAKE_VERSION VERSION_LESS 2.8.11)
            include_directories("${gtest_SOURCE_DIR}/include")
        endif ()
    endif ()
endmacro()
