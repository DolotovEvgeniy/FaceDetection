set(target ${TESTS})

file(GLOB srcs "*.cpp")

# VS2012 doesn't support correctly the tuples yet,
# see http://code.google.com/p/googletest/issues/detail?id=412
if(MSVC)
    add_definitions(/D _VARIADIC_MAX=10)
endif()

add_executable(${target} ${srcs} ${hdrs})
if (UNIX)
  target_link_libraries(${target} gtest ${CMAKE_THREAD_LIBS_INIT} pthread)
endif (UNIX)
target_link_libraries(${target} gtest ${LIBRARY})
