add_library(mat mat.hip)
add_library(cat cat.hip)
add_library(quantum quantum.cpp)

# Create your target
add_executable(interact interact.cpp)
target_link_libraries(interact PRIVATE mat GL SDL2)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE quantum cat GL SDL2)

add_executable(decomp decomp.cpp)
target_link_libraries(decomp PRIVATE quantum cat GL SDL2)

add_executable(play play.cpp)
target_link_libraries(play PRIVATE quantum cat)

# Enable HIP language
set_target_properties(
  interact main play decomp
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             HIP_ARCHITECTURES "gfx1201")
