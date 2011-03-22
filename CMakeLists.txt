cmake_minimum_required (VERSION 2.6)

project (Example)
link_libraries(OpenCL)
add_executable(Example example.c clrand.c)
