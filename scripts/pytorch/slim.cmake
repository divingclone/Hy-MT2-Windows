# Injected with CMAKE_PROJECT_Torch_INCLUDE, against the pinned upstream commit.
# Build only the CUDA backend; the shipping CPU/Python ABI stays unchanged.
function(hymt_customize_cuda)
  if(NOT WIN32 OR NOT MSVC OR NOT TARGET torch_cuda)
    message(FATAL_ERROR "HyMT CUDA customization requires Windows MSVC and torch_cuda")
  endif()
  if(NOT CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0 OR
     NOT CUDAToolkit_VERSION VERSION_LESS 13.1)
    message(FATAL_ERROR "HyMT CUDA customization is pinned to CUDA 13.0")
  endif()
  if(NOT DEFINED ENV{HYMT_BASE_TORCH})
    message(FATAL_ERROR "HYMT_BASE_TORCH must name the pinned torch package")
  endif()
  file(TO_CMAKE_PATH "$ENV{HYMT_BASE_TORCH}" base)
  foreach(name torch_cpu c10 c10_cuda)
    if(NOT EXISTS "${base}/lib/${name}.lib" OR NOT EXISTS "${base}/lib/${name}.dll")
      message(FATAL_ERROR "Missing original import library or DLL: ${name}")
    endif()
    add_library(hymt_original_${name} SHARED IMPORTED GLOBAL)
    set_target_properties(hymt_original_${name} PROPERTIES
      IMPORTED_IMPLIB "${base}/lib/${name}.lib"
      IMPORTED_LOCATION "${base}/lib/${name}.dll")
    foreach(prop INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS
                 INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
      set_property(TARGET hymt_original_${name} PROPERTY ${prop}
        "$<TARGET_PROPERTY:${name},${prop}>")
    endforeach()
  endforeach()
  foreach(prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(libraries torch_cuda ${prop})
    list(TRANSFORM libraries REPLACE "^torch_cpu_library$" "hymt_original_torch_cpu")
    list(TRANSFORM libraries REPLACE "^c10_cuda$" "hymt_original_c10_cuda")
    list(APPEND libraries hymt_original_c10)
    set_property(TARGET torch_cuda PROPERTY ${prop} "${libraries}")
  endforeach()
  # Generate headers without compiling the CPU backend.
  add_dependencies(torch_cuda ATEN_CPU_FILES_GEN_TARGET)
  target_sources(torch_cuda PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/delay_load.cpp")
  target_link_libraries(torch_cuda PRIVATE delayimp)
  foreach(dll cufft64_12.dll cusparse64_12.dll cusolver64_12.dll)
    target_link_options(torch_cuda PRIVATE "/DELAYLOAD:${dll}")
  endforeach()
  message(STATUS "HyMT: using original CPU/c10 binaries; FFT/sparse/solver are delay-loaded")
endfunction()

cmake_language(DEFER CALL hymt_customize_cuda)
