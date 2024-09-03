cmake_minimum_required(VERSION 3.15)

project(vkHPC)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# 寻找 Vulkan 和 GLFW 库
if(WIN32)
    set(Vulkan_INCLUDE_DIRS "$ENV{VULKAN_SDK}\\Include")
    set(Vulkan_LIBRARY "$ENV{VULKAN_SDK}\\Lib32")
else()
    find_package(Vulkan REQUIRED FATAL_ERROR)
endif()

# 根据选项定义 GPU
option(GPU "define your GPU" MOORETHREADS)
message("GPU is ${GPU}")
if(GPU STREQUAL "MooreThread")
    add_definitions(-DMOORETHREADS)
endif()

# 添加源文件
file(GLOB_RECURSE SOURCES "modules/*.cpp")

# 设置可执行文件
add_executable(main main.cpp ${SOURCES})

# 定义宏和资源路径
set(SHADER_PATH "${CMAKE_SOURCE_DIR}/shaders/spirv/")
add_definitions(-DSHADER_PATH="${SHADER_PATH}")

# set(ASSET_PATH "${CMAKE_SOURCE_DIR}/assets/")
# add_definitions(-DASSET_PATH="${ASSET_PATH}")

# 包含自定义目录和项目内部目录
target_include_directories(main PRIVATE
    # ./glm/
    ./include/
    ${Vulkan_INCLUDE_DIRS}  # Vulkan 包含目录
)

# 特定于 Windows 的设置
if(WIN32)
    # 确保链接到 Vulkan 和 GLFW 的动态库
    target_link_libraries(main
        PRIVATE
        ${Vulkan_LIBRARY}/vulkan-1.lib
    )
else()
    # 链接 Vulkan 和 GLFW 库
    target_link_libraries(main
        PRIVATE
        Vulkan::Vulkan  # Vulkan 库
    )
endif()