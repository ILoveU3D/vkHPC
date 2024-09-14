echo "构建项目"
cmake -S . -B build
cmake --build build

echo "编译着色器"
glslc shaders/headless.comp -o shaders/headless.comp.spv

echo "完成"
