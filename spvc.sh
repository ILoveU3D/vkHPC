#!/bin/bash

echo "编译着色器..."
if [ ! -d "./shaders/spirv" ]; then
    mkdir ./shaders/spirv
fi
glslc ./shaders/headless.comp -o ./shaders/spirv/headless.comp.spv
echo "完成"