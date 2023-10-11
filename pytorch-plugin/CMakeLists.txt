# Copyright (c) Borislav Stanimirov
# SPDX-License-Identifier: MIT
#
add_library(pytorch-plugin MODULE
    pytorch-plugin.cpp
)
target_link_libraries(pytorch-plugin PRIVATE
    ggml::ggml
    torch
)
