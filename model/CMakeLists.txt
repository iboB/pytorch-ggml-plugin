add_library(model STATIC
    model/Model.cpp
)
target_include_directories(model INTERFACE .)
target_link_libraries(model PUBLIC ggml::ggml)
