cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(include)

set(SOURCE_FILES include/neural/Network.h include/neural/Matrix.h src/Network.cpp)

add_executable(NeuralNetwork main.cpp ${SOURCE_FILES})
add_executable(NeuralNetworkAsan main.cpp ${SOURCE_FILES})

target_compile_options(NeuralNetworkAsan
        PUBLIC "-fsanitize=address"
        PUBLIC "-fno-omit-frame-pointer"
        PUBLIC "-fvisibility=hidden")

target_link_libraries(NeuralNetwork PUBLIC ${OpenCV_LIBS})
target_link_libraries(NeuralNetworkAsan
        PUBLIC ${OpenCV_LIBS}
        -fsanitize=address
        -fno-omit-frame-pointer)