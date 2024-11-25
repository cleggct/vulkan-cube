all:
	clang++ -std=c++17 -v -o -std=c++20 main.cpp -lglfw3 -framework Cocoa -framework IOKit -I/Users/cc/VulkanSDK/macOS/include -L/Users/cc/VulkanSDK/macOS/lib -lvulkan.1 -Wl,-rpath,/Users/cc/VulkanSDK/macOS/lib -o main
