#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <cassert>

// NECESSARY ON MACOS
#define VK_USE_PLATFORM_MACOS_MVK
// NECESSARY ON MACOS ARM (M1, M2, etc. processors)

#if (defined(VK_USE_PLATFOR_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT))
#define VK_ENABLE_BETA_EXTENSIONS
#endif

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>

#define DEBUG (!NDEBUG)

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
const std::vector<const char*> optionalDeviceExtensions = {
  "VK_KHR_portability_subset"    
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
      VkVertexInputBindingDescription bindingDescription{};
      bindingDescription.binding = 0;
      bindingDescription.stride = sizeof(Vertex);
      bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
      std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

      // the binding parameter tells us from which binding the per-vertex data comes
      // we are using a single array for all our vertex data so there is only one binding,
      // hence we select the first
      attributeDescriptions[0].binding = 0;
      // the location parameter references the location directive of the input in the vertex
      //  shader corresponding to this attribute
      // in our case it is the vertex position, given as a vec2 of 32-bit floats
      attributeDescriptions[0].location = 0;
      // the format parameter tells us the format of the attribute
      // below we have each listed the possible formats with their corresponding
      //  handles for this parameter (they are a bit odd):
      // 
      // float: VK_FORMAT_R32_SFLOAT
      // vec2:  VK_FORMAT_R32G32_SFLOAT
      // vec3:  VK_FORMAT_R32G32B32_SFLOAT
      // vec4:  VK_FORMAT_R32G32B32A32_SFLOAT
      // 
      // ivec2: VK_FORMAT_R32G32_SINT, a 2-component vector of 32-bit signed integers
      // uvec4: VK_FORMAT_R32G32B32A32_UINT, a 4-component vector of 32-bit unsigned integers
      // double: VK_FORMAT_R64_SFLOAT, a double-precision (64-bit) float
      //
      // since the position attribute is a vec2, we set this parameter to VK_FORMAT_R32G32_SFLOAT
      attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      // the format parameter (previous) implicitly determines the byte size of the attribute data.
      // the offset parameter specifies how many bytes since the start of the per-vertex data to
      //  read from
      // the offest is automatically computed using the offsetof macro
      attributeDescriptions[0].offset = offsetof(Vertex, pos);

      // largely the same as for the previous attribute
      attributeDescriptions[1].binding = 0;
      attributeDescriptions[1].location = 1;
      attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributeDescriptions[1].offset = offsetof(Vertex, color);

      return attributeDescriptions;
    }
};

struct Volume {
  alignas(16) glm::vec4 volume;
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}}

};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    3, 2, 6, 6, 7, 3,
    5, 4, 7, 7, 6, 5,
    4, 0, 3, 3, 7, 4,
    4, 5, 1, 1, 0, 4,
    1, 5, 6, 6, 2, 1
};

/*
 * Helper for reading files.  We use this to load our SPIR-V shaders
 */
static std::vector<char> readFile(const std::string& filename) {

#if DEBUG
  std::cout << "Reading file " << filename << "...\n";
#endif

  // read the file starting from the end and in binary (no text conversions)
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  // get file size from the position of the cursor (since we start reading
  // from the end of the file)
  size_t fileSize = (size_t) file.tellg();
  // allocate a buffer for the file
  std::vector<char> buffer(fileSize);

  // go back to beginning and read bytes into buffer
  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

#if DEBUG
  std::cout <<  "File " << filename << " read successfully.\n";
#endif

  return buffer;
}

/*
 * This one's weird.  In order to get debug tools we need to set up something
 * called a messenger which in this case tells vulkan where to find our debug
 * callback.  Setting up this messenger involves passing a struct representing
 * various attributes of the messenger to a special function, which has to be
 * loaded from an extension.  The function defined below is responsible for 
 * loading the extension function and passing the messenger struct to it.
 */
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}
/*
 * The debug messenger also needs to be cleaned up on program exit.  This function
 * takes care of that.
 */
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}



/*
 * This class contains all the vulkan code we need to set up our application
 *
 */
class VulkanApplication {
  public:
    void run() {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }

  private:

    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // don't worry about cleaning this up
    
    VkDevice device;
    
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue transferQueue;
    
    VkSwapchainKHR swapChain;
    
    std::vector<VkImage> swapChainImages; // this doesn't need cleanup
    
    VkFormat swapChainImageFormat; // or this
    
    VkExtent2D swapChainExtent; // or this
    
    std::vector<VkImageView> swapChainImageViews;
    
    VkRenderPass renderPass;
    
    VkDescriptorSetLayout descriptorSetLayout;
    
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    
    std::vector<VkFramebuffer> swapChainFramebuffers;
    
    VkCommandPool renderCommandPool;
    VkCommandPool transferCommandPool;
    
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    VkImageView textureImageView;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    
    std::vector<VkCommandBuffer> commandBuffers; // this doesn't need cleanup
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
   
    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    
    void initWindow() {
      glfwInit();

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // disable window resize

      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
      glfwSetWindowUserPointer(window, this);
      glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    /*
     * This function initializes all our vulkan state.
     */
    void initVulkan() {
      createInstance();
      setUpDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createDescriptorSetLayout();
      createGraphicsPipeline();
      createFramebuffers();
      createCommandPools();
      createVertexBuffer();
      createIndexBuffer();
      createUniformBuffers();
      createDescriptorPool();
      createDescriptorSets();
      createCommandBuffers();
      createSyncObjects();
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
      auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
      app->framebufferResized = true;
    }

    /*
     * The main loop of the program
     */
    void mainLoop() {
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
      }

      // wait for the CPU to finish doing what it's doing before exiting the main loop
      vkDeviceWaitIdle(device);
    }

    /*
     * Clean up all our allocated resources
     */
    void cleanup() {
      cleanupSwapChain();
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
      }
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
      vkDestroyBuffer(device, indexBuffer, nullptr);
      vkFreeMemory(device, indexBufferMemory, nullptr);
      vkDestroyBuffer(device, vertexBuffer, nullptr);
      vkFreeMemory(device, vertexBufferMemory, nullptr);
      vkDestroyPipeline(device, graphicsPipeline, nullptr);
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyRenderPass(device, renderPass, nullptr);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyFence(device, inFlightFences[i], nullptr);
      }

      vkDestroyCommandPool(device, renderCommandPool, nullptr);

      vkDestroyDevice(device, nullptr);


      // when in debug mode we must destroy the debug messenger
#if DEBUG
      // you can comment this line out to test that validation layers work...
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
#endif 
      // all other vulkan resources should be deallocated before destroying
      // the vulkan instance
      vkDestroySurfaceKHR(instance, surface, nullptr);
      vkDestroyInstance(instance, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();
    }

    void createInstance() {
      // if validation layers are enabled, check that the validation layer
      // extension is supported
#if DEBUG
      if (!checkValidationLayerSupport()) {
        throw std::runtime_error(
            "validation layers requested, but not available!");
      }
#endif
      // The VkApplicationInfo struct stores info about the application
      // which the driver uses for optimization
      VkApplicationInfo appInfo = VkApplicationInfo();
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = "Hello Triangle";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;

      // VkInstanceCreateInfo struct sets up our validation layers and global
      // extensions
      VkInstanceCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;

      // this sets up our required extensions
      std::vector<const char*> extensions = getRequiredExtensions();
#if (defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT))
      // When running on macOS with MoltenVK, enable VK_KHR_get_physical_device_properties2 (required by VK_KHR_portability_subset)
      extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#if defined(VK_KHR_portability_enumeration)
      // When running on macOS with MoltenVK, enable VK_KHR_portability_enumeration is defined and supported
      uint32_t instanceExtCount = 0;
      vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, nullptr);
      if (instanceExtCount > 0) {
        std::vector<VkExtensionProperties> instanceExtensions(instanceExtCount);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, &instanceExtensions.front()) == VK_SUCCESS) {
          for (VkExtensionProperties extension : instanceExtensions) {
            if (strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
              /* this stuff is necessary on MacOS to avoid encountering
               * VK_ERROR_INCOMPATIBLE_DRIVER when we try to create our vulkan instance
               * 
               * From the documentation:
               *      VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR specifies that
               *      the instance will enumerate available 
               *      Vulkan Portability-compliant physical devices and groups in 
               *      addition to the Vulkan physical devices and groups that are 
               *      enumerated by default.
               */
              extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
              createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
              break;
            }
          }
        }
      }
#endif
#endif
      createInfo.enabledExtensionCount = 
        static_cast<uint32_t>(extensions.size());
      createInfo.ppEnabledExtensionNames = extensions.data();

      // we create the debugCreateInfo struct outside of the if-statement to
      // ensure that it exists when vkCreateInstance is called
      VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
      // if validation layers are enabled
      if (enableValidationLayers) {
        // set up validation layers
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        // link up diagnostics for the vkCreateInstance and vkDestroyInstance calls
        // we use an extra debug messenger for this, but it is cleaned up
        // automatically so don't worry about having to destroy it later
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
      } else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
      }

      // create a vulkan instance and check for success
      const VkResult status = 
        vkCreateInstance(&createInfo, nullptr, &instance);
      assert(status == VK_SUCCESS);
    }

    /*
     * This function gets us debug info for the vkCreateInstance and vkDestroyInstance
     * calls, which we pass to vkCreateDebugUtilsMessengerEXT in order to make
     * this information visible to us in debug mode.
     */
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
      createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      createInfo.pfnUserCallback = debugCallback;
    }

    /*
     * This is our debug callback function.
     */
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {

      std::cerr << "validation layer: " << 
        pCallbackData->pMessage << std::endl;

      return VK_FALSE;
    }

    /*
     * This function sets up the whole debug messenger system
     */
    void setUpDebugMessenger() {
      if (!enableValidationLayers) return;

      // set up diagnostics for the vkCreateInstance and vkDestroyInstance functions
      VkDebugUtilsMessengerCreateInfoEXT createInfo;
      populateDebugMessengerCreateInfo(createInfo);

      // create the debug messenger for our vulkan instance
      const VkResult createDebugUtilsMessengerStatus = CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, 
            &debugMessenger);
      assert(createDebugUtilsMessengerStatus == VK_SUCCESS);
    }

    /*
     * Returns a vector of C strings representing the names of the required extensions.
     */
    std::vector<const char*> getRequiredExtensions() {
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

      // This extension is only required in debug mode i.e.  when validation layers
      // are enabled, it lets us use the debug messenger.
#if DEBUG
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

      // get supported extensions...
      uint32_t supportedExtensionCount = 0;

      vkEnumerateInstanceExtensionProperties(
          nullptr, &supportedExtensionCount, nullptr);

      std::vector<VkExtensionProperties> supportedExtensions(
          supportedExtensionCount);

      vkEnumerateInstanceExtensionProperties(
          nullptr, &supportedExtensionCount, supportedExtensions.data());

#if DEBUG
        std::cout << "available extensions:\n";

        for (const auto& extension : supportedExtensions) {
          std::cout << '\t' << extension.extensionName << '\n';
        }
#endif
      return extensions;
    }

    /*
     * Lists available layers and checks that each of the layers we need 
     * is in the set of available layers.  Returns true if so, false otherwise.
     * @return: true if each of the layers our program requests is in
     *      the set of available layers, false otherwise
     */
    bool checkValidationLayerSupport() {

      // get available layers and list them
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

      // check that each of our validation layers is in the set of available layers
      for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
          if (strcmp(layerName, layerProperties.layerName) == 0) {
            layerFound = true;
            break;
          }
        }

        if (!layerFound) {
          return false;
        }
      }

      return true;
    }

    /*
     * This function is responsible for creating the rendering surface
     */
    void createSurface() {
      // create a surface with GLFW
      const VkResult STATUS = glfwCreateWindowSurface(instance, window, nullptr, &surface);
      assert(STATUS == VK_SUCCESS);
    }

    /*
     * IMPORTANT: This function is responsible for selecting a physical device
     */
    void pickPhysicalDevice() {
      uint32_t deviceCount = 0;
      vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

      assert(deviceCount > 0);

      std::vector<VkPhysicalDevice> devices(deviceCount);
      vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

      for (const auto& device : devices) {

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        const char* name = deviceProperties.deviceName;

#if DEBUG
        std::cout << "Physical device discovered: " << name << "\n";
#endif
        if (isDeviceSuitable(device)) {
#if DEBUG
          std::cout << "Selecting physical device " << name << "\n";
#endif
          physicalDevice = device;                
          break;
        }
      }

      if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
      }
    }
    /*
     * Checks if the given device is suitable for our operations.  Returns true if the device
     * has at least one queue family which supports graphics and one which supports
     * presentation.
     */
    bool isDeviceSuitable(VkPhysicalDevice device) {
      QueueFamilyIndices indices = findQueueFamilies(device);

      // Check the device extensions are supported
      bool extensionsSupported = checkDeviceExtensionSupport(device);

      // Check that swap chain support is adequate
      bool swapChainAdequate = false;
      if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      }

      return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    /*
     * This function is responsible for checking the device supports all required
     * extensions
     */
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
      uint32_t extensionCount;
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

      std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

      for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
      }



      return requiredExtensions.empty();
    }

    struct QueueFamilyIndices {
      std::optional<uint32_t> graphicsFamily;
      std::optional<uint32_t> presentFamily;
      std::optional<uint32_t> transferFamily;

      bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
      }
    };
    /*
     * Gets indices of queue families which support graphics and/or presentation
     */
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
      QueueFamilyIndices indices;

      // Retrieve the list of queue families
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(
          device, &queueFamilyCount, queueFamilies.data());
      // Get the indices of queue families that support graphics, presentation and
      // data transfer
      int i = 0;
      for (const auto& queueFamily : queueFamilies) {

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if ((!indices.presentFamily.has_value()) && presentSupport) {
          indices.presentFamily = i;
        }

        if ((!indices.graphicsFamily.has_value()) && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
          indices.graphicsFamily = i;
        }

        if ((!indices.transferFamily.has_value()) && (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)) {
          indices.transferFamily = i;
        }

        if (indices.isComplete()) {
          break;
        }

        i++;
      }
      return indices;
    }

    /*
     * This function sets up a logical device for us
     */
    void createLogicalDevice() {
      // this struct specifies that we want 1 queue per queue family
      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
      std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
      std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value(), indices.transferFamily.value()};

      // we need to check whether the graphics queue family and the presentation
      // queue family are the same; if they are then we must take care not to
      // initialize the queue twice
      float queuePriority = 1.0f;
      for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
      }

      // specifies device features that we will be using
      VkPhysicalDeviceFeatures deviceFeatures{};

      // get the list of all supported extensions
      uint32_t extensionCount;
      vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

      // we must include all required as well as any available optional extensions
      std::vector<const char*> enabledExtensions = deviceExtensions;

#if VK_KHR_portability_enumeration
      for (const auto& extension : availableExtensions) {
        const char* name = extension.extensionName;

        if (strcmp(name, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0) {
          enabledExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

#if DEBUG
              std::cout << "Optional device extension enabled: " << name << "\n";
#endif
          }
        }
#endif

      // here is the createInfo struct for the logical device
      VkDeviceCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

      // add pointers to the queue creation info and device features structs
      createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
      createInfo.pQueueCreateInfos = queueCreateInfos.data();
      createInfo.pEnabledFeatures = &deviceFeatures;

      createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
      createInfo.ppEnabledExtensionNames = enabledExtensions.data();

      // this stuff actually isn't used by newer vulkan implementations but
      // is included for compatibility reasons
      if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
      } else {
        createInfo.enabledLayerCount = 0;
      }

      // instantiate the logical device from the physical device we picked earlier
      const VkResult STATUS = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
      assert(STATUS == VK_SUCCESS);

      // bind the queue handles for the graphics and presentation queues
      vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
      vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
      vkGetDeviceQueue(device, indices.transferFamily.value(), 0, &transferQueue);
    }

    struct SwapChainSupportDetails {
      VkSurfaceCapabilitiesKHR capabilities;
      std::vector<VkSurfaceFormatKHR> formats;
      std::vector<VkPresentModeKHR> presentModes;
    };
    /*
     * Handles all the fine details of setting up the swap chain
     */
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
      SwapChainSupportDetails details;

      // This gets us the basic surface capabilities
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

      // Next we query the supported surface formats
      uint32_t formatCount;
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

      if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            device, surface, &formatCount, details.formats.data());
      }

      // Query supported presentation modes
      uint32_t presentModeCount;
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

      if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device, surface, &presentModeCount, details.presentModes.data());
      }

      return details;
    }

    /*
     * Sets up the surface format and colorspace, with an automatic preference for SRGB
     */
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
      for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
          return availableFormat;
        }
      }

      // if there were no SRGB formats then just give us the first available format
      return availableFormats[0];
    }

    /*
     * Chooses the swap chain presentation mode, favoring VK_PRESENT_MODE_MAILBOX_KHR,
     * in which images submitted by the application are added to a fixed-size queue
     * with the oldest images being dropped if the queue fills
     * If this mode isn't available, we go with VK_PRESENT_MODE_FIFO_KHR which
     * is guaranteed to be available, in which images are added to a fixed-size
     * queue but if the queue is full then the program must wait
     */
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
      for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
          return availablePresentMode;
        }
      }

      return VK_PRESENT_MODE_FIFO_KHR;
    }

    /*
     *  This function sets the swap extent.  The swap extent is the resolution of
     *  the swap chain images, and here it is equal to the resolution of the
     *  window we are drawing to in pixels
     */
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

      // in the case of this if-statement, we have no choice but to use the
      // values specified in currentExtent
      if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
      } else {
        // however, in this case we are allowed to choose a resolution so we get
        // the pixel dimensions of the window from GLFW and clamp them within our range
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
      }
    }

    /*
     * This function is responsible for creating the swap chain
     */
    void createSwapChain() {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

      VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
      VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
      VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

      // get minimum required number of images in the swap chain and add 1,
      // this will be the number of images in the swap chain
      uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

      // take care not to exceed the maximum allowable size of the swap chain
      if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
      }

      // set up swap chain details
      VkSwapchainCreateInfoKHR createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      createInfo.surface = surface;

      // set up the swap chain image details
      createInfo.minImageCount = imageCount;
      createInfo.imageFormat = surfaceFormat.format;
      createInfo.imageColorSpace = surfaceFormat.colorSpace;
      createInfo.imageExtent = extent;
      createInfo.imageArrayLayers = 1; // the number of layers in each swap chain image
      createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
      uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value(), indices.transferFamily.value()};

      if ((indices.graphicsFamily != indices.presentFamily) || (indices.graphicsFamily != indices.transferFamily)) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;

        if (indices.presentFamily.value() != indices.transferFamily.value()) {
          createInfo.queueFamilyIndexCount = 3;
        } else {
          createInfo.queueFamilyIndexCount = 2;
        }
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
      } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
      }

      createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

      createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

      createInfo.presentMode = presentMode;
      createInfo.clipped = VK_TRUE;

      createInfo.oldSwapchain = VK_NULL_HANDLE;

      // create the swap chain
      const VkResult STATUS = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
      assert(STATUS == VK_SUCCESS);

      // store handles to the swap chain images
      vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
      swapChainImages.resize(imageCount);
      vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

      swapChainImageFormat = surfaceFormat.format;
      swapChainExtent = extent;
    }

    /*
     * This function is responsible for creating the image views for the swap chain. We need
     * an image view object to store each of the swap chain images, in order for us to be able
     * to do anything with the images.
     */
    void createImageViews() {
      swapChainImageViews.resize(swapChainImages.size());

      // iterate over swap chain
      for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];

        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;

        // use default mappings
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        // create image view
        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create image views!");
        }
      }
    }

    /*
     * This function is responsible for setting up the graphics pipeline
     */
    void createGraphicsPipeline() {

      // load pre-compiled SPIR-V bytecode
      auto vertShaderCode = readFile("shaders/vert.spv");
      auto fragShaderCode = readFile("shaders/frag.spv");

      VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
      VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

      // assign the shaders to a pipeline stage
      VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
      vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
      // this line specifies the shader module containing the code
      vertShaderStageInfo.module = vertShaderModule;
      // this line specifies the entrypoint
      vertShaderStageInfo.pName = "main";

      VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
      fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragShaderStageInfo.module = fragShaderModule;
      fragShaderStageInfo.pName = "main";


      // add the configuration structs to this array so we can reference them
      // in the pipeline creation step
      VkPipelineShaderStageCreateInfo shaderStages[] = 
      {vertShaderStageInfo, fragShaderStageInfo};

      // get vertex data descriptions
      auto bindingDescription = Vertex::getBindingDescription();
      auto attributeDescriptions = Vertex::getAttributeDescriptions();

      VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
      vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vertexInputInfo.vertexBindingDescriptionCount = 1;
      vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
      vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
      vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

      // specify what type of primitives we are drawing, with the options:
      // VK_PRIMITIVE_TOPOLOGY_POINT_LIST - points from vertices
      // VK_PRIMITIVE_TOPOLOGY_LINE_LIST - line from every 2 vertices without
      //      re-use
      // VK_PRIMITIVE_TOPOLOGY_LINE_STRIP - the end vertex of every line is 
      //      the start vertex of the next
      // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST - triangle from every 3 vertices
      //      without re-use
      // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP - the second and third vertex
      //      of every triangle are used as first two vertices of next triangle
      VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
      inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      inputAssembly.primitiveRestartEnable = VK_FALSE;

      // This code is for creating a static viewport. Be sure to disable the
      //      dynamic viewport stuff before using this.
      // VkViewport viewport{};
      // viewport.x = 0.0f;
      // viewport.y = 0.0f;
      // viewport.width = (float) swapChainExtent.width;
      // viewport.height = (float) swapChainExtent.height;
      // viewport.minDepth = 0.0f;
      // viewport.maxDepth = 1.0f;
      // 
      // VkRect2D scissor{};
      // scissor.offset = {0, 0};
      // scissor.extent = swapChainExtent;

      VkPipelineViewportStateCreateInfo viewportState{};
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.scissorCount = 1;


      // Set up the rasterizer
      VkPipelineRasterizationStateCreateInfo rasterizer{};
      rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterizer.depthClampEnable = VK_FALSE;

      // setting this to VK_TRUE causes the rasterizer stage to be skipped,
      //      which basically disables any output to the framebuffer
      rasterizer.rasterizerDiscardEnable = VK_FALSE;

      // This setting determines how fragments are generated for geometry.
      // It takes the following modes:
      // VK_POLYGON_MODE_FILL - fill the area of the polygon with fragments
      // VK_POLYGON_MODE_LINE - polygon edges are drawn as lines
      // VK_POLYGON_MODE_POINT - polygon vertices are drawn as points
      //
      // Using any mode other than fill requires enabling a GPU feature
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

      // specifies the thickness of lines in fragments
      // any line thicker than 1.0f requires you to enable the wideLines GPU feature
      rasterizer.lineWidth = 1.0f;

      // specifies type of culling and vertex order for faces to be considered
      //      front-facing.
      rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

      // depth bias settings - this stuff isn't used much
      rasterizer.depthBiasEnable = VK_FALSE;
      rasterizer.depthBiasConstantFactor = 0.0f; // Optional
      rasterizer.depthBiasClamp = 0.0f; // Optional
      rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

      // multisample settings
      VkPipelineMultisampleStateCreateInfo multisampling{};
      multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisampling.sampleShadingEnable = VK_FALSE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      multisampling.minSampleShading = 1.0f; // Optional
      multisampling.pSampleMask = nullptr; // Optional
      multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
      multisampling.alphaToOneEnable = VK_FALSE; // Optional

      // per-framebuffer color blend configuration
      VkPipelineColorBlendAttachmentState colorBlendAttachment{};
      colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      colorBlendAttachment.blendEnable = VK_FALSE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

      // enable alpha blending
      colorBlendAttachment.blendEnable = VK_TRUE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

      VkPipelineColorBlendStateCreateInfo colorBlending{};
      colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      colorBlending.logicOpEnable = VK_FALSE;
      colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
      colorBlending.attachmentCount = 1;
      colorBlending.pAttachments = &colorBlendAttachment;
      colorBlending.blendConstants[0] = 0.0f; // Optional
      colorBlending.blendConstants[1] = 0.0f; // Optional
      colorBlending.blendConstants[2] = 0.0f; // Optional
      colorBlending.blendConstants[3] = 0.0f; // Optional

      // This code is for creating a dynamic viewport.
      std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
      };
      VkPipelineDynamicStateCreateInfo dynamicState{};
      dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamicState.pDynamicStates = dynamicStates.data();


      // create the pipeline layout object
      VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
      pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutInfo.setLayoutCount = 1; // Optional
      pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
      pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
      pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

      if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
      }

      VkGraphicsPipelineCreateInfo pipelineInfo{};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipelineInfo.stageCount = 2;
      pipelineInfo.pStages = shaderStages;

      pipelineInfo.pVertexInputState = &vertexInputInfo;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState = &multisampling;
      pipelineInfo.pDepthStencilState = nullptr; // Optional
      pipelineInfo.pColorBlendState = &colorBlending;
      pipelineInfo.pDynamicState = &dynamicState;

      pipelineInfo.layout = pipelineLayout;

      pipelineInfo.renderPass = renderPass;
      pipelineInfo.subpass = 0;

      pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
      pipelineInfo.basePipelineIndex = -1; // Optional

      if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
      }

      // cleanup code, this should come last
      vkDestroyShaderModule(device, fragShaderModule, nullptr);
      vkDestroyShaderModule(device, vertShaderModule, nullptr);

    }

    /*
     * Helper function which takes a buffer with shader bytecode as its parameter
     * and creates a VkShaderModule from it
     */
    VkShaderModule createShaderModule(const std::vector<char>& code) {
      VkShaderModuleCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.codeSize = code.size();
      createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

      VkShaderModule shaderModule;
      if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
      }
      return shaderModule;
    }

    void createRenderPass() {

      // here we set up a single color buffer attachment represented by one
      // of the images from the swap chain
      // whatever that means...
      // the format of the color attachment should match the format of the
      // swap chain images
      // we're not doing anything with multisampling, so use 1 sample
      VkAttachmentDescription colorAttachment{};
      colorAttachment.format = swapChainImageFormat;
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

      colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

      colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

      colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

      VkAttachmentReference colorAttachmentRef{};
      colorAttachmentRef.attachment = 0;
      colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

      VkSubpassDescription subpass{};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &colorAttachmentRef;

      VkRenderPassCreateInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.attachmentCount = 1;
      renderPassInfo.pAttachments = &colorAttachment;
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;

      VkSubpassDependency dependency{};
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

      renderPassInfo.dependencyCount = 1;
      renderPassInfo.pDependencies = &dependency;

      if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
      }
    }

    /*
     * This function creates the framebuffers. It runs after we have created the graphics
     * pipeline and render pass. The framebuffers are stored in the swapChainFramebuffers
     * data structure.
     */
    void createFramebuffers() {
      //resize the framebuffer data structure to hold all the framebuffers
      swapChainFramebuffers.resize(swapChainImageViews.size());

      for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
          swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create framebuffer!");
        }
      }
    }

    /*
     * This function creates the command pools, the objects for storing the command buffer objects
     * from which we issue vulkan commands. The command pools are responsible for managing the
     * memory used to store the command buffers. Command buffers are allocated from the command pool.
     */
    void createCommandPools() {
      QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

      VkCommandPoolCreateInfo renderPoolInfo{};
      renderPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      renderPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      renderPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
      
      VkCommandPoolCreateInfo transferPoolInfo{};
      transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      transferPoolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();

      const VkResult renderCommandPoolStatus = vkCreateCommandPool(device, &renderPoolInfo, nullptr, &renderCommandPool);
      assert(renderCommandPoolStatus == VK_SUCCESS);

      const VkResult transferCommandPoolStatus = vkCreateCommandPool(device, &transferPoolInfo, nullptr, &transferCommandPool);
      assert(transferCommandPoolStatus == VK_SUCCESS);
    }

    /*
     * This function allocates a single command buffer from the command pool. Command buffers
     * allow us to issue commands to vulkan, for things like drawing etc.
     */
    void createCommandBuffers() {
      commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

      VkCommandBufferAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = renderCommandPool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

      if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffers[currentFrame]) != VK_SUCCESS) {
          throw std::runtime_error("failed to allocate command buffers!");
      }
    }

    /* 
     * This function writes the commands we wish to execute into the supplied command buffer.
     */
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
      // To begin recording a command buffer we must populate this struct which
      // specifies some details about the usage of this command buffer.
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0; // Optional
      beginInfo.pInheritanceInfo = nullptr; // Optional

      if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
          throw std::runtime_error("failed to begin recording command buffer!");
      }

      // configure render pass
      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;
    
      VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      // begin the render pass!
      vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      // bind the graphics pipeline :)
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = static_cast<float>(swapChainExtent.width);
      viewport.height = static_cast<float>(swapChainExtent.height);
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

      VkRect2D scissor{};
      scissor.offset = {0, 0};
      scissor.extent = swapChainExtent;
      vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
      
      VkBuffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

      vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

      // draw
      vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

      // end render pass
      vkCmdEndRenderPass(commandBuffer);

      // finish recording the command buffer
      if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }

    /*
     * This function creates the vulkan synchronization structures we will use.
     */
    void createSyncObjects() {
      imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
      renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
      inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

      VkSemaphoreCreateInfo semaphoreInfo{};
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

      VkFenceCreateInfo fenceInfo{};
      fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

          throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
      } 
    }

    /*
     * Graphics devices offer multiple kinds of memory to allocate from. This function finds a
     * memory type supported by the device which suits our needs (in this case for the vertex buffer)
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
      
      // query for available memory types
      VkPhysicalDeviceMemoryProperties memProperties;
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

      for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
      }

      throw std::runtime_error("failed to find suitable memory type!");
    }

    /*
     * Helper function for setting up a memory buffer
     */
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
      VkBufferCreateInfo bufferInfo{};
      bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferInfo.size = size;
      bufferInfo.usage = usage;
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      // create buffer
      const VkResult createBufferStatus = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
      assert(createBufferStatus == VK_SUCCESS);
      
      // we still need to allocate memory for the buffer so we start by getting the
      // memory requirements of the buffer
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

      // now fill in the memory allocation struct
      VkMemoryAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

      // allocate memory of the selected type
      const VkResult allocateMemoryStatus = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
      assert(allocateMemoryStatus == VK_SUCCESS);

      vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    /*
     * Function for allocating vertex buffer memory on device and transferring data to it
     */
    void createVertexBuffer() {
      VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

      // create staging buffer
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
      
      // map vertex data to host visible staging buffer
      void* data;
      vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
          memcpy(data, vertices.data(), (size_t) bufferSize);
      vkUnmapMemory(device, stagingBufferMemory);
      
      // create a buffer on the device for the vertex data
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
      // NOTE: it is not guaranteed that the write to memory will immediately be reflected in the
      // device's memory. The specification does however guarantee that the transfer of data to 
      // the GPU is to be complete as of the next call to VkQueueSubmit.
      // However, since we set the allocation info type to VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      // we actually don't have to worry about this as that setting guarantees that the mapped
      // memory always matches the contents of the allocated memory.

      copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    /*
     *  Helper function for copying device data buffers.
     */
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = transferCommandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, transferCommandPool, 1, &commandBuffer);
    }

    /*
     *  Function responsible for allocating the index buffer memory and transferring data to it
     */
    void createIndexBuffer() {
      VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

      void* data;
      vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
      memcpy(data, indices.data(), (size_t) bufferSize);
      vkUnmapMemory(device, stagingBufferMemory);

      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

      copyBuffer(stagingBuffer, indexBuffer, bufferSize);

      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createDescriptorSetLayout() {
      VkDescriptorSetLayoutBinding uboLayoutBinding{};
      uboLayoutBinding.binding = 0;
      uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      uboLayoutBinding.descriptorCount = 1;

      uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

      uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

      VkDescriptorSetLayoutCreateInfo layoutInfo{};
      layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layoutInfo.bindingCount = 1;
      layoutInfo.pBindings = &uboLayoutBinding;

      const VkResult descriptorStatus = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
      assert( descriptorStatus == VK_SUCCESS );
    }

    /*
     *  Create a uniform buffer for every swap chain image
     */
    void createUniformBuffers() {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);

      uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
      uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
      uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

          vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
      }
    }

    /*
     * This is our main drawing function
     */
    void drawFrame() {

      // wait for the CPU fence to be signaled when the previous frame finishes rendering
      vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

      uint32_t imageIndex;
      VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

      if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
          recreateSwapChain();
          return;
      } else if (result != VK_SUCCESS) {
          throw std::runtime_error("failed to acquire swap chain image!");
      }
      
      // only reset the fence if we are submitting work
      vkResetFences(device, 1, &inFlightFences[currentFrame]);

      updateUniformBuffer(currentFrame);
      
      vkResetCommandBuffer(commandBuffers[currentFrame], 0);
      recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

      VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
      
      VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
      VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores = waitSemaphores;
      submitInfo.pWaitDstStageMask = waitStages;
    
      VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores = signalSemaphores;
      
      // submit command buffer to graphics queue
      if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
      }
      VkPresentInfoKHR presentInfo{};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores = signalSemaphores;
      VkSwapchainKHR swapChains[] = {swapChain};
      
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = swapChains;
      presentInfo.pImageIndices = &imageIndex;
    
      presentInfo.pResults = nullptr; // Optional

      vkQueuePresentKHR(presentQueue, &presentInfo);

      currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage) {
      static auto startTime = std::chrono::high_resolution_clock::now();

      auto currentTime = std::chrono::high_resolution_clock::now();
      float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

      UniformBufferObject ubo{};
      ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0, 0, 1));

      ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

      ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);

      ubo.proj[1][1] *= -1;

      memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createDescriptorPool() {
      VkDescriptorPoolSize poolSize{};
      poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

      VkDescriptorPoolCreateInfo poolInfo{};
      poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      poolInfo.poolSizeCount = 1;
      poolInfo.pPoolSizes = &poolSize;

      poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

      const VkResult descriptorPoolStatus = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
      assert(descriptorPoolStatus == VK_SUCCESS);
    }

    void createDescriptorSets() {
      std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
      VkDescriptorSetAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptorPool;
      allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
      allocInfo.pSetLayouts = layouts.data();

      descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

      const VkResult descriptorSetStatus = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
      assert(descriptorSetStatus == VK_SUCCESS);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;

        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr; // Optional
        descriptorWrite.pTexelBufferView = nullptr; // Optional

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
      }

      
    }

    void cleanupSwapChain() {
     
      // we must destroy the semaphores as well to avoid a validation error
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      }

      for (auto framebuffer : swapChainFramebuffers) {
          vkDestroyFramebuffer(device, framebuffer, nullptr);
      }

      for (auto imageView : swapChainImageViews) {
          vkDestroyImageView(device, imageView, nullptr);
      }

      vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void recreateSwapChain() {
      int width = 0, height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      while (width == 0 || height == 0) {
          glfwGetFramebufferSize(window, &width, &height);
          glfwWaitEvents();
      }
      vkDeviceWaitIdle(device);

      cleanupSwapChain();

      createSwapChain();
      createImageViews();
      createFramebuffers();
      
      // we must re-create the semaphores as well to avoid a validation error
      VkSemaphoreCreateInfo semaphoreInfo{};
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ) {

          throw std::runtime_error("failed to re-create semaphore objects for a frame!");
        }
      }
    }
};

int main() {
  VulkanApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
