#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <imgui.h>

namespace application {
namespace {

constexpr float pi = 3.14159265358979323846f;

struct Mat4 {
    float data[16]{};
};

struct alignas(16) Vec4 {
    float x, y, z, w;
};

struct SceneData {
    Mat4 view_projection;
    Vec4 camera_position;
    Vec4 ambient;
    Vec4 directional_direction;
    Vec4 directional_color_intensity;
    int32_t light_counts[4];
};

struct ModelData {
    Mat4 transform;
    Mat4 normal_transform;
    Vec4 albedo;
    Vec4 specular_shininess;
};

struct PointLightData {
    Vec4 position;
    Vec4 color_intensity;
};

struct SpotLightData {
    Vec4 position;
    Vec4 direction_inner_cos;
    Vec4 color_intensity;
    Vec4 outer_cos;
};

static_assert(sizeof(SceneData) == 144);
static_assert(sizeof(ModelData) == 160);
static_assert(sizeof(PointLightData) == 32);
static_assert(sizeof(SpotLightData) == 64);
static_assert(offsetof(SceneData, camera_position) == 64);
static_assert(offsetof(SceneData, ambient) == 80);
static_assert(offsetof(SceneData, directional_direction) == 96);
static_assert(offsetof(SceneData, directional_color_intensity) == 112);
static_assert(offsetof(SceneData, light_counts) == 128);
static_assert(offsetof(ModelData, normal_transform) == 64);
static_assert(offsetof(ModelData, albedo) == 128);
static_assert(offsetof(ModelData, specular_shininess) == 144);

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct PointLight {
    float position[3] = {2.0f, 2.0f, 2.0f};
    float color[3] = {1.0f, 0.65f, 0.35f};
    float intensity = 6.0f;
};

struct SpotLight {
    float position[3] = {-2.0f, 3.0f, 2.0f};
    float direction[3] = {0.4f, -1.0f, -0.4f};
    float color[3] = {0.45f, 0.65f, 1.0f};
    float intensity = 10.0f;
    float inner_angle = 18.0f;
    float outer_angle = 28.0f;
};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;
Buffer scene_buffer;
Buffer model_buffer;
Buffer point_buffer;
Buffer spot_buffer;

float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {25.0f, 35.0f, 0.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};
bool use_perspective = true;
float camera_position[3] = {0.0f, 0.0f, 6.0f};
float camera_rotation[2] = {0.0f, 0.0f};
float camera_fov = 45.0f;
float ambient_color[3] = {0.12f, 0.12f, 0.15f};
float directional_direction[3] = {-0.5f, -1.0f, -0.6f};
float directional_color[3] = {1.0f, 1.0f, 1.0f};
float directional_intensity = 1.3f;
float albedo[3] = {0.65f, 0.72f, 0.95f};
float specular_color[3] = {1.0f, 1.0f, 1.0f};
float shininess = 48.0f;
std::vector<PointLight> point_lights = {PointLight{}};
std::vector<SpotLight> spot_lights = {SpotLight{}};
bool animation_playing = false;
float animation_speed = 1.0f;
float orbit_radius = 0.8f;
float vertical_amplitude = 0.4f;
float animation_time = 0.0f;
double previous_time = -1.0;

Mat4 identity() {
    Mat4 m{};
    m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0f;
    return m;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result.data[column * 4 + row] +=
                    a.data[k * 4 + row] * b.data[column * 4 + k];
            }
        }
    }
    return result;
}

Mat4 translate(float x, float y, float z) {
    Mat4 m = identity();
    m.data[12] = x;
    m.data[13] = y;
    m.data[14] = z;
    return m;
}

Mat4 rotateX(float angle) {
    Mat4 m = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    m.data[5] = c;
    m.data[6] = s;
    m.data[9] = -s;
    m.data[10] = c;
    return m;
}

Mat4 rotateY(float angle) {
    Mat4 m = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    m.data[0] = c;
    m.data[2] = -s;
    m.data[8] = s;
    m.data[10] = c;
    return m;
}

Mat4 rotateZ(float angle) {
    Mat4 m = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    m.data[0] = c;
    m.data[1] = s;
    m.data[4] = -s;
    m.data[5] = c;
    return m;
}

Mat4 nonUniformScale(float x, float y, float z) {
    Mat4 m = identity();
    m.data[0] = x;
    m.data[5] = y;
    m.data[10] = z;
    return m;
}

Mat4 perspective(float aspect, float fov_degrees) {
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const float f = 1.0f / std::tan(fov_degrees * pi / 360.0f);

    Mat4 m{};
    m.data[0] = f / aspect;
    m.data[5] = -f;
    m.data[10] = far_plane / (near_plane - far_plane);
    m.data[11] = -1.0f;
    m.data[14] = far_plane * near_plane / (near_plane - far_plane);
    return m;
}

Mat4 orthographic(float aspect) {
    const float half_height = 2.5f;
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;

    Mat4 m = identity();
    m.data[0] = 1.0f / (half_height * aspect);
    m.data[5] = -1.0f / half_height;
    m.data[10] = 1.0f / (near_plane - far_plane);
    m.data[14] = near_plane / (near_plane - far_plane);
    return m;
}

Vec4 direction4(const float value[3], float w) {
    const float length = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
    );
    if (length < 0.0001f) {
        return {0.0f, -1.0f, 0.0f, w};
    }
    return {value[0] / length, value[1] / length, value[2] / length, w};
}

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    auto& context = graphics::internal::context;
    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo result{};
    if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
                        &buffer.handle, &buffer.allocation, &result) != VK_SUCCESS) {
        std::cerr << "Cannot allocate lighting buffer\n";
        return false;
    }
    buffer.mapped = result.pMappedData;
    buffer.size = size;
    return true;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(graphics::internal::context.allocator,
                         buffer.handle, buffer.allocation);
    }
    buffer = {};
}

void writeBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
    std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
    vmaFlushAllocation(graphics::internal::context.allocator,
                       buffer.allocation, 0, size);
}

void updateBufferDescriptor(VkDescriptorSet set, uint32_t binding,
                            VkDescriptorType type, const Buffer& buffer) {
    const VkDescriptorBufferInfo info = {
        .buffer = buffer.handle,
        .offset = 0,
        .range = buffer.size,
    };
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void destroyResources() {
    auto& context = graphics::internal::context;
    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    destroyBuffer(scene_buffer);
    destroyBuffer(model_buffer);
    destroyBuffer(point_buffer);
    destroyBuffer(spot_buffer);
    if (scene_set_layout) vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
    if (model_set_layout) vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
    pipeline = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    descriptor_pool = VK_NULL_HANDLE;
    scene_set_layout = VK_NULL_HANDLE;
    model_set_layout = VK_NULL_HANDLE;
    scene_set = VK_NULL_HANDLE;
    model_set = VK_NULL_HANDLE;
}

bool initializeDescriptors() {
    auto& context = graphics::internal::context;
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo scene_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = scene_bindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &scene_info, nullptr,
                                    &scene_set_layout) != VK_SUCCESS) {
        std::cerr << "Cannot create scene descriptor layout\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding model_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo model_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &model_binding,
    };
    if (vkCreateDescriptorSetLayout(context.device, &model_info, nullptr,
                                    &model_set_layout) != VK_SUCCESS) {
        std::cerr << "Cannot create model descriptor layout\n";
        return false;
    }

    const VkDescriptorPoolSize pool_sizes[] = {
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
                               &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Cannot create lighting descriptor pool\n";
        return false;
    }

    if (!createBuffer(scene_buffer, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
        !createBuffer(model_buffer, sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
        !createBuffer(point_buffer, sizeof(PointLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !createBuffer(spot_buffer, sizeof(SpotLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    const VkDescriptorSetLayout layouts[] = {scene_set_layout, model_set_layout};
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet sets[2]{};
    if (vkAllocateDescriptorSets(context.device, &allocate_info, sets) != VK_SUCCESS) {
        std::cerr << "Cannot allocate lighting descriptor sets\n";
        return false;
    }
    scene_set = sets[0];
    model_set = sets[1];
    updateBufferDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
    updateBufferDescriptor(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer);
    updateBufferDescriptor(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer);
    updateBufferDescriptor(model_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, model_buffer);
    return true;
}

bool ensureStorage(Buffer& buffer, size_t count, VkDeviceSize element_size,
                   uint32_t binding) {
    const VkDeviceSize required = std::max<VkDeviceSize>(1, count) * element_size;
    if (required <= buffer.size) return true;

    Buffer expanded;
    if (!createBuffer(expanded, std::max(required, buffer.size * 2),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }
    destroyBuffer(buffer);
    buffer = expanded;
    updateBufferDescriptor(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
    return true;
}

bool loadShader(const char* path, VkShaderModule& module) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open shader: " << path << '\n';
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        std::cerr << "Invalid shader: " << path << '\n';
        return false;
    }

    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), size)) {
        std::cerr << "Cannot read shader: " << path << '\n';
        return false;
    }

    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<size_t>(size),
        .pCode = code.data(),
    };

    return vkCreateShaderModule(
        graphics::internal::context.device, &info, nullptr, &module
    ) == VK_SUCCESS;
}

} // namespace

bool initialize() {
    auto& context = graphics::internal::context;

    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;

    if (!loadShader("shaders/cube.vert.spv", vert) ||
        !loadShader("shaders/cube.frag.spv", frag)) {
        if (vert) vkDestroyShaderModule(context.device, vert, nullptr);
        if (frag) vkDestroyShaderModule(context.device, frag, nullptr);
        return false;
    }

    if (!initializeDescriptors()) {
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        destroyResources();
        return false;
    }

    const VkDescriptorSetLayout layouts[] = {scene_set_layout, model_set_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = layouts,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
                               &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Cannot create pipeline layout\n";
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        destroyResources();
        return false;
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag,
            .pName = "main",
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState color_attachment = {
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = context.render_pass,
    };

    const VkResult result = vkCreateGraphicsPipelines(
        context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline
    );

    vkDestroyShaderModule(context.device, vert, nullptr);
    vkDestroyShaderModule(context.device, frag, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create graphics pipeline: " << result << '\n';
        destroyResources();
        return false;
    }

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    destroyResources();
}

void update(double time) {
	if (previous_time >= 0.0 && animation_playing) {
		const double delta = std::clamp(time - previous_time, 0.0, 0.1);
		animation_time += static_cast<float>(delta) * animation_speed;
	}
	previous_time = time;
    ImGui::Begin("Camera and lighting");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Camera position", camera_position, -12.0f, 12.0f);
        ImGui::SliderFloat("Yaw", &camera_rotation[0], -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &camera_rotation[1], -89.0f, 89.0f);
        ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 100.0f);
        ImGui::Checkbox("Perspective projection", &use_perspective);
    }
    if (ImGui::CollapsingHeader("Cube", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", position, -2.0f, 2.0f);
        ImGui::SliderFloat3("Rotation (degrees)", rotation, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale XYZ", scale, 0.1f, 2.0f);
        ImGui::Checkbox("Play animation", &animation_playing);
        ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 4.0f);
        ImGui::SliderFloat("Orbit radius", &orbit_radius, 0.0f, 1.5f);
        ImGui::SliderFloat("Vertical amplitude", &vertical_amplitude, 0.0f, 1.0f);
    }
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", albedo);
        ImGui::ColorEdit3("Specular color", specular_color);
        ImGui::SliderFloat("Shininess", &shininess, 2.0f, 128.0f);
    }
    if (ImGui::CollapsingHeader("Ambient and directional light",
                               ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient light", ambient_color);
        ImGui::SliderFloat3("Light direction", directional_direction, -1.0f, 1.0f);
        ImGui::ColorEdit3("Directional color", directional_color);
        ImGui::SliderFloat("Directional intensity", &directional_intensity, 0.0f, 4.0f);
    }
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("point_lights");
        if (ImGui::Button("Add point light")) point_lights.emplace_back();
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::SeparatorText("Point light");
            ImGui::SliderFloat3("Position", point_lights[i].position, -8.0f, 8.0f);
            ImGui::ColorEdit3("Color", point_lights[i].color);
            ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 30.0f);
            const bool remove = ImGui::Button("Remove");
            ImGui::PopID();
            if (remove) {
                point_lights.erase(point_lights.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        ImGui::PopID();
    }
    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("spot_lights");
        if (ImGui::Button("Add spot light")) spot_lights.emplace_back();
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::SeparatorText("Spot light");
            ImGui::SliderFloat3("Position", spot_lights[i].position, -8.0f, 8.0f);
            ImGui::SliderFloat3("Direction", spot_lights[i].direction, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", spot_lights[i].color);
            ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 30.0f);
            ImGui::SliderFloat("Inner angle", &spot_lights[i].inner_angle, 1.0f, 80.0f);
            ImGui::SliderFloat("Outer angle", &spot_lights[i].outer_angle,
                               spot_lights[i].inner_angle, 89.0f);
            spot_lights[i].outer_angle =
                std::max(spot_lights[i].outer_angle, spot_lights[i].inner_angle);
            const bool remove = ImGui::Button("Remove");
            ImGui::PopID();
            if (remove) {
                spot_lights.erase(spot_lights.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        ImGui::PopID();
    }
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;

    const float radians = pi / 180.0f;
    const float t = animation_time;

	Mat4 model = translate(
		position[0] + orbit_radius * (std::cos(t) - 1.0f),
		position[1] + vertical_amplitude * std::sin(2.0f * t),
		position[2] + orbit_radius * std::sin(t)
	);
	model = multiply(model, rotateZ(rotation[2] * radians + 0.3f * t));
    model = multiply(model, rotateY(rotation[1] * radians + t));
    model = multiply(model, rotateX(rotation[0] * radians + 0.5f * t));
    model = multiply(model, nonUniformScale(scale[0], scale[1], scale[2]));

    Mat4 normal_transform = rotateZ(rotation[2] * radians + 0.3f * t);
    normal_transform = multiply(normal_transform, rotateY(rotation[1] * radians + t));
    normal_transform = multiply(normal_transform, rotateX(rotation[0] * radians + 0.5f * t));
    normal_transform = multiply(normal_transform, nonUniformScale(
        1.0f / scale[0], 1.0f / scale[1], 1.0f / scale[2]
    ));

    const Mat4 view = multiply(
        rotateX(-camera_rotation[1] * radians),
        multiply(rotateY(-camera_rotation[0] * radians),
                 translate(-camera_position[0], -camera_position[1], -camera_position[2]))
    );
    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

    ensureStorage(point_buffer, point_lights.size(), sizeof(PointLightData), 1);
    ensureStorage(spot_buffer, spot_lights.size(), sizeof(SpotLightData), 2);
    const size_t point_count = std::min(
        point_lights.size(), static_cast<size_t>(point_buffer.size / sizeof(PointLightData))
    );
    const size_t spot_count = std::min(
        spot_lights.size(), static_cast<size_t>(spot_buffer.size / sizeof(SpotLightData))
    );

    std::vector<PointLightData> gpu_points;
    gpu_points.reserve(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        const auto& light = point_lights[i];
        gpu_points.push_back({
            {light.position[0], light.position[1], light.position[2], 1.0f},
            {light.color[0], light.color[1], light.color[2], light.intensity},
        });
    }
    if (!gpu_points.empty()) {
        writeBuffer(point_buffer, gpu_points.data(), gpu_points.size() * sizeof(PointLightData));
    }

    std::vector<SpotLightData> gpu_spots;
    gpu_spots.reserve(spot_count);
    for (size_t i = 0; i < spot_count; ++i) {
        const auto& light = spot_lights[i];
        gpu_spots.push_back({
            {light.position[0], light.position[1], light.position[2], 1.0f},
            direction4(light.direction, std::cos(light.inner_angle * radians)),
            {light.color[0], light.color[1], light.color[2], light.intensity},
            {std::cos(light.outer_angle * radians), 0.0f, 0.0f, 0.0f},
        });
    }
    if (!gpu_spots.empty()) {
        writeBuffer(spot_buffer, gpu_spots.data(), gpu_spots.size() * sizeof(SpotLightData));
    }

    const SceneData scene_data = {
        .view_projection = multiply(
            use_perspective ? perspective(aspect, camera_fov) : orthographic(aspect), view
        ),
        .camera_position = {camera_position[0], camera_position[1], camera_position[2], 1.0f},
        .ambient = {ambient_color[0], ambient_color[1], ambient_color[2], 1.0f},
        .directional_direction = direction4(directional_direction, 0.0f),
        .directional_color_intensity = {
            directional_color[0], directional_color[1], directional_color[2],
            directional_intensity
        },
        .light_counts = {static_cast<int32_t>(point_count),
                         static_cast<int32_t>(spot_count), 0, 0},
    };
    const ModelData model_data = {
        .transform = model,
        .normal_transform = normal_transform,
        .albedo = {albedo[0], albedo[1], albedo[2], 1.0f},
        .specular_shininess = {
            specular_color[0], specular_color[1], specular_color[2], shininess
        },
    };
    writeBuffer(scene_buffer, &scene_data, sizeof(scene_data));
    writeBuffer(model_buffer, &model_data, sizeof(model_data));

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    const VkClearValue clear_values[] = {
        {.color = {{0.02f, 0.08f, 0.20f, 1.0f}}},
        {.depthStencil = {1.0f, 0}},
    };

    const VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {.extent = context.swapchain_extent},
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };

    vkResetCommandBuffer(fd.command_buffer, 0);
    vkBeginCommandBuffer(fd.command_buffer, &begin_info);
    vkCmdBeginRenderPass(
        fd.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE
    );

    vkCmdBindPipeline(
        fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline
    );

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .extent = context.swapchain_extent,
    };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    const VkDescriptorSet sets[] = {scene_set, model_set};
    vkCmdBindDescriptorSets(
        fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
        0, 2, sets, 0, nullptr
    );
    vkCmdDraw(fd.command_buffer, 36, 1, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
