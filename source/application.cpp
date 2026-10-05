#include "application.hpp"

#include <cmath>
#include <cstdint>
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

struct PushData {
    Mat4 mvp;
    float tint[4];
};

static_assert(sizeof(PushData) == 80);

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {25.0f, 35.0f, 0.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};
float tint[3] = {1.0f, 1.0f, 1.0f};
bool use_perspective = true;
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

Mat4 perspective(float aspect) {
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const float f = 1.0f / std::tan(45.0f * pi / 360.0f);

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

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushData),
    };

    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
                               &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Cannot create pipeline layout\n";
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
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
        vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        pipeline_layout = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
}

void update(double time) {
	if (previous_time >= 0.0 && animation_playing) {
		const double delta = std::clamp(time - previous_time, 0.0, 0.1);
		animation_time += static_cast<float>(delta) * animation_speed;
	}
	previous_time = time;
    ImGui::Begin("Cube");
	ImGui::Checkbox("Perspective projection", &use_perspective);
    ImGui::SliderFloat3("Position", position, -2.0f, 2.0f);
    ImGui::SliderFloat3("Rotation (degrees)", rotation, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale XYZ", scale, 0.1f, 2.0f);
	ImGui::ColorEdit3("Cube color", tint);
	ImGui::Separator();
	ImGui::Checkbox("Play animation", &animation_playing);
	ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 4.0f);
	ImGui::SliderFloat("Orbit radius", &orbit_radius, 0.0f, 1.5f);
	ImGui::SliderFloat("Vertical amplitude", &vertical_amplitude, 0.0f, 1.0f);
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

    const Mat4 view = translate(0.0f, 0.0f, -6.0f);
    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

    const PushData push_data = {
        .mvp = multiply(
			use_perspective ? perspective(aspect) : orthographic(aspect),
			multiply(view, model)
		),
        .tint = {tint[0], tint[1], tint[2], 1.0f},
    };

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
    vkCmdPushConstants(
        fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
        0, sizeof(push_data), &push_data
    );
    vkCmdDraw(fd.command_buffer, 36, 1, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
