#version 450

layout(set = 0, binding = 0, std140) uniform Scene {
    layout(offset = 64) mat4 lightViewProjection;
} scene;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 transform;
} model;

layout(location = 0) in vec3 localPosition;

void main() {
    gl_Position = scene.lightViewProjection * model.transform * vec4(localPosition, 1.0);
}
