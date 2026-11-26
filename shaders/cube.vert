#version 450

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    ivec4 lightCounts;
} scene;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 transform;
    mat4 normalTransform;
    vec4 albedo;
    vec4 specularShininess;
} model;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 textureUV;
layout(location = 3) out vec4 lightPosition;

layout(location = 0) in vec3 localPosition;
layout(location = 1) in vec3 localNormal;
layout(location = 2) in vec2 localUV;

void main() {
    vec4 world = model.transform * vec4(localPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(mat3(model.normalTransform) * localNormal);
    textureUV = localUV;
    lightPosition = scene.lightViewProjection * world;
    gl_Position = scene.viewProjection * world;
}
