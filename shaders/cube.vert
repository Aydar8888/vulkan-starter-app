#version 450

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
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

const vec3 positions[8] = vec3[8](
    vec3(-1, -1, -1), vec3( 1, -1, -1),
    vec3( 1,  1, -1), vec3(-1,  1, -1),
    vec3(-1, -1,  1), vec3( 1, -1,  1),
    vec3( 1,  1,  1), vec3(-1,  1,  1)
);

const int indices[36] = int[36](
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5
);

void main() {
    int triangleStart = (gl_VertexIndex / 3) * 3;
    vec3 a = positions[indices[triangleStart]];
    vec3 b = positions[indices[triangleStart + 1]];
    vec3 c = positions[indices[triangleStart + 2]];
    vec3 localNormal = normalize(cross(b - a, c - a));

    vec3 localPosition = positions[indices[gl_VertexIndex]];
    vec4 world = model.transform * vec4(localPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(mat3(model.normalTransform) * localNormal);
    gl_Position = scene.viewProjection * world;
}