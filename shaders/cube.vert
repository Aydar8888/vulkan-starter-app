#version 450

layout(location = 0) out vec3 vertexColor;

layout(push_constant) uniform DrawData {
    mat4 mvp;
    vec4 tint;
} drawData;

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
    vec3 position = positions[indices[gl_VertexIndex]];
    gl_Position = drawData.mvp * vec4(position, 1.0);
    vertexColor = (position * 0.5 + 0.5) * drawData.tint.rgb;
}