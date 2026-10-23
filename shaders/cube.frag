#version 450

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    ivec4 lightCounts;
} scene;

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

struct SpotLight {
    vec4 position;
    vec4 directionInnerCos;
    vec4 colorIntensity;
    vec4 outerCos;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
    PointLight items[];
} pointLights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
    SpotLight items[];
} spotLights;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 transform;
    mat4 normalTransform;
    vec4 albedo;
    vec4 specularShininess;
} model;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 lightDirection, vec3 lightColor, vec3 normal, vec3 viewDirection) {
    float diffuse = max(dot(normal, lightDirection), 0.0);
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float specular = diffuse > 0.0
        ? pow(max(dot(normal, halfVector), 0.0), model.specularShininess.w)
        : 0.0;
    return lightColor * (
        model.albedo.rgb * diffuse +
        model.specularShininess.rgb * specular
    );
}

void main() {
    vec3 normal = normalize(worldNormal);
    vec3 viewDirection = normalize(scene.cameraPosition.xyz - worldPosition);
    vec3 color = scene.ambient.rgb * model.albedo.rgb;

    vec3 directionalLight = normalize(-scene.directionalDirection.xyz);
    color += blinnPhong(
        directionalLight,
        scene.directionalColorIntensity.rgb * scene.directionalColorIntensity.w,
        normal, viewDirection
    );

    for (int i = 0; i < scene.lightCounts.x; ++i) {
        vec3 offset = pointLights.items[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(offset, offset), 0.01);
        vec3 direction = normalize(offset);
        vec3 lightColor = pointLights.items[i].colorIntensity.rgb *
            (pointLights.items[i].colorIntensity.w / distanceSquared);
        color += blinnPhong(direction, lightColor, normal, viewDirection);
    }

    for (int i = 0; i < scene.lightCounts.y; ++i) {
        vec3 offset = spotLights.items[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(offset, offset), 0.01);
        vec3 direction = normalize(offset);
        float angleCos = dot(
            normalize(spotLights.items[i].directionInnerCos.xyz), -direction
        );
        float outer = spotLights.items[i].outerCos.x;
        float inner = spotLights.items[i].directionInnerCos.w;
        float cone = smoothstep(outer, max(inner, outer + 0.0001), angleCos);
        vec3 lightColor = spotLights.items[i].colorIntensity.rgb *
            (spotLights.items[i].colorIntensity.w * cone / distanceSquared);
        color += blinnPhong(direction, lightColor, normal, viewDirection);
    }

    outColor = vec4(color, 1.0);
}
