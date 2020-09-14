#version 450

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec4 a_color;

layout(set = 0, binding = 0) uniform transform {
    mat4 u_transform;
};
layout(set = 0, binding = 1) uniform viewpoint {
    vec3 u_viewpoint;
};

layout(location = 0) out vec4 v_color;

void main() {
    v_color = a_color * vec4(vec3(max(0.0, dot(vec3(a_normal), normalize(u_viewpoint - vec3(a_position))))), 1.0);
    gl_Position = u_transform * a_position;
}
