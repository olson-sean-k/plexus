#version 150 core

in vec3 a_position;
in vec3 a_normal;

out vec4 v_color;

uniform transform {
    vec3 u_viewpoint;
    mat4 u_camera;
    mat4 u_model;
};

void main() {
    v_color = vec4(vec3(max(0.1, dot(a_normal, normalize(u_viewpoint - a_position)))), 1.0);
    gl_Position = u_camera * u_model * vec4(a_position, 1.0);
}
