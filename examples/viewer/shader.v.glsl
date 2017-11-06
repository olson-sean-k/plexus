#version 150 core

in vec3 a_position;
in vec4 a_color;

out vec4 v_color;

uniform transform {
    mat4 u_camera;
    mat4 u_model;
};

void main() {
    v_color = a_color;
    gl_Position = u_camera * u_model * vec4(a_position, 1.0);
}
