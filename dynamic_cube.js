"use strict";
class Particle{
	constructor(position, force, mass, velocity){
		this.position = position;
		this.force = force;
		this.mass = mass;
		this.velocity = velocity;
	}
	
	update_position(delta_t){
		delta_t = Math.min(delta_t, 50);
		delta_t *= 0.004;
		const acceleration = scalar_multiply(this.force, 1/this.mass);
		this.velocity = scalar_multiply(acceleration, delta_t);
		const new_position = integrate_midpoint(this.position, this.velocity, acceleration, delta_t);
		this.position = new_position;
	}
	
	remove_all_force(){
		this.force = [0, 0, 0];
	}
}

class Spring{
	constructor(first_particle, second_particle, spring_constant, dampening_constant, rest_length){
		this.first_particle = first_particle;
		this.second_particle = second_particle;
		this.spring_constant = spring_constant;
		this.dampening_constant = dampening_constant;
		this.rest_length = rest_length;
	}
	
	calculate_forces_on_particles(){
		const delta_vec = minus(this.first_particle.position, this.second_particle.position);
		const length = magnitude(delta_vec);
		const delta_velocity = minus(this.first_particle.velocity, this.second_particle.velocity);
		
		const force = scalar_multiply(scalar_multiply(delta_vec, 1/length), (this.spring_constant * (length - this.rest_length) + this.dampening_constant * (dot(delta_velocity, delta_vec) / length)));
		return {
			"force_first_particle": scalar_multiply(force, -1),
			"force_second_particle": force
		}
	}
}

class ParticleSystem{
	constructor(particles, springs){
		this.particles = particles;
		this.springs = springs;
	}
	
	update(delta_t){
		this.springs.forEach((spring)=>{
			const forces = spring.calculate_forces_on_particles();
			spring.first_particle.force = plus(spring.first_particle.force, forces.force_first_particle);
			spring.second_particle.force = plus(spring.second_particle.force, forces.force_second_particle);
		});
		this.particles.forEach((particle)=>{
			particle.update_position(delta_t);
		});
	}
	
	get_particles(){
		return this.particles;
	}
	
	get_springs(){
		return this.springs;
	}
}

class AnchoredParticleDecorator{
	constructor(particle_system, anchored_particles){
		this.particle_system = particle_system;
		this.anchored_particles = anchored_particles;
		this.initial_positions = [];
		this.anchored_particles.forEach((particle)=>{
			this.initial_positions.push(particle.position);
		});
	}
	
	update(delta_t){
		this.particle_system.update(delta_t);
		this.anchored_particles.forEach((particle, i)=>{
			particle.remove_all_force();
			particle.position = this.initial_positions[i];
		});
	}
	
	get_particles(){
		return this.particle_system.get_particles();
	}
	
	get_springs(){
		return this.particle_system.get_springs();
	}
}

class ParticleSystemFace{
	constructor(triangle_strip_ordered_particles, texture_id){
		this.triangle_strip_ordered_particles = triangle_strip_ordered_particles;
		this.texture_id = texture_id;
	}
}

class RenderFacesDecorator{
	constructor(particle_system, particle_system_faces){
		this.particle_system = particle_system;
		this.particle_system_faces = particle_system_faces;
	}
	
	update(delta_t){
		this.particle_system.update(delta_t);
	}
	
	get_particles(){
		return this.particle_system.get_particles();
	}
	
	get_springs(){
		return this.particle_system.get_springs();
	}
	
	get_particle_system_faces(){
		return this.particle_system_faces;
	}
}

class DampeningForcesDecorator{
	constructor(particle_system){
		this.particle_system = particle_system;
	}
	
	update(delta_t){
		const dampening_factor = 0.99;
	
		this.particle_system.update(delta_t);
		this.get_particles().forEach((particle)=>{
			particle.force = scalar_multiply(particle.force, dampening_factor);
		});
	}
	
	get_particles(){
		return this.particle_system.get_particles();
	}
	
	get_springs(){
		return this.particle_system.get_springs();
	}
	
	get_particle_system_faces(){
		return this.particle_system.get_particle_system_faces();
	}
}

function integrate_midpoint(current_x, current_v, current_a, delta_t){
	return plus(plus(current_x, scalar_multiply(current_v, delta_t)), scalar_multiply(current_a, delta_t * delta_t));
}

const vertex_shader_source = 
	`// an attribute will receive data from a buffer
	attribute vec4 position;
	
	attribute vec2 aTexturePosition;
	varying vec2 vTexturePosition;
	
	uniform mat4 mvp_matrix;

	// all shaders have a main function
	void main() {

	  // gl_Position is a special variable a vertex shader 
	  // is responsible for setting
	  gl_Position = mvp_matrix * position;
	  
	  vTexturePosition = aTexturePosition;
	}`;

const fragment_shader_source = 
	`// fragment shaders don't have a default precision so we need
	// to pick one. mediump, short for medium precision, is a good default.
	precision mediump float;

	varying vec2 vTexturePosition;
	
	uniform bool useTexture;
	uniform sampler2D uTexture;

	void main() {
	  if(useTexture){
		gl_FragColor = texture2D(uTexture, vTexturePosition);
	  }
	  else{
		// gl_FragColor is a special variable a fragment shader
		// is responsible for setting
		gl_FragColor = vec4(0, 0, 0.5412, 1); // return dark-azure 
	  }
	}`;
	
const canvas = document.getElementById("c");
const canvas_picking = {
	// set by the mouse event handler, reset/consumed by the render loop
	"pick_request": null,
	"pick_drag_request": null,
	"pick_reset_request": false
};
bind_mouse_events_to_canvas(canvas, canvas_picking);
bind_touch_events_to_canvas(canvas, canvas_picking);

function bind_mouse_events_to_canvas(canvas, canvas_picking){
	canvas.addEventListener("mousedown", (event)=>{
		const bounds = canvas.getBoundingClientRect();
		canvas_picking.pick_request = {"x": event.x - bounds.left, "y": event.y - bounds.top};
	});
	
	canvas.addEventListener("mouseup", (event)=>{
		canvas_picking.pick_reset_request = true;
	});
	
	canvas.addEventListener("mousemove", (event)=>{
		const bounds = canvas.getBoundingClientRect();
		canvas_picking.pick_drag_request = {"x": event.x - bounds.left, "y": event.y - bounds.top};
	});
}

function bind_touch_events_to_canvas(canvas, canvas_picking){
	canvas.addEventListener("touchstart", (event)=>{
		if(event.changedTouches.length==0)
			return;
		
		const bounds = canvas.getBoundingClientRect();
		const first_touch = event.changedTouches[0];
		canvas_picking.pick_request = {"x": first_touch.clientX - bounds.left, "y": first_touch.clientY - bounds.top};
		
		event.preventDefault()
	});
	
	canvas.addEventListener("touchend", (event)=>{
		canvas_picking.pick_reset_request = true;
		event.preventDefault()
	});
	
	canvas.addEventListener("touchmove", (event)=>{
		if(event.changedTouches.length==0)
			return;
		
		const bounds = canvas.getBoundingClientRect();
		const first_touch = event.changedTouches[0];
		canvas_picking.pick_drag_request = {"x": first_touch.clientX - bounds.left, "y": first_touch.clientY - bounds.top};
		event.preventDefault()
	});
}

const gl = canvas.getContext("webgl", {antialias: true});
if (!gl) {
	alert("No OpenGL!");
}
else{
	start_rendering_to_gl_context(canvas, canvas_picking, gl);
}

function start_rendering_to_gl_context(canvas, canvas_picking, gl){
	gl.clearColor(1, 1, 1, 1);
	gl.enable(gl.DEPTH_TEST);
	
	const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertex_shader_source);
	const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragment_shader_source);
	const program = createProgram(gl, vertexShader, fragmentShader);		

	gl.useProgram(program);	
	const active_shader = bind_shader_program_handles(gl, program);
	
	const elements_to_render = {
		"particle_system": create_cube_with_particle_corners(gl),
		"grid_lines": create_bottom_grid_lines(0, 1000, 1/30)
	};
	
	const render_buffers = {}; //to be filled in during rendering
	const frame_query = {
		// When null, no extra routine would be done to enable picking, 
		// when not null, this field is a function that takes two arguments: 
		// 1. an array of triangles in normalized device coordinates, provided by the lower level rendering calls
		// 2. the object corresponding to the aforementioned triangles
		"picking_query": null,
		"picking_query_result": []
	};
	
	const urlSearchParams = new URLSearchParams(window.location.search);
	
	const fps_counter_dom_element = document.getElementById("fps_counter");
		
	const visibility = {
		"fps_counter": urlSearchParams.get("fpscounter") != null && urlSearchParams.get("fpscounter") == "1"
	}
	
	update_fps_counter_visibility(visibility, fps_counter_dom_element);
	
	window.requestAnimationFrame((timestamp)=>draw(timestamp, timestamp, visibility, fps_counter_dom_element, active_shader, elements_to_render, render_buffers, frame_query, canvas, canvas_picking));
}

function update_fps_counter_visibility(visibility, fps_counter_dom_element){
	if(fps_counter_dom_element!=null){
		fps_counter_dom_element.style.display = visibility.fps_counter ? "inline" : "none";
	}
}

// Shader should already be in use
function bind_shader_program_handles(gl, program){
	const positionAttributeLocation = gl.getAttribLocation(program, "position");
	gl.enableVertexAttribArray(positionAttributeLocation);
	
	const mvp_matrix_position = gl.getUniformLocation(program, "mvp_matrix");
	
	const useTexture_handle = gl.getUniformLocation(program, "useTexture");
	gl.uniform1i(useTexture_handle, 0); // Don't use a texture by default
	
	const texture_handle = gl.getUniformLocation(program, "uTexture");
	const texture_position_attribute_handle = gl.getAttribLocation(program, "aTexturePosition");
	// enableVertexAttribArray is not yet called on texture_position_attribute_handle here (not like with positionAttributeLocation)
	// when not rendering textures this must be disabled, or some rendering might fail
	// texture_position_attribute_handle is to be enabled/disabled explicitly when starting/stopping texture rendering
	
	gl.activeTexture(gl.TEXTURE0);
	gl.uniform1i(texture_handle, 0);
	
	return {
		"positionAttributeLocation": positionAttributeLocation,
		"mvp_matrix_position": mvp_matrix_position,
		"useTexture_handle": useTexture_handle,
		"texture_position_attribute_handle": texture_position_attribute_handle
	};
}

function draw(timestamp, previous_timestamp, visibility, fps_counter_dom_element, active_shader, elements_to_render, render_buffers, frame_query, canvas, canvas_picking){
	gl.clear(gl.COLOR_BUFFER_BIT);
	
	const milliseconds_per_frame = timestamp - previous_timestamp;
	update_fps_counter(fps_counter_dom_element, milliseconds_per_frame);
	
	const delta_t = Math.min(milliseconds_per_frame, 1000);
	
	const vp_matrix = adjust_viewport(canvas, gl);
	consume_canvas_picking_request(canvas_picking, frame_query, canvas, vp_matrix);

	render_grid(gl, elements_to_render.grid_lines, active_shader.positionAttributeLocation, vp_matrix, active_shader.mvp_matrix_position, render_buffers);
	
	elements_to_render.particle_system.update(delta_t);
	render_particle_system(gl, elements_to_render.particle_system, vp_matrix, active_shader, render_buffers, frame_query);
	
	frame_query.picking_query = null;
	
	window.requestAnimationFrame((next_timestamp)=>draw(next_timestamp, timestamp, visibility, fps_counter_dom_element, active_shader, elements_to_render, render_buffers, frame_query, canvas, canvas_picking));			
}

function update_fps_counter(fps_counter_dom_element, milliseconds_per_frame){
	const frames_per_second = (1.0 / (milliseconds_per_frame/1000));
	fps_counter_dom_element.innerHTML = "FPS: " + frames_per_second.toFixed(2);
}

function adjust_viewport(canvas, gl){
	const NEAR = 0.1;
	const FAR = 1100;
	
	const rect = canvas.getBoundingClientRect();
	canvas.width = rect.width;
	gl.viewport(0,0,canvas.width,canvas.height);
	
	const p_matrix = create_perspective_matrix(90.0, canvas.width/canvas.height, NEAR, FAR);
	const v_matrix = create_lookat_matrix(
		[0, 90, 1], 
		[Math.sin(0), 90, 1 - Math.cos(0)], 
		[0, 1, 0]);
	
	const vp_matrix = multiply(v_matrix, p_matrix);
	return vp_matrix;
}

function consume_canvas_picking_request(canvas_picking, frame_query, canvas, vp_matrix){
	if(canvas_picking.pick_reset_request){ // canvas requests clearing all previously picked objects
		clear_picking_request(canvas_picking, frame_query);
	}else if(frame_query.picking_query_result.length>0 
			&& canvas_picking.pick_drag_request!=null ){ // There are previously picked objects, and the canvas requests dragging them
		const picked_scene_element = frame_query.picking_query_result[0];
		if("particle" in picked_scene_element && "world_to_normalized" in picked_scene_element){
			const normalized_to_viewport = invert_mat4(picked_scene_element.world_to_normalized);
			const normalized_mouse_position = normalize_viewport_coordinates(canvas_picking.pick_drag_request.x, canvas_picking.pick_drag_request.y, canvas);
			normalized_mouse_position[2] = picked_scene_element.depth;
			let mouse_in_world = multiply_4x4_4x1(normalized_to_viewport, [...normalized_mouse_position, 1]);
			mouse_in_world = scalar_multiply4(mouse_in_world, 1/mouse_in_world[3]);
			mouse_in_world = deflate_particle_position(mouse_in_world);
			picked_scene_element.particle.position[0] = mouse_in_world[0];
			picked_scene_element.particle.position[1] = mouse_in_world[1];
			picked_scene_element.particle.remove_all_force();
		}
		canvas_picking.pick_drag_request;
		
	}
	else if(canvas_picking.pick_request!=null){ // No reset was requested, no items were previously picked, canvas requests a new pick
		frame_query.picking_query_result = [];
		frame_query.picking_query = create_picking_query(canvas_picking.pick_request, frame_query, canvas);
		canvas_picking.pick_request = null;
	}
}

function clear_picking_request(canvas_picking, frame_query){
	canvas_picking.pick_reset_request = false;
	canvas_picking.pick_drag_request = null;
	canvas_picking.pick_request = null;
	frame_query.picking_query_result = [];
}

function invert_mat4(mat4){
	const glmatrix_mat4 = glMatrix.mat4.fromValues(...mat4);
	const glmatrix_mat4_inverted = glMatrix.mat4.create();
	glMatrix.mat4.invert(glmatrix_mat4_inverted, glmatrix_mat4);
	return [...glmatrix_mat4_inverted];
	
}

function create_picking_query(pick_request, frame_query, canvas){
	return (triangle, picked_item)=>{
		if(triangle_includes_point2d(triangle, normalize_viewport_coordinates(pick_request.x, pick_request.y, canvas))){
			frame_query.picking_query_result.push(picked_item);
		}
	};
}

// Z will be set as 0
function normalize_viewport_coordinates(viewport_x, viewport_y, canvas){
	return [
		viewport_x / canvas.width * 2 - 1,
		(canvas.height - viewport_y) / canvas.height * 2 - 1,
		0
	];
}

function triangle_includes_point2d(triangle, point2d){
	const first_sign = sign(point2d, triangle[0], triangle[1]);
	const second_sign = sign(point2d, triangle[1], triangle[2]);
	const third_sign = sign(point2d, triangle[2], triangle[0]);
	
	const has_negative = (first_sign < 0) || (second_sign < 0) || (third_sign < 0);
	const has_positive = (first_sign > 0) || (second_sign > 0) || (third_sign > 0);
	
	return !(has_negative && has_positive);
}

function sign(point2d, startpoint, endpoint){
	return (point2d[0] - endpoint[0]) * (startpoint[1] - endpoint[1]) - (startpoint[0] - endpoint[0]) * (point2d[1] - endpoint[1]);
}

function createShader(gl, type, source) {
	const shader = gl.createShader(type);
	gl.shaderSource(shader, source);  
	gl.compileShader(shader);
	if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		return shader;
	}

	console.log(gl.getShaderInfoLog(shader));
	gl.deleteShader(shader);
}

function createProgram(gl, vertexShader, fragmentShader) {
	const program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
		return program;
	}

	console.log(gl.getProgramInfoLog(program));
	gl.deleteProgram(program);
}

function create_perspective_matrix(fov, aspect, near, far){
	const tanHalfFov = Math.tan(fov / 2);
	
	return [
		1/(aspect * tanHalfFov), 0, 0, 0, 
		0, 1/tanHalfFov, 0, 0, 
		0, 0, -(far + near)/(far - near), -1, 
		0, 0, -(2 * far * near) / (far - near), 0
	];
}

function create_lookat_matrix(eye, center, up){			
	const f = normalize(minus(center, eye));
	const s = normalize(cross(f, up));
	const u = normalize(cross(s, f));
	
	return [
		s[0], u[0], -f[0], 0,
		s[1], u[1], -f[1], 0,
		s[2], u[2], -f[2], 0,
		-dot(s, eye), -dot(u, eye), dot(f, eye), 1
	];
}

function create_translation_matrix(translation){
	return[
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translation[0], translation[1], translation[2], 1
	];
}

function create_rotation_matrix_along_x(angle){
	const cos_angle = Math.cos(angle);
	const sin_angle = Math.sin(angle);
	return[
		1, 0, 0, 0,
		0, cos_angle, sin_angle, 0,
		0, -sin_angle, cos_angle, 0,
		0, 0, 0, 1
	];
}

function create_scale_matrix(scale_x, scale_y, scale_z){
	return [
		scale_x, 0, 0, 0,
		0, scale_y, 0, 0,
		0, 0, scale_z, 0,
		0, 0, 0, 1
	];
}

function normalize(vec3){
	const m = magnitude(vec3);
	return [vec3[0] / m, vec3[1] / m, vec3[2] / m];
}

function magnitude(vec3){
	return Math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
}

function dot(lhs_vec3, rhs_vec3){
	return lhs_vec3[0] * rhs_vec3[0] + lhs_vec3[1] * rhs_vec3[1] + lhs_vec3[2] * rhs_vec3[2];
}

function minus(lhs_vec3, rhs_vec3){
	return [lhs_vec3[0] - rhs_vec3[0], lhs_vec3[1] - rhs_vec3[1], lhs_vec3[2] - rhs_vec3[2]];
}

function plus(lhs_vec3, rhs_vec3){
	return [lhs_vec3[0] + rhs_vec3[0], lhs_vec3[1] + rhs_vec3[1], lhs_vec3[2] + rhs_vec3[2]];
}

function cross(lhs_vec3, rhs_vec3){
	return [ 
		lhs_vec3[1] * rhs_vec3[2] - lhs_vec3[2] * rhs_vec3[1], 
		lhs_vec3[2] * rhs_vec3[0] - lhs_vec3[0] * rhs_vec3[2],
		lhs_vec3[0] * rhs_vec3[1] - lhs_vec3[1] * rhs_vec3[0]]
}

function multiply(lhs_mat4, rhs_mat4){
	return [
		dot4(row4(lhs_mat4, 0), col4(rhs_mat4, 0)), dot4(row4(lhs_mat4, 0), col4(rhs_mat4, 1)), dot4(row4(lhs_mat4, 0), col4(rhs_mat4, 2)), dot4(row4(lhs_mat4, 0), col4(rhs_mat4, 3)),
		dot4(row4(lhs_mat4, 1), col4(rhs_mat4, 0)), dot4(row4(lhs_mat4, 1), col4(rhs_mat4, 1)), dot4(row4(lhs_mat4, 1), col4(rhs_mat4, 2)), dot4(row4(lhs_mat4, 1), col4(rhs_mat4, 3)),
		dot4(row4(lhs_mat4, 2), col4(rhs_mat4, 0)), dot4(row4(lhs_mat4, 2), col4(rhs_mat4, 1)), dot4(row4(lhs_mat4, 2), col4(rhs_mat4, 2)), dot4(row4(lhs_mat4, 2), col4(rhs_mat4, 3)),
		dot4(row4(lhs_mat4, 3), col4(rhs_mat4, 0)), dot4(row4(lhs_mat4, 3), col4(rhs_mat4, 1)), dot4(row4(lhs_mat4, 3), col4(rhs_mat4, 2)), dot4(row4(lhs_mat4, 3), col4(rhs_mat4, 3))
	]
}

function multiply_4x4_4x1(mat, vec){
	return [
		dot4(col4(mat, 0), vec),
		dot4(col4(mat, 1), vec),
		dot4(col4(mat, 2), vec),
		dot4(col4(mat, 3), vec)
	]
}

function scalar_multiply(vec3, scalar){
	return [vec3[0]*scalar, vec3[1]*scalar, vec3[2]*scalar]
}

function scalar_multiply4(vec4, scalar){
	return [...scalar_multiply(vec4, scalar), vec4[3]*scalar]
}

function row4(mat, row_index){
	return row(mat, row_index, 4);
}

function row(mat, row_index, stride){
	const offset = stride * row_index;
	return [mat[offset], mat[offset + 1], mat[offset + 2], mat[offset + 3]];
}

function col4(mat, col_index){
	return col(mat, col_index, 4);
}

function col(mat, col_index, stride){
	return [mat[col_index], mat[col_index + stride], mat[col_index + stride * 2], mat[col_index + stride * 3]];
}

function dot4(lhs_vec4, rhs_vec4){
	return dot(lhs_vec4, rhs_vec4) + lhs_vec4[3] * rhs_vec4[3];
}

function create_bottom_grid_lines(near, far, lines_per_unit){
	const grid_magnitude = far - near;
	const line_amount = Math.ceil(grid_magnitude * lines_per_unit); //amount for both horizontal and vertical
	const line_spacing = grid_magnitude / line_amount;
	let lines = [];
	for(let i=0; i<line_amount+1; ++i){
		const current_horizontal_pos = i * line_spacing;
		lines.push(0); //x1
		lines.push(current_horizontal_pos); //y1
		lines.push(grid_magnitude); //x2
		lines.push(current_horizontal_pos); //y2
	}
	for(let j=0; j<line_amount+1; ++j){
		const current_vertical_pos = j * line_spacing;
		lines.push(current_vertical_pos); //x1
		lines.push(0); //y1
		lines.push(current_vertical_pos); //x2
		lines.push(grid_magnitude); //y2
	}
	return lines;
}

function render_grid(gl, grid_lines, positionAttributeLocation, vp_matrix, mvp_matrix_position, render_buffers){
	
	if(!("render_grid_mvp_matrix" in render_buffers)){
		const m_matrix = multiply_sequence([
			create_translation_matrix([-500, -500, 0]),
			create_scale_matrix(5, 1, 1),
			create_translation_matrix([0, 0, -180]),
			create_rotation_matrix_along_x(-Math.PI/2),
			create_translation_matrix([0, 0, -500])
		]);
		
		const mvp_matrix = multiply(m_matrix, vp_matrix);
		
		render_buffers.render_grid_mvp_matrix = mvp_matrix;
	}
	
	if(!("render_grid_gl_lines_buffer" in render_buffers)){
		render_buffers.render_grid_gl_lines_buffer = buffer_lines(gl, grid_lines);
	}else{
		gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_grid_gl_lines_buffer);
	}
	
				
	render_lines(
		gl, 
		grid_lines, 
		positionAttributeLocation, 
		render_buffers.render_grid_mvp_matrix, 
		mvp_matrix_position, 
		render_buffers.render_grid_gl_lines_buffer,
		2);
}

// Returns a handle to a buffer, it is guaranteed that buffer is bound to gl.ARRAY_BUFFER when this function is done
function buffer_lines(gl, lines, buffer_type = gl.STATIC_DRAW){
	const buffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), buffer_type);
	return buffer;
}

// Shader program should already be set up, line_position_buffer should be bound to gl.ARRAY_BUFFER
function render_lines(gl, lines, positionAttributeLocation, mvp_matrix, mvp_matrix_position, line_position_buffer, components_per_coordinate){
	const size = components_per_coordinate;
	const type = gl.FLOAT;   // the data is 32bit floats
	const normalize = false; // use the data as is
	const stride = 0;        // 0 = move size * sizeof(type) each iteration
	const buffer_offset = 0; // start at the beginning of the buffer
	gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, buffer_offset);
	
	gl.uniformMatrix4fv(mvp_matrix_position, false, mvp_matrix);
	
	const primitiveType = gl.LINES;
	const draw_offset = 0;
	gl.drawArrays(primitiveType, draw_offset, lines.length/size);
}

function multiply_sequence(sequence){
	let result = create_identity_matrix();
	for(let i=0; i<sequence.length; i++){
		result = multiply(result, sequence[i]);
	}
	return result;
}

function create_identity_matrix(){
	return [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	];
}

function create_cube_with_particle_corners(gl){
	const particles = [
		new Particle([-1.0, -1.0, 1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([-1.0, -1.0, -1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([1.0, -1.0, -1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([1.0, -1.0, 1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([-1.0, 1.0, 1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([-1.0, 1.0, -1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([1.0, 1.0, -1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([1.0, 1.0, 1.0], [0, 0, 0], 5.0, [0, 0, 0]),
		new Particle([0.0, 0.0, 0.0], [0, 0, 0], 5.0, [0, 0, 0]) // to be anchored
	];
	
	const diagonal_distance_in_face = Math.sqrt(8);
	const corner_to_center_distance = Math.sqrt(4 + 8) / 2.0;
	
	const springs = [
		new Spring(particles[0], particles[1], 8.0, 5.0, 2.0),
		new Spring(particles[0], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[0], particles[3], 8.0, 5.0, 2.0),
		new Spring(particles[0], particles[4], 8.0, 5.0, 2.0),
		new Spring(particles[1], particles[2], 8.0, 5.0, 2.0),
		new Spring(particles[1], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[1], particles[5], 8.0, 5.0, 2.0),
		new Spring(particles[2], particles[3], 8.0, 5.0, 2.0),
		new Spring(particles[2], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[2], particles[6], 8.0, 5.0, 2.0),
		new Spring(particles[3], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[8], particles[6], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[3], particles[7], 8.0, 5.0, 2.0),
		new Spring(particles[4], particles[5], 8.0, 5.0, 2.0),
		new Spring(particles[4], particles[7], 8.0, 5.0, 2.0),
		new Spring(particles[5], particles[6], 8.0, 5.0, 2.0),
		new Spring(particles[5], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[6], particles[7], 8.0, 5.0, 2.0),
		new Spring(particles[7], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[4], particles[8], 8.0, 5.0, corner_to_center_distance),
		new Spring(particles[0], particles[2], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[1], particles[3], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[1], particles[4], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[0], particles[5], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[2], particles[5], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[1], particles[6], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[3], particles[4], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[0], particles[7], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[3], particles[6], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[2], particles[7], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[5], particles[7], 8.0, 5.0, diagonal_distance_in_face),
		new Spring(particles[4], particles[6], 8.0, 5.0, diagonal_distance_in_face)
	];
	
	const danger_zones_texture_id = loadTexture(gl, "Android_icon_192.png");
	const diceroller_texture_id = loadTexture(gl, "diceroller_icon.png");
	
	return new DampeningForcesDecorator(
		new RenderFacesDecorator(
			new AnchoredParticleDecorator(
				new ParticleSystem(particles, springs), 
					[particles[8]]), // anchored particle
						[new ParticleSystemFace([particles[0], particles[3], particles[4], particles[7]], danger_zones_texture_id),
						new ParticleSystemFace([particles[2], particles[3], particles[6], particles[7]], diceroller_texture_id),
						new ParticleSystemFace([particles[1], particles[2], particles[5], particles[6]], danger_zones_texture_id),
						new ParticleSystemFace([particles[0], particles[3], particles[1], particles[2]], danger_zones_texture_id),
						new ParticleSystemFace([particles[7], particles[4], particles[6], particles[5]], diceroller_texture_id),
						new ParticleSystemFace([particles[0], particles[1], particles[4], particles[5]], diceroller_texture_id)]) // particles as faces in triangle strip order
		);
}

//Shader should already be set up and active, particle_position_buffer should be bound to ARRAY_BUFFER
function render_particle(gl, particle, quad_triangles, positionAttributeLocation, mvp_matrix, mvp_matrix_position, particle_position_buffer){
	const size = 2;          // 2 components per iteration
	const type = gl.FLOAT;   // the data is 32bit floats
	const normalize = false; // use the data as is
	const stride = 0;        // 0 = move size * sizeof(type) each iteration
	const buffer_offset = 0; // start at the beginning of the buffer
	gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, buffer_offset);
	
	gl.uniformMatrix4fv(mvp_matrix_position, false, mvp_matrix);
	
	const primitiveType = gl.TRIANGLE_STRIP;
	const draw_offset = 0;
	gl.drawArrays(primitiveType, draw_offset, quad_triangles.length/size);
}

// Shader should be active, positionAttributeLocation will be left enabled as VertexAttribArray
function render_particle_system_face(gl, active_shader, render_buffers, mvp_matrix, quad_triangles, face){
	if(!("render_particle_system_face_tex_buffer" in render_buffers)){
		render_buffers.render_particle_system_face_tex_buffer = gl.createBuffer();
	}
	gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_particle_system_face_tex_buffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(quad_triangles), gl.STATIC_DRAW);
	
	UpgradeToVertexArray(active_shader.texture_position_attribute_handle, 2);
	
	gl.bindTexture(gl.TEXTURE_2D, face.texture_id);
	
	if(!("render_particle_system_face_position_buffer" in render_buffers)){
		render_buffers.render_particle_system_face_position_buffer = gl.createBuffer();
	}
	gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_particle_system_face_position_buffer);
	let particle_positions = [];
	face.triangle_strip_ordered_particles.forEach((particle)=>{
		particle_positions.push(...particle.position);
	});
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(particle_positions), gl.STATIC_DRAW);
	
	const size = 3;
	UpgradeToVertexArray(active_shader.positionAttributeLocation, size);
	
	gl.uniformMatrix4fv(active_shader.mvp_matrix_position, false, mvp_matrix);
	
	const primitiveType = gl.TRIANGLE_STRIP;
	const draw_offset = 0;
	gl.drawArrays(primitiveType, draw_offset, particle_positions.length/size);
}

/** Whatever's bound to gl.ARRAY_BUFFER is upgraded to a n-dimensional VAO
 * 
 * The datatype will be gl.FLOAT
 * Coordinates will not be normalized
 * Stride = 0
 * Buffer offset = 0
 */
function UpgradeToVertexArray(attribute_handle, components_per_iteration){
	const tex_size = components_per_iteration;
	const tex_type = gl.FLOAT;   // the data is 32bit floats
	const tex_normalize = false; // use the data as is
	const tex_stride = 0;        // 0 = move size * sizeof(type) each iteration
	const tex_buffer_offset = 0; // start at the beginning of the buffer
	gl.vertexAttribPointer(attribute_handle, tex_size, tex_type, tex_normalize, tex_stride, tex_buffer_offset);
}

function render_particle_system(gl, particle_system, vp_matrix, active_shader, render_buffers, frame_query){
	const lines = [];
	particle_system.get_springs().forEach((spring)=>{
		lines.push(spring.first_particle.position[0]);
		lines.push(spring.first_particle.position[1]);
		lines.push(spring.first_particle.position[2]);
		lines.push(spring.second_particle.position[0]);
		lines.push(spring.second_particle.position[1]);
		lines.push(spring.second_particle.position[2]);
	});
	
	const m_matrix = multiply_sequence([
		create_scale_matrix(50, 50, 50),
		create_translation_matrix([0, 120, -180])
	]);
	
	const mvp_matrix = multiply(m_matrix, vp_matrix);
	
	if(!("render_particle_system_lines_buffer" in render_buffers)){
		render_buffers.render_particle_system_lines_buffer = buffer_lines(gl, lines);
	}else{
		gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_particle_system_lines_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
	}
	
	render_lines(gl, lines, active_shader.positionAttributeLocation, mvp_matrix, active_shader.mvp_matrix_position, render_buffers.render_particle_system_lines_buffer, 3);
	
	//Meant to be rendered as a triangle strip
	const quad_triangles = [
		0, 1,
		1, 1,
		0, 0,
		1, 0,			
		
	];
	
	if(!("render_particle_system_particle_buffer" in render_buffers)){
		render_buffers.render_particle_system_particle_buffer = gl.createBuffer();
		
		gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_particle_system_particle_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(quad_triangles), gl.STATIC_DRAW);
	}else{
		gl.bindBuffer(gl.ARRAY_BUFFER, render_buffers.render_particle_system_particle_buffer);
	}
	
	particle_system.get_particles().forEach((particle, index)=>{
		const m_matrix = multiply_sequence([
			create_translation_matrix([-0.5, -0.5, 0]),
			create_scale_matrix(20, 20, 1),
			create_translation_matrix(inflate_particle_position(particle.position))
		]);
		
		const mvp_matrix = multiply(m_matrix, vp_matrix);
	
		if(frame_query.picking_query!=null){
			provide_particle_triangles_to_picking_query(particle, quad_triangles, m_matrix, vp_matrix, frame_query.picking_query);
		}
	
		render_particle(gl, particle, quad_triangles, active_shader.positionAttributeLocation, mvp_matrix, active_shader.mvp_matrix_position, render_buffers.render_particle_system_particle_buffer);
	});
	
	if("get_particle_system_faces" in particle_system){
		gl.uniform1i(active_shader.useTexture_handle, 1);
		gl.enableVertexAttribArray(active_shader.texture_position_attribute_handle);
		particle_system.get_particle_system_faces().forEach((face)=>{
			render_particle_system_face(gl, active_shader, render_buffers, mvp_matrix, quad_triangles, face);
		});
		gl.disableVertexAttribArray(active_shader.texture_position_attribute_handle);
		gl.uniform1i(active_shader.useTexture_handle, 0);
	}
}

function inflate_particle_position(particle_position){
	return [particle_position[0] * 50, particle_position[1] * 50 + 120, particle_position[2] * 50 - 180];
}

// This should do the exact opposite of apply_particle_position, important for picking and dragging to work
// Because when picking/dragging, the mouse position has to be transformed into particle position space
function deflate_particle_position(particle_position){
	return [particle_position[0] / 50, (particle_position[1] - 120) / 50, particle_position[2] + 180 / 50];
}

function provide_particle_triangles_to_picking_query(particle, quad_triangles, m_matrix, vp_matrix, picking_query){
	const grouped_quad_triangles = [];
	for(let i=0; i<quad_triangles.length;i+=2){
		grouped_quad_triangles.push([quad_triangles[i], quad_triangles[i+1]]);
	}
	
	const normalized_quad_triangles = [];
	const mvp_matrix = multiply(m_matrix, vp_matrix);
	
	grouped_quad_triangles.forEach((position)=>{
		let viewport_position = multiply_4x4_4x1(mvp_matrix, [...position, 0, 1]);
		viewport_position = scalar_multiply4(viewport_position, 1/viewport_position[3]);
		normalized_quad_triangles.push(viewport_position.slice(0, 3));
	});
	
	const pick_result_object = {"particle": particle, "world_to_normalized": vp_matrix, "depth": normalized_quad_triangles[0][2]};
	
	picking_query(normalized_quad_triangles.slice(0, 3), pick_result_object);
	picking_query(normalized_quad_triangles.slice(1, 4), pick_result_object);
}

function loadTexture(gl, url) {
	const texture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, texture);

	// Because images have to be downloaded over the internet
	// they might take a moment until they are ready.
	// Until then put a single pixel in the texture so we can
	// use it immediately. When the image has finished downloading
	// we'll update the texture with the contents of the image.
	const level = 0;
	const internalFormat = gl.RGBA;
	const width = 1;
	const height = 1;
	const border = 0;
	const srcFormat = gl.RGBA;
	const srcType = gl.UNSIGNED_BYTE;
	const pixel = new Uint8Array([0, 0, 255, 255]);  // opaque blue
	gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
	width, height, border, srcFormat, srcType,
		pixel);

	const image = new Image();
	image.onload = function() {
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
			srcFormat, srcType, image);

		// WebGL1 has different requirements for power of 2 images
		// vs non power of 2 images so check if the image is a
		// power of 2 in both dimensions.
		if (isPowerOf2(image.width) && isPowerOf2(image.height)) {
			// Yes, it's a power of 2. Generate mips.
			gl.generateMipmap(gl.TEXTURE_2D);
		} else {
			// No, it's not a power of 2. Turn off mips and set
			// wrapping to clamp to edge
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		}
	};
	image.src = url;

	return texture;
}

function isPowerOf2(value) {
	return (value & (value - 1)) == 0;
}
