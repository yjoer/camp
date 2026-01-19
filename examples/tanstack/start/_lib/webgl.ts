export function create_shader(
  gl: WebGL2RenderingContext,
  type: WebGLRenderingContextBase['FRAGMENT_SHADER'] | WebGLRenderingContextBase['VERTEX_SHADER'],
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('failed to create shader');

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!success) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`failed to compile shader: ${log}`);
  }

  return shader;
}

export function create_program(
  gl: WebGL2RenderingContext,
  vertex_shader: WebGLShader,
  fragment_shader: WebGLShader,
): WebGLProgram {
  const program = gl.createProgram();
  gl.attachShader(program, vertex_shader);
  gl.attachShader(program, fragment_shader);
  gl.linkProgram(program);

  const success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (!success) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`failed to link program: ${log}`);
  }

  return program;
}
