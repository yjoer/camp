import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef } from 'react';

import f from '@/build/rectangle.frag?raw';
import v from '@/build/rectangle.vert?raw';
import { create_program, create_shader } from '@/lib/webgl';

export const Route = createFileRoute('/webgl/rectangle')({
  component: Rectangle,
});

function Rectangle() {
  const ref = useRef<HTMLDivElement>(null!);

  useEffect(() => {
    const canvas = document.createElement('canvas');
    ref.current.append(canvas);

    const gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('failed to get context');

    const vertex_shader = create_shader(gl, gl.VERTEX_SHADER, v);
    const fragment_shader = create_shader(gl, gl.FRAGMENT_SHADER, f);
    const program = create_program(gl, vertex_shader, fragment_shader);

    const a_position_location = gl.getAttribLocation(program, 'input_a_position');

    const position_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, position_buffer);

    const positions = [10, 20, 80, 20, 10, 30, 10, 30, 80, 20, 80, 30];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.enableVertexAttribArray(a_position_location);

    const size = 2;
    const type = gl.FLOAT;
    const normalize = false;
    const stride = 0;
    const offset = 0;
    gl.vertexAttribPointer(a_position_location, size, type, normalize, stride, offset);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.2, 0.2, 0.2, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    gl.bindVertexArray(vao);

    // const u_resolution_location = gl.getUniformLocation(program, 'u_resolution');
    // gl.uniform2f(u_resolution_location, gl.canvas.width, gl.canvas.height);

    const block_idx = gl.getUniformBlockIndex(program, 'GlobalParams_std140');
    const block_size = gl.getActiveUniformBlockParameter(
      program,
      block_idx,
      gl.UNIFORM_BLOCK_DATA_SIZE,
    );

    const ubo = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
    gl.bufferData(gl.UNIFORM_BUFFER, block_size, gl.DYNAMIC_DRAW);

    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);
    gl.uniformBlockBinding(program, block_idx, 0);

    const uniform_names = ['GlobalParams_std140.u_resolution'];
    const uniform_idx = gl.getUniformIndices(program, uniform_names)!;
    const uniform_offsets = gl.getActiveUniforms(program, uniform_idx, gl.UNIFORM_OFFSET);
    const u_resolution_offset = uniform_offsets[0];

    gl.bufferSubData(
      gl.UNIFORM_BUFFER,
      u_resolution_offset,
      new Float32Array([gl.canvas.width, gl.canvas.height]),
    );

    const primitive_type = gl.TRIANGLES;
    const draw_offset = 0;
    const count = 6;
    gl.drawArrays(primitive_type, draw_offset, count);

    return () => {
      canvas.remove();
    };
  }, []);

  return <div ref={ref} className="mx-2 my-1"></div>;
}
