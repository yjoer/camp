import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef } from 'react';

import f from '@/build/triangle.frag?raw';
import v from '@/build/triangle.vert?raw';
import { create_program, create_shader } from '@/lib/webgl';

export const Route = createFileRoute('/webgl/triangle')({
  component: Triangle,
});

function Triangle() {
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

    const positions = [0, 0, 0, 0.5, 0.7, 0];
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

    const primitive_type = gl.TRIANGLES;
    const draw_offset = 0;
    const count = 3;
    gl.drawArrays(primitive_type, draw_offset, count);

    return () => {
      canvas.remove();
    };
  }, []);

  return <div ref={ref} className="mx-2 my-1"></div>;
}
