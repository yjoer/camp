uniform float2 u_resolution;

struct VertexInput {
  float2 a_position;
}

struct VertexOutput {
  float4 position : SV_Position;
}

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
  float2 zero_to_one = input.a_position / u_resolution;
  float2 zero_to_two = zero_to_one * 2.0;

  float2 clip_space = zero_to_two - 1.0;

  VertexOutput output;
  output.position = float4(clip_space, 0.0, 1.0);

  return output;
}

[shader("fragment")]
float4 fs_main() : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
