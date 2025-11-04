struct VertexInput {
  float4 a_position;
}

struct VertexOutput {
  float4 position : SV_Position;
}

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
  VertexOutput output;
  output.position = input.a_position;

  return output;
}

[shader("fragment")]
float4 fs_main() : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
