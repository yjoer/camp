package schemes;

import lib.Simulator.Job;

public sealed interface PartitionScheme permits FixedPartitionScheme, DynamicPartitionScheme {
  int allocate(Job job);
  int free(Job job);
}
