package lib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.function.BiConsumer;
import schemes.DynamicPartitionScheme;
import schemes.FixedPartitionScheme;
import schemes.PartitionScheme;

public class Simulator implements Iterator<ArrayList<SimulatorOps.Ops>> {

  public record Job(int number, int arrival_time, int processing_time, int size) {}

  public static class Partition {

    int number;
    int size;
    Job job;

    public Partition(int number, int size, Job job) {
      this.number = number;
      this.size = size;
      this.job = job;
    }

    public int number() {
      return number;
    }

    public void set_number(int number) {
      this.number = number;
    }

    public int size() {
      return size;
    }

    public void set_size(int size) {
      this.size = size;
    }

    public Job job() {
      return job;
    }

    public void set_job(Job job) {
      this.job = job;
    }
  }

  Constants.PartitionSchemes partition_scheme;
  Constants.PlacementStrategies placement_strategy;
  int max_partitions;
  int max_size;

  int t = 0;

  ArrayList<Job> jobs = new ArrayList<>();
  ArrayList<Job> jobs_allocated = new ArrayList<>();
  HashMap<Job, Integer> jobs_ttl = new HashMap<>();

  ArrayList<Job> waiting_jobs = new ArrayList<>();
  HashMap<Job, Integer> waiting_jobs_duration = new HashMap<>();
  ArrayList<Integer> waiting_jobs_queue_length = new ArrayList<>();

  ArrayList<Partition> partitions = new ArrayList<>();
  PartitionScheme partitioner = null;
  ArrayList<Integer> fragmentations = new ArrayList<>();

  public Simulator(
    Constants.PartitionSchemes partition_scheme,
    Constants.PlacementStrategies placement_strategy,
    int max_partitions,
    int max_size
  ) {
    this.partition_scheme = partition_scheme;
    this.placement_strategy = placement_strategy;
    this.max_partitions = max_partitions;
    this.max_size = max_size;

    if (partition_scheme == Constants.PartitionSchemes.FIXED) {
      partitioner = new FixedPartitionScheme(partitions, placement_strategy);
    } else if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) {
      partitioner = new DynamicPartitionScheme(partitions, placement_strategy);
    }
  }

  public ArrayList<SimulatorOps.Ops> load() {
    ArrayList<SimulatorOps.Ops> ops = new ArrayList<>();

    String jobs_file = """
      40
      1 0 7 5860
      2 2 11 4876
      3 3 13 4030
      4 3 7 620
      5 4 8 6050
      6 5 9 11000
      7 6 13 14208
      8 6 3 680
      9 7 16 3950
      10 7 19 6000
      11 10 6 710
      12 10 8 400
      13 11 7 9800
      14 13 11 3900
      15 13 12 5500
      16 15 21 4850
      17 15 16 10200
      18 17 4 14200
      19 17 9 2300
      20 18 5 2505
      21 18 7 660
      22 19 26 2920
      23 20 13 6580
      24 20 11 7904
      25 21 10 12050
      26 22 6 4300
      27 23 17 8000
      28 24 8 1850
      29 25 5 2705
      30 26 9 450
      31 27 14 5500
      32 28 18 3000
      33 29 9 6540
      34 30 8 3710
      35 30 20 8890
      36 31 24 6750
      37 32 2 310
      38 33 13 840
      39 34 9 4710
      40 36 11 8390
      """;

    String partitions_file = """
      10
      700
      4000
      5600
      10300
      5000
      13000
      5600
      600
      2400
      2800
      """;

    String[] jobs_entries = jobs_file.split("\n");
    String[] partitions_entries = partitions_file.split("\n");

    int n_jobs = Integer.parseInt(jobs_entries[0]);
    int n_partitions = Integer.parseInt(partitions_entries[0]);

    if (partition_scheme == Constants.PartitionSchemes.FIXED) {
      max_size = 0;

      for (int i = 1; i < n_partitions + 1; i++) {
        if (i > max_partitions) break;

        int p = Integer.parseInt(partitions_entries[i]);
        if (p > max_size) max_size = p;

        partitions.add(new Partition(i, p, null));
        ops.add(new SimulatorOps.AddPartition(null, String.valueOf(i), p));
      }
    } else if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) {
      partitions.add(new Partition(1, max_size, null));
      ops.add(new SimulatorOps.AddPartition(null, "1", max_size));
    }

    for (int i = 1; i < n_jobs + 1; i++) {
      int[] j = Arrays.stream(jobs_entries[i].split(" ")).mapToInt(Integer::parseInt).toArray();
      if (j[3] > max_size) continue;

      jobs.add(new Job(j[0], j[1], j[2], j[3]));
      ops.add(new SimulatorOps.AddJob(String.valueOf(j[0]), j[3]));
    }

    return ops;
  }

  @Override
  public boolean hasNext() {
    return !jobs.isEmpty() || !waiting_jobs.isEmpty() || !jobs_ttl.isEmpty();
  }

  @Override
  public ArrayList<SimulatorOps.Ops> next() {
    ArrayList<SimulatorOps.Ops> ops = new ArrayList<>();

    BiConsumer<Integer, Job> fragment = (idx, j) -> {
      int fragment_idx = ((DynamicPartitionScheme) partitioner).fragment(idx);
      if (fragment_idx == -1) return;

      ops.add(new SimulatorOps.UpdatePartition(idx, j.size()));

      String fragment_name = String.valueOf(fragment_idx + 1);
      int fragment_size = partitions.get(fragment_idx).size();
      ops.add(new SimulatorOps.AddPartition(fragment_idx, fragment_name, fragment_size));
    };

    Runnable compact = () -> {
      DynamicPartitionScheme dp = ((DynamicPartitionScheme) partitioner);
      DynamicPartitionScheme.CompactionResult compaction_result = dp.compact();

      if (compaction_result.updated().isEmpty()) return;

      for (int i : compaction_result.updated()) {
        Partition p = partitions.get(i);
        ops.add(new SimulatorOps.UpdatePartition(i, p.size()));
      }

      for (int i : compaction_result.removed()) {
        ops.add(new SimulatorOps.RemovePartition(i));
      }
    };

    // free jobs whose ttl has ended
    for (var it = jobs_ttl.entrySet().iterator(); it.hasNext(); ) {
      HashMap.Entry<Job, Integer> entry = it.next();
      if (entry.getValue() != t) continue;

      Job j = entry.getKey();
      int idx = partitioner.free(j);
      ops.add(new SimulatorOps.Free(idx, String.valueOf(j.number())));
      if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) compact.run();
      it.remove();
    }

    // prioritize waiting jobs
    for (Iterator<Job> it = waiting_jobs.iterator(); it.hasNext(); ) {
      Job j = it.next();

      int idx = partitioner.allocate(j);
      if (idx == -1) continue;

      ops.add(new SimulatorOps.RemoveWaitingJob(String.valueOf(j.number())));
      ops.add(new SimulatorOps.Allocate(idx, String.valueOf(j.number()), j.size()));
      if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) fragment.accept(idx, j);
      jobs_allocated.add(j);
      jobs_ttl.put(j, t + j.processing_time());
      it.remove();
    }

    // try to allocate jobs
    for (Iterator<Job> it = jobs.iterator(); it.hasNext(); ) {
      Job j = it.next();
      if (j.arrival_time() != t) continue;

      ops.add(new SimulatorOps.RemoveJob(String.valueOf(j.number())));
      it.remove();

      int idx = partitioner.allocate(j);

      if (idx == -1) {
        ops.add(new SimulatorOps.AddWaitingJob(String.valueOf(j.number()), j.size()));
        waiting_jobs.add(j);
      } else {
        ops.add(new SimulatorOps.Allocate(idx, String.valueOf(j.number()), j.size()));
        if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) fragment.accept(idx, j);
        jobs_allocated.add(j);
        jobs_ttl.put(j, t + j.processing_time());
      }
    }

    waiting_jobs_queue_length.add(waiting_jobs.size());
    for (Job j : waiting_jobs) waiting_jobs_duration.put(j, t - j.arrival_time());

    if (partition_scheme == Constants.PartitionSchemes.FIXED) {
      int sum = 0;
      for (Partition p : partitions) if (p.job() != null) sum += p.size() - p.job().size();
      fragmentations.add(sum);
    } else if (partition_scheme == Constants.PartitionSchemes.DYNAMIC) {
      int sum = 0;
      for (Partition p : partitions) if (p.job() == null) sum += p.size();
      fragmentations.add(sum);
    }

    t++;
    return ops;
  }

  public int t() {
    return t;
  }

  public double throughput() {
    int sum = 0;
    for (Job j : jobs_allocated) sum += j.processing_time();

    return (double) sum / t;
  }

  public int max_waiting_time() {
    int max = 0;
    for (int d : waiting_jobs_duration.values()) if (d > max) max = d;

    return max;
  }

  public double avg_waiting_time() {
    int sum = 0;
    for (int d : waiting_jobs_duration.values()) sum += d;

    return (double) sum / jobs_allocated.size();
  }

  public int max_queue_length() {
    int max = 0;
    for (int l : waiting_jobs_queue_length) if (l > max) max = l;

    return max;
  }

  public double avg_queue_length() {
    int sum = 0;
    for (int l : waiting_jobs_queue_length) sum += l;

    return (double) sum / t;
  }

  public double avg_fragmentation() {
    int sum = 0;
    for (int f : fragmentations) sum += f;

    return (double) sum / t;
  }
}
