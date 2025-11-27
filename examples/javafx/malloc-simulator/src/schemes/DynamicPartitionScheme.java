package schemes;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.ListIterator;
import lib.Constants;
import lib.Simulator.Job;
import lib.Simulator.Partition;

public final class DynamicPartitionScheme implements PartitionScheme {

  public record CompactionResult(HashSet<Integer> updated, ArrayList<Integer> removed) {}

  ArrayList<Partition> table = null;
  PlacementStrategy placement_strategy = null;

  public DynamicPartitionScheme(
    ArrayList<Partition> table,
    Constants.PlacementStrategies strategy
  ) {
    this.table = table;

    if (strategy == Constants.PlacementStrategies.FIRST_FIT) {
      placement_strategy = new FirstFitStrategy();
    } else if (strategy == Constants.PlacementStrategies.BEST_FIT) {
      placement_strategy = new BestFitStrategy();
    } else if (strategy == Constants.PlacementStrategies.WORST_FIT) {
      placement_strategy = new WorstFitStrategy();
    }
  }

  @Override
  public int allocate(Job job) {
    int idx = placement_strategy.next(table, job.size());
    if (idx == -1) return idx;

    table.get(idx).set_job(job);
    return idx;
  }

  @Override
  public int free(Job job) {
    for (int i = 0; i < table.size(); i++) {
      Partition p = table.get(i);
      if (p.job() == null) continue;
      if (p.job() != job) continue;

      p.set_job(null);
      return i;
    }

    return -1;
  }

  public int fragment(int idx) {
    Partition p = table.get(idx);
    if (p.job() == null) return -1;

    int unused_space = p.size() - p.job().size();
    if (unused_space <= 0) return -1;

    // shift the partition numbers
    for (int i = idx + 1; i < table.size(); i++) {
      table.get(i).set_number(table.get(i).number() + 1);
    }

    p.set_size(p.job().size());
    table.add(idx + 1, new Partition(idx + 2, unused_space, null));
    return idx + 1;
  }

  public CompactionResult compact() {
    HashSet<Integer> updated = new HashSet<>();
    ArrayList<Integer> removed = new ArrayList<>();

    for (ListIterator<Partition> it = table.listIterator(); it.hasNext(); ) {
      Partition p = it.next();
      if (p.job() != null) continue;

      boolean compacted = false;
      int p_idx = it.previousIndex();

      while (it.hasNext()) {
        Partition next_p = it.next();
        if (next_p.job() != null) break;

        p.set_size(p.size() + next_p.size());
        compacted = true;
        removed.add(it.previousIndex());
        it.remove();
      }

      if (compacted) updated.add(p_idx);
    }

    if (!removed.isEmpty()) {
      for (int i = removed.getFirst(); i < table.size(); i++) {
        table.get(i).set_number(i + 1);
      }
    }

    return new CompactionResult(updated, removed);
  }
}
