package schemes;

import java.util.ArrayList;
import lib.Constants;
import lib.Simulator.Job;
import lib.Simulator.Partition;

public final class FixedPartitionScheme implements PartitionScheme {

  ArrayList<Partition> table = null;
  PlacementStrategy placement_strategy = null;

  public FixedPartitionScheme(ArrayList<Partition> table, Constants.PlacementStrategies strategy) {
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
}
