package schemes;

import java.util.ArrayList;
import lib.Simulator.Partition;

final class WorstFitStrategy implements PlacementStrategy {

  @Override
  public int next(ArrayList<Partition> table, int size) {
    int diff = Integer.MIN_VALUE;
    int idx = -1;

    for (int i = 0; i < table.size(); i++) {
      Partition p = table.get(i);
      if (p.job() != null) continue;
      if (p.size() < size) continue;

      int d = p.size() - size;
      if (d > diff) {
        diff = d;
        idx = i;
      }
    }

    return idx;
  }
}
