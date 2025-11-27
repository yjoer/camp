package schemes;

import java.util.ArrayList;
import lib.Simulator.Partition;

final class FirstFitStrategy implements PlacementStrategy {

  @Override
  public int next(ArrayList<Partition> table, int size) {
    for (int i = 0; i < table.size(); i++) {
      Partition p = table.get(i);
      if (p.job() != null) continue;
      if (p.size() < size) continue;

      return i;
    }

    return -1;
  }
}
