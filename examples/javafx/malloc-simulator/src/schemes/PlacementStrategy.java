package schemes;

import java.util.ArrayList;
import lib.Simulator.Partition;

sealed interface PlacementStrategy permits BestFitStrategy, FirstFitStrategy, WorstFitStrategy {
  int next(ArrayList<Partition> table, int size);
}
