package lib;

public class AppState {

  static AppState ref = null;

  public static AppState get_instance() {
    if (ref == null) ref = new AppState();
    return ref;
  }

  Constants.PartitionSchemes partition_scheme;
  Constants.PlacementStrategies placement_strategy;
  int max_partitions = 0;
  int max_size = 0;

  public Constants.PartitionSchemes get_partition_scheme() {
    return partition_scheme;
  }

  public void set_partition_scheme(Constants.PartitionSchemes scheme) {
    partition_scheme = scheme;
  }

  public Constants.PlacementStrategies get_placement_strategy() {
    return placement_strategy;
  }

  public void set_placement_strategy(Constants.PlacementStrategies strategy) {
    placement_strategy = strategy;
  }

  public int get_max_partitions() {
    return max_partitions;
  }

  public void set_max_partitions(int max_partitions) {
    this.max_partitions = max_partitions;
  }

  public int get_max_size() {
    return max_size;
  }

  public void set_max_size(int max_size) {
    this.max_size = max_size;
  }
}
