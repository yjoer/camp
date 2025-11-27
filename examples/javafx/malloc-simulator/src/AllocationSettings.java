import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import lib.AppState;
import lib.Constants;

class AllocationSettings extends GridPane {

  AllocationSettings() {
    AppState state = AppState.get_instance();

    Label scheme_label = new Label("Partition Scheme");
    Label strategy_label = new Label("Placement Strategy");
    Label max_partitions_label = new Label("Maximum Number of Partitions");
    Label max_size_label = new Label("Maximum Memory Size");

    ObservableList<Constants.PartitionSchemes> schemes = FXCollections.observableArrayList(
      Constants.PartitionSchemes.FIXED,
      Constants.PartitionSchemes.DYNAMIC
    );

    ComboBox<Constants.PartitionSchemes> scheme_selector = new ComboBox<>();
    scheme_selector.setMaxWidth(Double.MAX_VALUE);
    scheme_selector.setItems(schemes);
    scheme_selector.getSelectionModel().selectFirst();
    state.set_partition_scheme(scheme_selector.getValue());
    scheme_selector.setOnAction(e -> {
      state.set_partition_scheme(scheme_selector.getValue());
    });

    ObservableList<Constants.PlacementStrategies> strategies = FXCollections.observableArrayList(
      Constants.PlacementStrategies.FIRST_FIT,
      Constants.PlacementStrategies.BEST_FIT,
      Constants.PlacementStrategies.WORST_FIT
    );

    ComboBox<Constants.PlacementStrategies> strategy_selector = new ComboBox<>();
    strategy_selector.setMaxWidth(Double.MAX_VALUE);
    strategy_selector.setItems(strategies);
    strategy_selector.getSelectionModel().selectFirst();
    state.set_placement_strategy(strategy_selector.getValue());
    strategy_selector.setOnAction(e -> {
      state.set_placement_strategy(strategy_selector.getValue());
    });

    TextField max_partitions = new TextField();
    max_partitions.setText("10");
    state.set_max_partitions(Integer.parseInt(max_partitions.getText()));
    max_partitions.setOnKeyReleased(e -> {
      String text = max_partitions.getText();
      if (!text.matches("\\d+")) return;

      state.set_max_partitions(Integer.parseInt(text));
    });

    TextField max_size = new TextField();
    max_size.setText("50000");
    state.set_max_size(Integer.parseInt(max_size.getText()));
    max_size.setOnKeyReleased(e -> {
      String text = max_size.getText();
      if (!text.matches("\\d+")) return;

      state.set_max_size(Integer.parseInt(text));
    });

    ColumnConstraints column_1 = new ColumnConstraints();
    column_1.setPercentWidth(50);

    ColumnConstraints column_2 = new ColumnConstraints();
    column_2.setPercentWidth(50);

    this.getColumnConstraints().addAll(column_1, column_2);

    this.setPadding(new Insets(8, 16, 8, 16));
    this.setHgap(16);
    this.setVgap(8);

    this.add(scheme_label, 0, 0);
    this.add(scheme_selector, 0, 1);
    this.add(strategy_label, 1, 0);
    this.add(strategy_selector, 1, 1);
    this.add(max_partitions_label, 0, 2);
    this.add(max_partitions, 0, 3);
    this.add(max_size_label, 1, 2);
    this.add(max_size, 1, 3);
  }
}
