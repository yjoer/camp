import java.util.ArrayList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.util.Duration;

class TraceViewer extends TabPane {

  TracesTab traces_tab = new TracesTab();
  InsightsTab insights_tab = new InsightsTab();

  TraceViewer() {
    Tab tab_1 = new Tab();
    tab_1.setText("Traces");
    tab_1.setContent(traces_tab);

    Tab tab_2 = new Tab();
    tab_2.setText("Insights");
    tab_2.setContent(insights_tab);

    this.setPadding(new Insets(8, 0, 8, 0));
    this.setTabClosingPolicy(TabClosingPolicy.UNAVAILABLE);

    this.getTabs().addAll(tab_1, tab_2);
  }
}

class TracesTab extends VBox {

  HBox jobs = new HBox();
  HBox waiting_list = new HBox();
  HBox partitions = new HBox();

  int partitions_size = 0;
  ArrayList<Integer> partitions_sizes = new ArrayList<>();

  TracesTab() {
    Label jobs_label = new Label("Jobs");

    Label waiting_list_label = new Label("Waiting List");
    VBox.setMargin(waiting_list_label, new Insets(16, 0, 0, 0));

    Label partitions_label = new Label("Partitions");
    VBox.setMargin(partitions_label, new Insets(16, 0, 0, 0));

    VBox.setMargin(jobs, new Insets(4, 0, 0, 0));
    jobs.setMinHeight(64);
    jobs.setStyle("-fx-background-color: #ffffff; -fx-border-color: #117dbb;");

    VBox.setMargin(waiting_list, new Insets(4, 0, 0, 0));
    waiting_list.setMinHeight(64);
    waiting_list.setStyle("-fx-background-color: #ffffff; -fx-border-color: #117dbb;");

    VBox.setMargin(partitions, new Insets(4, 0, 0, 0));
    partitions.setMinHeight(64);
    partitions.setStyle("-fx-background-color: #ffffff; -fx-border-color: #8b12ae;");

    this.setPadding(new Insets(8, 16, 8, 16));

    this.getChildren().addAll(
      jobs_label,
      jobs,
      waiting_list_label,
      waiting_list,
      partitions_label,
      partitions
    );
  }

  void add_job(String name, int size) {
    Label label = new Label(name);
    HBox rect = new HBox();
    rect.getChildren().add(label);
    rect.setAlignment(Pos.CENTER);

    Tooltip tooltip = new Tooltip();
    tooltip.setShowDelay(Duration.millis(0));
    tooltip.setHideDelay(Duration.millis(0));
    tooltip.setText("Job: " + name + "\nSize: " + size);
    tooltip.setStyle("-fx-font-size: 13px;");
    Tooltip.install(rect, tooltip);

    jobs.getChildren().add(rect);
    _resize_children(jobs);
  }

  void remove_job(String name) {
    jobs
      .getChildren()
      .removeIf(node -> {
        if (node instanceof HBox hbox) {
          if (hbox.getChildren().getFirst() instanceof Label label) {
            if (label.getText().equals(name)) {
              return true;
            }
          }
        }

        return false;
      });

    _resize_children(jobs);
  }

  void add_waiting_job(String name, int size) {
    Label label = new Label(name);
    HBox rect = new HBox();
    rect.getChildren().add(label);
    rect.setAlignment(Pos.CENTER);

    Tooltip tooltip = new Tooltip();
    tooltip.setShowDelay(Duration.millis(0));
    tooltip.setHideDelay(Duration.millis(0));
    tooltip.setText("Job: " + name + "\nSize: " + size);
    tooltip.setStyle("-fx-font-size: 13px;");
    Tooltip.install(rect, tooltip);

    waiting_list.getChildren().add(rect);
    _resize_children(waiting_list);
  }

  void remove_waiting_job(String name) {
    waiting_list
      .getChildren()
      .removeIf(node -> {
        if (node instanceof HBox hbox) {
          if (hbox.getChildren().getFirst() instanceof Label label) {
            if (label.getText().equals(name)) {
              return true;
            }
          }
        }

        return false;
      });

    _resize_children(waiting_list);
  }

  private void _resize_children(HBox hbox) {
    int n = hbox.getChildren().size();
    for (int i = 0; i < n; i++) {
      Node node = hbox.getChildren().get(i);

      if (node instanceof HBox box) {
        if (i != n - 1) {
          box.setStyle("-fx-border-color: #117dbb; -fx-border-width: 0 0.5 0 0;");
        } else {
          box.setStyle("");
        }

        box.prefWidthProperty().bind(hbox.widthProperty().multiply(1.0 / n));
      }
    }
  }

  void add_partition(Integer idx, String name, int size) {
    if (idx != null) partitions_sizes.add(idx, size);
    else partitions_sizes.add(size);
    partitions_size += size;

    Label label = new Label(name);
    StackPane rect = new StackPane();
    rect.getChildren().add(label);

    Tooltip tooltip = new Tooltip();
    tooltip.setShowDelay(Duration.millis(0));
    tooltip.setHideDelay(Duration.millis(0));
    tooltip.setText("Partition: " + name + "\nSize: " + size);
    tooltip.setStyle("-fx-font-size: 13px;");
    Tooltip.install(rect, tooltip);

    if (idx == null) {
      partitions.getChildren().add(rect);
    } else {
      for (int i = idx; i < partitions.getChildren().size(); i++) {
        StackPane partition = (StackPane) partitions.getChildren().get(i);
        Tooltip t = (Tooltip) partition.getProperties().get("javafx.scene.control.Tooltip");
        t.setText("Partition: " + (i + 2) + "\nSize: " + partitions_sizes.get(i + 1));

        for (Node child : partition.getChildren()) {
          if (child instanceof Label l) {
            l.setText(String.valueOf(i + 2));
          }
        }
      }

      partitions.getChildren().add(idx, rect);
    }

    _resize_partitions(partitions);
  }

  void update_partition(int idx, int size) {
    partitions_size -= partitions_sizes.get(idx);
    partitions_size += size;
    partitions_sizes.set(idx, size);

    StackPane partition = (StackPane) partitions.getChildren().get(idx);
    Tooltip tooltip = (Tooltip) partition.getProperties().get("javafx.scene.control.Tooltip");
    tooltip.setText("Partition: " + (idx + 1) + "\nSize: " + size);

    for (Node child : partition.getChildren()) {
      if (child instanceof HBox job) {
        job.prefWidthProperty().bind(partition.widthProperty().multiply(1));
        break;
      }
    }

    _resize_partitions(partitions);
  }

  void remove_partition(int idx) {
    int size = partitions_sizes.get(idx);
    partitions_size -= size;
    partitions_sizes.remove(idx);

    partitions.getChildren().remove(idx);

    _resize_partitions(partitions);
  }

  private void _resize_partitions(HBox hbox) {
    int n = hbox.getChildren().size();
    for (int i = 0; i < n; i++) {
      Node node = hbox.getChildren().get(i);

      if (node instanceof StackPane stack_pane) {
        if (i != n - 1) {
          stack_pane.setStyle("-fx-border-color: #8b12ae; -fx-border-width: 0 0.5 0 0;");
        } else {
          stack_pane.setStyle("");
        }

        double pct = (double) partitions_sizes.get(i) / partitions_size;
        stack_pane.prefWidthProperty().bind(hbox.widthProperty().multiply(pct));
      }
    }
  }

  public void allocate_partition(int idx, String name, int size) {
    StackPane partition = (StackPane) partitions.getChildren().get(idx);
    int partition_size = partitions_sizes.get(idx);
    double pct = (double) size / partition_size;

    HBox rect = new HBox();
    StackPane.setAlignment(rect, Pos.CENTER_LEFT);
    rect.setMaxWidth(Region.USE_PREF_SIZE);
    rect.prefWidthProperty().bind(partition.widthProperty().multiply(pct));
    rect.setStyle("-fx-background-color: #8b12ae20;");

    Tooltip tooltip = new Tooltip();
    tooltip.setShowDelay(Duration.millis(0));
    tooltip.setHideDelay(Duration.millis(0));
    tooltip.setText("Job: " + name + "\nSize: " + size);
    tooltip.setStyle("-fx-font-size: 13px;");
    Tooltip.install(rect, tooltip);

    partition.getChildren().add(rect);
  }

  public void free_partition(int idx) {
    StackPane partition = (StackPane) partitions.getChildren().get(idx);

    for (int i = 0; i < partition.getChildren().size(); i++) {
      Node node = partition.getChildren().get(i);

      if (node instanceof HBox) {
        partition.getChildren().remove(i);
        break;
      }
    }
  }
}

class InsightsTab extends GridPane {

  Text throughput_text = new Text("-");
  Text max_waiting_time_text = new Text("-");
  Text avg_waiting_time_text = new Text("-");
  Text max_queue_length_text = new Text("-");
  Text avg_queue_length_text = new Text("-");
  Text avg_fragmentation_text = new Text("-");

  TextArea logs_text_area = new TextArea();

  InsightsTab() {
    Label details_label = new Label("Details");
    details_label.setUnderline(true);

    Label throughput_label = new Label("Throughput");

    Label max_waiting_time_label = new Label("Maximum Waiting Time");
    max_waiting_time_label.setPadding(new Insets(12, 0, 0, 0));

    Label avg_waiting_time_label = new Label("Average Waiting Time");
    avg_waiting_time_label.setPadding(new Insets(12, 0, 0, 0));

    Label max_queue_length_label = new Label("Maximum Queue Length");
    max_queue_length_label.setPadding(new Insets(12, 0, 0, 0));

    Label avg_queue_length_label = new Label("Average Queue Length");
    avg_queue_length_label.setPadding(new Insets(12, 0, 0, 0));

    Label avg_fragmentation_label = new Label("Average Internal/External Fragmentation");
    avg_fragmentation_label.setPadding(new Insets(12, 0, 0, 0));

    Label logs_label = new Label("Logs");

    logs_text_area.setEditable(false);
    VBox.setMargin(logs_text_area, new Insets(4, 0, 0, 0));
    VBox.setVgrow(logs_text_area, Priority.ALWAYS);

    VBox details = new VBox();
    details
      .getChildren()
      .addAll(
        details_label,
        throughput_label,
        throughput_text,
        max_waiting_time_label,
        max_waiting_time_text,
        avg_waiting_time_label,
        avg_waiting_time_text,
        max_queue_length_label,
        max_queue_length_text,
        avg_queue_length_label,
        avg_queue_length_text,
        avg_fragmentation_label,
        avg_fragmentation_text
      );

    VBox logs = new VBox();
    logs.getChildren().addAll(logs_label, logs_text_area);

    ColumnConstraints column_1 = new ColumnConstraints();
    column_1.setPercentWidth(50);

    ColumnConstraints column_2 = new ColumnConstraints();
    column_2.setPercentWidth(50);

    this.getColumnConstraints().addAll(column_1, column_2);

    this.setPadding(new Insets(8, 16, 8, 16));

    this.add(details, 0, 0);
    this.add(logs, 1, 0);
  }
}
