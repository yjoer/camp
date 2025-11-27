import java.util.ArrayList;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Separator;
import javafx.scene.control.SplitMenuButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.util.Duration;
import lib.AppState;
import lib.Simulator;
import lib.SimulatorOps;

class SimulationControls extends VBox {

  Simulator simulator = null;
  boolean is_started = false;
  Timeline timeline = null;

  TraceViewer trace_viewer;

  Button start_button = new Button("Start");
  Button step_button = new Button("Step");
  SplitMenuButton fast_forward_button = new SplitMenuButton();
  MenuItem animate_item = new MenuItem("Animate");

  SimulationControls(TraceViewer trace_viewer) {
    this.trace_viewer = trace_viewer;

    start_button.setOnAction(e -> {
      if (is_started) handle_stop();
      else handle_start();
    });

    step_button.setVisible(false);
    step_button.setOnAction(e -> {
      handle_step();
    });

    animate_item.setOnAction(e -> {
      if (timeline == null) handle_animate();
      else handle_stop_animate();
    });

    fast_forward_button.setText("Fast Forward");
    fast_forward_button.getItems().addAll(animate_item);
    fast_forward_button.setVisible(false);
    fast_forward_button.setOnAction(e -> {
      handle_fast_forward();
    });

    HBox buttons = new HBox();
    buttons.setSpacing(8);
    buttons.setPadding(new Insets(8, 16, 8, 16));
    buttons.getChildren().addAll(start_button, step_button, fast_forward_button);

    Separator separator = new Separator();
    this.getChildren().addAll(separator, buttons);
  }

  private void handle_start() {
    AppState state = AppState.get_instance();

    simulator = new Simulator(
      state.get_partition_scheme(),
      state.get_placement_strategy(),
      state.get_max_partitions(),
      state.get_max_size()
    );

    ArrayList<SimulatorOps.Ops> ops = simulator.load();
    for (SimulatorOps.Ops op : ops) {
      switch (op) {
        case SimulatorOps.AddJob j -> {
          trace_viewer.traces_tab.add_job(j.name(), j.size());
        }
        case SimulatorOps.AddPartition p -> {
          trace_viewer.traces_tab.add_partition(p.idx(), p.name(), p.size());
        }
        default -> {}
      }
    }

    is_started = true;
    start_button.setText("Stop");
    step_button.setVisible(true);
    fast_forward_button.setVisible(true);

    if (simulator.hasNext()) {
      step_button.setDisable(false);
      fast_forward_button.setDisable(false);
    } else {
      step_button.setDisable(true);
      fast_forward_button.setDisable(true);
    }
  }

  private void handle_stop() {
    simulator = null;
    is_started = false;
    if (timeline != null) handle_stop_animate();

    trace_viewer.traces_tab.jobs.getChildren().clear();
    trace_viewer.traces_tab.waiting_list.getChildren().clear();
    trace_viewer.traces_tab.partitions.getChildren().clear();
    trace_viewer.traces_tab.partitions_size = 0;
    trace_viewer.traces_tab.partitions_sizes = new ArrayList<>();

    trace_viewer.insights_tab.throughput_text.setText("-");
    trace_viewer.insights_tab.max_waiting_time_text.setText("-");
    trace_viewer.insights_tab.avg_waiting_time_text.setText("-");
    trace_viewer.insights_tab.max_queue_length_text.setText("-");
    trace_viewer.insights_tab.avg_queue_length_text.setText("-");
    trace_viewer.insights_tab.avg_fragmentation_text.setText("-");

    trace_viewer.insights_tab.logs_text_area.clear();

    start_button.setText("Start");
    step_button.setVisible(false);
    fast_forward_button.setVisible(false);
  }

  private void handle_step() {
    ArrayList<SimulatorOps.Ops> ops = simulator.next();

    for (SimulatorOps.Ops op : ops) {
      switch (op) {
        case SimulatorOps.RemoveJob j -> {
          trace_viewer.traces_tab.remove_job(j.name());
        }
        case SimulatorOps.AddWaitingJob j -> {
          trace_viewer.traces_tab.add_waiting_job(j.name(), j.size());
        }
        case SimulatorOps.RemoveWaitingJob j -> {
          trace_viewer.traces_tab.remove_waiting_job(j.name());
        }
        case SimulatorOps.AddPartition p -> {
          trace_viewer.traces_tab.add_partition(p.idx(), p.name(), p.size());
        }
        case SimulatorOps.UpdatePartition p -> {
          trace_viewer.traces_tab.update_partition(p.idx(), p.size());
        }
        case SimulatorOps.RemovePartition p -> {
          trace_viewer.traces_tab.remove_partition(p.idx());
        }
        case SimulatorOps.Allocate p -> {
          trace_viewer.traces_tab.allocate_partition(p.idx(), p.name(), p.size());

          String log = String.format("t: %d, allocate %s\n", simulator.t() - 1, p.name());
          trace_viewer.insights_tab.logs_text_area.appendText(log);
        }
        case SimulatorOps.Free p -> {
          trace_viewer.traces_tab.free_partition(p.idx());

          String log = String.format("t: %d, free %s\n", simulator.t() - 1, p.name());
          trace_viewer.insights_tab.logs_text_area.appendText(log);
        }
        default -> {}
      }
    }

    String throughput = String.format("%.4f", simulator.throughput());
    trace_viewer.insights_tab.throughput_text.setText(throughput);

    String max_waiting_time = String.valueOf(simulator.max_waiting_time());
    trace_viewer.insights_tab.max_waiting_time_text.setText(max_waiting_time);

    String avg_waiting_time = String.format("%.4f", simulator.avg_waiting_time());
    trace_viewer.insights_tab.avg_waiting_time_text.setText(avg_waiting_time);

    String max_queue_length = String.valueOf(simulator.max_queue_length());
    trace_viewer.insights_tab.max_queue_length_text.setText(max_queue_length);

    String avg_queue_length = String.format("%.4f", simulator.avg_queue_length());
    trace_viewer.insights_tab.avg_queue_length_text.setText(avg_queue_length);

    String avg_fragmentation = String.format("%.4f", simulator.avg_fragmentation());
    trace_viewer.insights_tab.avg_fragmentation_text.setText(avg_fragmentation);

    if (!simulator.hasNext()) {
      step_button.setDisable(true);
      fast_forward_button.setDisable(true);
    }
  }

  private void handle_animate() {
    animate_item.setText("Stop Animating");

    timeline = new Timeline(
      new KeyFrame(Duration.millis(500), e -> {
        if (!simulator.hasNext()) {
          handle_stop_animate();
          return;
        }

        handle_step();
      })
    );

    timeline.setCycleCount(Timeline.INDEFINITE);
    timeline.play();
  }

  private void handle_stop_animate() {
    animate_item.setText("Animate");
    timeline.stop();
    timeline = null;
  }

  private void handle_fast_forward() {
    while (simulator.hasNext()) handle_step();
  }
}
