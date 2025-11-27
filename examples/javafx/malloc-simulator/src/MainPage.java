import javafx.scene.layout.VBox;

class MainPage extends VBox {

  MainPage() {
    AllocationSettings allocation_settings = new AllocationSettings();
    TraceViewer trace_viewer = new TraceViewer();
    SimulationControls simulation_controls = new SimulationControls(trace_viewer);

    this.getChildren().addAll(allocation_settings, trace_viewer, simulation_controls);
  }
}
