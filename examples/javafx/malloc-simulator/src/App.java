import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {

  @Override
  public void start(Stage stage) {
    Scene scene = new Scene(new MainPage(), 800, 600);

    stage.setTitle("Memory Allocation Simulator");
    stage.setScene(scene);
    stage.show();
  }
}
