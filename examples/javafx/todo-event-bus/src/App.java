import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {

  @Override
  public void start(Stage stage) {
    Scene scene = new Scene(new TodoView(), 640, 480);

    stage.setTitle("Todo EventBus");
    stage.setScene(scene);
    stage.show();
  }
}
