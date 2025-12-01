import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import lib.LocalDatabase;

public class App extends Application {

  @Override
  public void start(Stage stage) {
    LocalDatabase.get_instance();

    Scene scene = new SignInPage();

    stage.setTitle("Lab Booking");
    stage.setScene(scene);
    stage.show();
  }
}
