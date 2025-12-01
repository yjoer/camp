import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Consumer;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Tab;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import lib.IconJar;
import lib.SeatingPlanMaker;

class SeatSelectionPage extends Scene {

  SeatSelectionPage() {
    super(on_create(), 800, 600);
  }

  private static Parent on_create() {
    try {
      String url = "/seat-selection.fxml";
      FXMLLoader loader = new FXMLLoader(SeatSelectionPage.class.getResource(url));
      loader.setController(new SeatSelectionController());

      return loader.load();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}

class SeatSelectionController {

  @FXML
  Tab tab_1;

  @FXML
  Tab tab_2;

  @FXML
  Tab tab_3;

  @FXML
  VBox legend;

  @FXML
  void initialize() {
    Consumer<String> handle_click = seat_id -> {
      ReservationPage p = new ReservationPage();
      ReservationController controller = p.controller();
      controller.set_seat_id(seat_id);

      Stage stage = (Stage) legend.getScene().getWindow();
      Parent parent = p.load();
      stage.setScene(new Scene(parent, 800, 600));
    };

    GridPane grid_1 = new SeatingPlanMaker()
      .set_cubicle_id(1)
      .set_rows(12)
      .set_columns(5)
      .set_skipped_rows(new HashSet<Integer>(Set.of(3, 7, 11)))
      .set_on_click(handle_click)
      .make();

    ScrollPane scroll_1 = new ScrollPane(grid_1);
    scroll_1.setPadding(new Insets(8, 16, 8, 16));
    tab_1.setContent(scroll_1);

    GridPane grid_2 = new SeatingPlanMaker()
      .set_cubicle_id(2)
      .set_rows(6)
      .set_columns(15)
      .set_skipped_rows(new HashSet<Integer>(Set.of(4)))
      .set_skipped_columns(new HashSet<Integer>(Set.of(2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15)))
      .set_skipped_x(new int[] { 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8 })
      .set_skipped_y(new int[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 })
      .set_on_click(handle_click)
      .make();

    ScrollPane scroll_2 = new ScrollPane(grid_2);
    scroll_2.setPadding(new Insets(8, 16, 8, 16));
    tab_2.setContent(scroll_2);

    GridPane grid_3 = new SeatingPlanMaker()
      .set_cubicle_id(3)
      .set_rows(13)
      .set_columns(6)
      .set_skipped_rows(new HashSet<Integer>(Set.of(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)))
      .set_skipped_columns(new HashSet<Integer>(Set.of(4)))
      .set_skipped_x(new int[] { 4, 5, 6, 4, 5, 6 })
      .set_skipped_y(new int[] { 12, 12, 12, 13, 13, 13 })
      .set_on_click(handle_click)
      .make();

    ScrollPane scroll_3 = new ScrollPane(grid_3);
    scroll_3.setPadding(new Insets(8, 16, 8, 16));
    tab_3.setContent(scroll_3);

    Text text_1 = new Text("Available Seats");
    Region seat_1 = IconJar.seat();
    seat_1.setStyle("-fx-background-color: #9e9e9e;");

    VBox vbox_1 = new VBox();
    vbox_1.setAlignment(Pos.TOP_CENTER);
    vbox_1.getChildren().addAll(seat_1, text_1);

    Text text_2 = new Text("Occupied Seats");
    Region seat_2 = IconJar.seat();
    seat_2.setStyle("-fx-background-color: #e91e63;");

    VBox vbox_2 = new VBox();
    vbox_2.setAlignment(Pos.TOP_CENTER);
    vbox_2.getChildren().addAll(seat_2, text_2);

    Text text_3 = new Text("Selected Seat");
    Region seat_3 = IconJar.seat();
    seat_3.setStyle("-fx-background-color: #ffc107;");

    VBox vbox_3 = new VBox();
    vbox_3.setAlignment(Pos.TOP_CENTER);
    vbox_3.getChildren().addAll(seat_3, text_3);

    HBox hbox = new HBox();
    hbox.setSpacing(16);
    hbox.getChildren().addAll(vbox_1, vbox_2, vbox_3);

    Text legend_title = new Text("Legend");
    legend_title.setUnderline(true);
    legend.setPadding(new Insets(8, 16, 8, 16));
    legend.setSpacing(12);
    legend.getChildren().addAll(legend_title, hbox);
  }
}
