import java.io.IOException;
import java.time.LocalDate;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.control.DatePicker;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import lib.AppState;
import lib.DataRepository;

class ReservationPage {

  FXMLLoader loader;
  ReservationController controller = new ReservationController();

  ReservationPage() {
    loader = new FXMLLoader(ReservationPage.class.getResource("/reservation.fxml"));
    loader.setController(controller);
  }

  Parent load() {
    try {
      return loader.load();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  ReservationController controller() {
    return controller;
  }
}

class ReservationController {

  DataRepository repo = new DataRepository();

  String seat_id;

  void set_seat_id(String seat_id) {
    this.seat_id = seat_id;
  }

  @FXML
  Text message;

  @FXML
  TextField cubicle_id;

  @FXML
  TextField seat_number;

  @FXML
  TextField name;

  @FXML
  TextField matric_number;

  @FXML
  DatePicker check_in_date;

  @FXML
  TextField supervisor_name;

  @FXML
  Button reserve_button;

  @FXML
  Button delete_button;

  @FXML
  Button cancel_button;

  @FXML
  void initialize() {
    cubicle_id.setText(seat_id.substring(0, 1));
    seat_number.setText(seat_id.substring(1));

    AppState state = AppState.get_instance();
    int user_id = state.user_id();
    String role = repo.find_role_by_user_id(user_id);

    if (role.equals("student")) reserve_button.setVisible(false);

    DataRepository.Booking booking = repo.find_booking(seat_id);
    if (booking == null) return;

    name.setText(booking.name());
    matric_number.setText(booking.matric_number());
    check_in_date.setValue(LocalDate.parse(booking.check_in_date()));
    supervisor_name.setText(booking.supervisor_name());

    if (role.equals("administrator")) {
      reserve_button.setText("Update");
      delete_button.setVisible(true);
    }
  }

  @FXML
  void handle_reserve() {
    if (name.getText().isEmpty()) {
      message.setText("Name is required.");
      return;
    }

    if (matric_number.getText().isEmpty()) {
      message.setText("Matric number is required.");
      return;
    }

    if (check_in_date.getValue() == null) {
      message.setText("Check-in date is required.");
      return;
    }

    if (supervisor_name.getText().isEmpty()) {
      message.setText("Supervisor's name is required.");
      return;
    }

    repo.create_booking(
      seat_id,
      name.getText(),
      matric_number.getText(),
      check_in_date.getValue().toString(),
      supervisor_name.getText()
    );

    Stage stage = (Stage) reserve_button.getScene().getWindow();
    stage.setScene(new SeatSelectionPage());
  }

  @FXML
  void handle_delete() {
    repo.delete_booking(seat_id);

    Stage stage = (Stage) delete_button.getScene().getWindow();
    stage.setScene(new SeatSelectionPage());
  }

  @FXML
  void handle_cancel() {
    Stage stage = (Stage) cancel_button.getScene().getWindow();
    stage.setScene(new SeatSelectionPage());
  }
}
