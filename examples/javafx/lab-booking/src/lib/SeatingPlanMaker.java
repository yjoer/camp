package lib;

import java.util.HashSet;
import java.util.function.Consumer;
import javafx.geometry.HPos;
import javafx.scene.Cursor;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Region;
import javafx.scene.text.Text;

public class SeatingPlanMaker {

  GridPane grid = new GridPane();

  int cubicle_id;

  int horizontal_gap = 8;
  int vertical_gap = 8;

  int n_rows = 0;
  int n_cols = 0;

  HashSet<Integer> skipped_rows = new HashSet<>();
  HashSet<Integer> skipped_cols = new HashSet<>();

  int[] skipped_x = {};
  int[] skipped_y = {};

  Consumer<String> on_click = (String seat_id) -> {};

  public SeatingPlanMaker() {
    grid.setHgap(horizontal_gap);
    grid.setVgap(vertical_gap);
  }

  public SeatingPlanMaker set_cubicle_id(int id) {
    this.cubicle_id = id;
    return this;
  }

  public SeatingPlanMaker set_horizontal_gap(int gap) {
    this.horizontal_gap = gap;
    return this;
  }

  public SeatingPlanMaker set_vertical_gap(int gap) {
    this.vertical_gap = gap;
    return this;
  }

  public SeatingPlanMaker set_rows(int n) {
    this.n_rows = n;
    return this;
  }

  public SeatingPlanMaker set_columns(int n) {
    this.n_cols = n;
    return this;
  }

  public SeatingPlanMaker set_skipped_rows(HashSet<Integer> rows) {
    this.skipped_rows = rows;
    return this;
  }

  public SeatingPlanMaker set_skipped_columns(HashSet<Integer> cols) {
    this.skipped_cols = cols;
    return this;
  }

  public SeatingPlanMaker set_skipped_x(int[] cols) {
    this.skipped_x = cols;
    return this;
  }

  public SeatingPlanMaker set_skipped_y(int[] rows) {
    this.skipped_y = rows;
    return this;
  }

  public SeatingPlanMaker set_on_click(Consumer<String> on_click) {
    this.on_click = on_click;
    return this;
  }

  void make_header() {
    int offset_x = 0;
    int offset_y = 0;

    for (int i = 0; i < n_rows + 1; i++) {
      if (skipped_rows.contains(i)) offset_y++;

      for (int j = 0; j < n_cols + 1; j++) {
        if (skipped_cols.contains(j)) offset_x++;

        if (i == 0 && j == 0) continue;

        if (i == 0) {
          Text text = new Text(String.valueOf(j));
          GridPane.setHalignment(text, HPos.CENTER);
          grid.add(text, j + offset_x, i + offset_y);
          continue;
        }

        if (j == 0) {
          Text text = new Text(String.valueOf((char) (i + 64)));
          grid.add(text, j + offset_x, i + offset_y);
        }
      }

      offset_x = 0;
    }
  }

  public GridPane make() {
    make_header();

    DataRepository repo = new DataRepository();
    int offset_x = 0;
    int offset_y = 0;

    for (int i = 1; i < n_rows + 1; i++) {
      if (skipped_rows.contains(i)) offset_y++;

      cols: for (int j = 1; j < n_cols + 1; j++) {
        if (skipped_cols.contains(j)) offset_x++;

        for (int k = 0; k < skipped_x.length; k++) {
          if (skipped_y[k] == i && skipped_x[k] == j) continue cols;
        }

        String seat_id = cubicle_id + String.valueOf((char) (i + 64)) + j;
        boolean is_occupied = repo.find_seat_occupied(seat_id);

        Region seat = IconJar.seat();
        if (is_occupied) seat.setStyle("-fx-background-color: #e91e63;");
        else seat.setStyle("-fx-background-color: #9e9e9e;");

        seat.setOnMouseEntered(e -> {
          seat.setCursor(Cursor.HAND);
          seat.setStyle("-fx-background-color: #ffc107;");
        });

        seat.setOnMouseExited(e -> {
          seat.setCursor(Cursor.DEFAULT);
          if (is_occupied) seat.setStyle("-fx-background-color: #e91e63;");
          else seat.setStyle("-fx-background-color: #9e9e9e;");
        });

        seat.setOnMouseClicked(e -> {
          on_click.accept(seat_id);
        });

        grid.add(seat, j + offset_x, i + offset_y);
      }

      offset_x = 0;
    }

    return grid;
  }
}
