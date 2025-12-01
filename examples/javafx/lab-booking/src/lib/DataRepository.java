package lib;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class DataRepository {

  LocalDatabase db = LocalDatabase.get_instance();
  Connection conn = db.connection();

  public int find_user_id_by_email(String email_address) {
    String query = "select user_id from users where email_address = ?";

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, email_address);

      ResultSet rs = statement.executeQuery();
      if (rs.next()) return rs.getInt("user_id");
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    return -1;
  }

  public String find_password_by_email(String email_address) {
    String query = "select password from users where email_address = ?";

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, email_address);

      ResultSet rs = statement.executeQuery();
      if (rs.next()) return rs.getString("password");
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    return null;
  }

  public String find_role_by_user_id(int user_id) {
    String query = "select role_name from users where user_id = ?";

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setInt(1, user_id);

      ResultSet rs = statement.executeQuery();
      if (rs.next()) return rs.getString("role_name");
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    return null;
  }

  public void create_user(
    String first_name,
    String last_name,
    String email_address,
    String password,
    String phone_number,
    String role_name
  ) {
    String query = """
      insert into users (first_name, last_name, email_address, password, phone_number, role_name)
      values(?, ?, ?, ?, ?, ?)
      """;

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, first_name);
      statement.setString(2, last_name);
      statement.setString(3, email_address);
      statement.setString(4, password);
      statement.setString(5, phone_number);
      statement.setString(6, role_name);
      statement.executeUpdate();
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }

  boolean find_seat_occupied(String seat_id) {
    String query = "select count(*) from bookings where seat_id = ?";

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, seat_id);

      ResultSet rs = statement.executeQuery();
      if (rs.next()) return rs.getInt(1) > 0;
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    return false;
  }

  public record Booking(
    String name,
    String matric_number,
    String check_in_date,
    String supervisor_name
  ) {}

  public Booking find_booking(String seat_id) {
    String query = """
      select name, matric_number, check_in_date, supervisor_name
      from bookings
      where seat_id = ?
      """;

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, seat_id);

      ResultSet rs = statement.executeQuery();
      if (rs.next()) {
        String name = rs.getString("name");
        String matric_number = rs.getString("matric_number");
        String check_in_date = rs.getString("check_in_date");
        String supervisor_name = rs.getString("supervisor_name");

        return new Booking(name, matric_number, check_in_date, supervisor_name);
      }
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    return null;
  }

  public void create_booking(
    String seat_id,
    String name,
    String matric_number,
    String check_in_date,
    String supervisor_name
  ) {
    String query = """
      insert into bookings (seat_id, name, matric_number, check_in_date, supervisor_name)
      values(?, ?, ?, ?, ?)
      on conflict (seat_id) do update set
        name = excluded.name,
        matric_number = excluded.matric_number,
        check_in_date = excluded.check_in_date,
        supervisor_name = excluded.supervisor_name
      """;

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, seat_id);
      statement.setString(2, name);
      statement.setString(3, matric_number);
      statement.setString(4, check_in_date);
      statement.setString(5, supervisor_name);
      statement.executeUpdate();
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }

  public void delete_booking(String seat_id) {
    String query = "delete from bookings where seat_id = ?";

    try {
      PreparedStatement statement = conn.prepareStatement(query);
      statement.setString(1, seat_id);
      statement.executeUpdate();
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }
}
