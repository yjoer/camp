package lib;

import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class LocalDatabase {

  static LocalDatabase ref = null;

  public static LocalDatabase get_instance() {
    if (ref == null) ref = new LocalDatabase();
    return ref;
  }

  Connection connection;

  LocalDatabase() {
    boolean exists = new File("data.db").exists();

    try {
      connection = DriverManager.getConnection("jdbc:sqlite:data.db");
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }

    if (exists) return;

    try (Statement statement = connection.createStatement()) {
      statement.executeUpdate("create table roles (role_name text primary key)");
      statement.executeUpdate("insert into roles values('student')");
      statement.executeUpdate("insert into roles values('administrator')");

      String users_table = """
        create table users (
          user_id integer primary key autoincrement,
          first_name text not null,
          last_name text not null,
          email_address text not null unique,
          password text not null,
          phone_number text,
          role_name text not null,
          foreign key (role_name) references roles(role_name)
        )
        """;

      String bookings_table = """
        create table bookings (
          booking_id integer primary key autoincrement,
          seat_id text not null unique,
          name text not null,
          matric_number text not null,
          check_in_date text not null,
          supervisor_name text not null
        )
        """;

      statement.executeUpdate(users_table);
      statement.executeUpdate(bookings_table);
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }

  Connection connection() {
    return connection;
  }
}
