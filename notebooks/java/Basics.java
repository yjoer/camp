// %%
import java.lang.Math;

// %% [markdown]
// ## Statements

// %% [markdown]
// ### For Loop

// %%
class StatementsFor {

  static void f1() {
    int n = 5;

    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        System.out.print(" ");
      }

      for (int k = 0; k < i + 1; k++) {
        System.out.print("* ");
      }

      System.out.println();
    }
  }
}

// %%
// StatementsFor.f1();

// %% [markdown]
// ### While Loop

// %%
class StatementsWhile {

  static void f1() {
    double total = 67.89;
    long n_fifty = (long) Math.floor(total / 0.5);
    total = total % 0.5;

    long n_twenty = (long) Math.floor(total / 0.2);
    total = total % 0.2;

    long n_ten = (long) Math.floor(total / 0.1);
    total = total % 0.1;

    long n_five = (long) Math.floor(total / 0.05);
    total = total % 0.05;

    System.out.printf("50: %d\n", n_fifty);
    System.out.printf("20: %d\n", n_twenty);
    System.out.printf("10: %d\n", n_ten);
    System.out.printf("5: %d\n", n_five);
    System.out.printf("remaining: %.2f", total);
  }

  static void f2() {
    double total = 67.89;
    int n_fifty, n_twenty, n_ten, n_five;
    n_fifty = n_twenty = n_ten = n_five = 0;

    while (total >= 0.05) {
      if (total >= 0.5) {
        n_fifty++;
        total -= 0.5;
      } else if (total >= 0.2) {
        n_twenty++;
        total -= 0.2;
      } else if (total >= 0.1) {
        n_ten++;
        total -= 0.1;
      } else if (total >= 0.05) {
        n_five++;
        total -= 0.05;
      } else {
        break;
      }
    }

    System.out.printf("50: %d\n", n_fifty);
    System.out.printf("20: %d\n", n_twenty);
    System.out.printf("10: %d\n", n_ten);
    System.out.printf("5: %d\n", n_five);
    System.out.printf("remaining: %.2f", total);
  }
}

// %%
// StatementsWhile.f1();

// %%
// StatementsWhile.f2();

// %% [markdown]
// ## Classes

// %%
class InitializationBlocks {

  static int i = 1;

  {
    i = 5;
    System.out.printf("instance initialization block: %d", i);
  }

  static {
    System.out.printf("static initialization block: %d\n", i);
  }
}

// %%
// new InitializationBlocks();

// %%
