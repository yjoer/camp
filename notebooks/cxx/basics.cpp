// %%
gClingOpts->AllowRedefinition = 1;

// %%
#include <deque>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// %% [markdown]
// ## Expressions

// %% [markdown]
// ### Pre and Post Increment Operators

// %%
int x = 0;
int y = 0;
std::cout << "x: " << ++x << std::endl;
std::cout << "y: " << y++ << std::endl;
std::cout << "y: " << y << std::endl;

// %%
float x = 0.f;
float y = 0.f;
std::cout << "x: " << ++x << std::endl;
std::cout << "y: " << y++ << std::endl;
std::cout << "y: " << y << std::endl;

// %%
double x = 0.;
double y = 0.;
std::cout << "x: " << ++x << std::endl;
std::cout << "y: " << y++ << std::endl;
std::cout << "y: " << y << std::endl;

// %% [markdown]
// ## Declarations

// %% [markdown]
// ### Arrays

// %%
int numbers[5] = {1, 2, 3, 4, 5};

for (int i = 0; i < 5; i++)
  std::cout << numbers[i] << " ";
std::cout << std::endl;

for (int i = 0; i < 5; i++)
  std::cout << *(numbers + i) << " ";
std::cout << std::endl;

// %%
int *numbers = new int[5];
numbers[0] = 1;

for (int i = 0; i < 5; i++)
  std::cout << numbers[i] << " ";
std::cout << std::endl;

delete[] numbers;

// %% [markdown]
// ## Statements

// %% [markdown]
// ### Switch Statement

// %%
int operands[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
int ops[] = {1, 1, 2, 3, 4, 1, 1, 2, 3, 4};
int value;

for (int i = 0; i < 10; i++) {
  switch (ops[i]) {
  case 1:
    value += operands[i];
    break;
  case 2:
    value -= operands[i];
    break;
  case 3:
    value *= operands[i];
    break;
  case 4:
    value /= operands[i];
    break;
  }
}

std::cout << value;

// %% [markdown]
// ### For Loop

// %%
int n = 5;
for (int i = 0; i < n; i++) {
  // 4, 3, 2, 1, 0
  for (int j = i + 1; j < n; j++) {
    std::cout << " ";
  }

  for (int k = 0; k < i + 1; k++) {
    std::cout << "* ";
  }

  std::cout << std::endl;
}

// %%
int n = 5;
for (int i = 0; i < n; i++) {
  for (int j = i + 1; j < n; j++) {
    std::cout << " ";
  }

  for (int k = 0; k < (2 * i) + 1; k++) {
    if ((k + 1) % 2 == 0) {
      std::cout << "A";
    } else {
      std::cout << "*";
    }
  }

  std::cout << std::endl;
}

// %% [markdown]
// ### Ranged-Based For Loop

// %%
std::string letters[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};

// %%
int number = 2147483647;
std::deque<int> digits;
while (number > 0) {
  digits.push_front(number % 10);
  number /= 10;
}

for (int d : digits) {
  std::cout << letters[d];
}

// %% [markdown]
// ### While Loop

// %%
double total = 67.89;
int n_fifty = floor(total / 0.5);
total = fmod(total, 0.5);

int n_twenty = floor(total / 0.2);
total = fmod(total, 0.2);

int n_ten = floor(total / 0.1);
total = fmod(total, 0.1);

int n_five = floor(total / 0.05);
total = fmod(total, 0.05);

std::cout << "50: " << n_fifty << std::endl
          << "20: " << n_twenty << std::endl
          << "10: " << n_ten << std::endl
          << "5: " << n_five << std::endl
          << "remaining: " << total;

// %%
double total = 67.89;
int n_fifty, n_twenty, n_ten, n_five;

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

std::cout << "50: " << n_fifty << std::endl
          << "20: " << n_twenty << std::endl
          << "10: " << n_ten << std::endl
          << "5: " << n_five << std::endl
          << "remaining: " << total;

// %% [markdown]
// ## Classes

// %% [markdown]
// ### Friend Declaration

// %%
class Rectangle {
  int width;
  int height;

public:
  Rectangle(int w, int h) : width(w), height(h) {}

  int area() {
    return width * height;
  }

  static Rectangle create_square(int length) {
    Rectangle square(length, length);
    return square;
  }

  friend std::string display(Rectangle &rect);
  friend class RectangleViewerA;
  friend class RectangleViewerB;
  friend class RectangleViewerC;
};

// %% [markdown]
// Access the private members via a friend function using an instance.

// %%
std::string display(Rectangle &rect) {
  return "w: " + std::to_string(rect.width) + ", h: " + std::to_string(rect.height);
}

// %% [markdown]
// Access the private members via a method taking an instance as an argument.

// %%
class RectangleViewerA {
public:
  std::string display(Rectangle &rect) {
    return "w: " + std::to_string(rect.width) + ", h: " + std::to_string(rect.height);
  }
};

// %% [markdown]
// Access the private members through inheritance.

// %%
class RectangleViewerB : Rectangle {
public:
  RectangleViewerB(Rectangle &rect) : Rectangle(rect.width, rect.height) {}

  std::string display() {
    return "w: " + std::to_string(width) + ", h: " + std::to_string(height);
  }
};

// %% [markdown]
// Access the private members through composition.

// %%
class RectangleViewerC {
  Rectangle rect;

public:
  RectangleViewerC(Rectangle r) : rect(r) {}

  std::string display() {
    return "w: " + std::to_string(rect.width) + ", h: " + std::to_string(rect.height);
  }
}

// %%
Rectangle square = Rectangle::create_square(8);
std::cout << square.area() << std::endl;

// %%
RectangleViewerA viewer_a;
RectangleViewerB viewer_b(square);
RectangleViewerC viewer_c(square);

std::cout << display(square) << std::endl;
std::cout << viewer_a.display(square) << std::endl;
std::cout << viewer_b.display() << std::endl;
std::cout << viewer_c.display();

// %% [markdown]
// ## Text Processing

// %%
std::cout << isdigit('1') << std::endl;
std::cout << isdigit('a');

// %%
std::cout << isspace(' ') << std::endl;
std::cout << isspace('1');

// %% [markdown]
// ## I/O

// %% [markdown]
// ### File I/O

// %%
char *filename = std::tmpnam(nullptr);
std::ofstream out;
out.open(filename);
out << "00000" << "," << "00001" << "," << "99" << std::endl;
out << "00001" << "," << "00002" << "," << "90" << std::endl;
out << "00002" << "," << "00003" << "," << "81" << std::endl;
out << "00003" << "," << "00004" << "," << "72" << std::endl;
out << "00004" << "," << "00005" << "," << "63" << std::endl;
out << "00005" << "," << "00006" << "," << "54" << std::endl;
out << "00006" << "," << "00007" << "," << "45" << std::endl;
out << "00007" << "," << "00008" << "," << "36" << std::endl;
out << "00008" << "," << "00009" << "," << "27" << std::endl;
out << "00009" << "," << "00000" << "," << "18" << std::endl;
out.close();

// %%
std::ifstream in;
in.open(filename);
std::vector<std::string> from, to;
std::vector<int> amount;

while (!in.eof()) {
  std::string line;
  in >> line;

  if (line.empty())
    continue;

  std::stringstream ss(line);
  std::string value;

  std::getline(ss, value, ',');
  from.push_back(value);

  std::getline(ss, value, ',');
  to.push_back(value);

  std::getline(ss, value, ',');
  amount.push_back(std::stoi(value));
}

in.close();

// %%
std::unordered_map<std::string, int> balances;

for (int i = 0; i < amount.size(); i++) {
  balances[from[i]] -= amount[i];
  balances[to[i]] += amount[i];
}

std::cout << balances["00000"];

// %%
