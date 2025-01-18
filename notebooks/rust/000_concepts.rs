// %%
// :dep rand = { version = "0.9.0-beta.3" }

// %%
use rand::Rng;
use std::any::type_name_of_val;
use std::cmp::Ordering;
use std::io;

// %%
macro_rules! call {
    ($func_name:ident $(, $args:expr)*) => {
        $func_name($($args),*)
    };
}

// %% [markdown]
// ## Guessing Game

// %%
fn guessing_game() {
    println!("Guess the number!");

    let secret_number = rand::rng().random_range(1..=100);
    println!("{}", type_name_of_val(&secret_number));
    println!("The secret number is: {secret_number}");

    loop {
        let mut guess = String::new();
        println!("EVCXR_INPUT_REQUEST:Please input your guess.");

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(number) => number,
            Err(_) => {
                println!("Please type a number!");
                continue;
            }
        };

        println!("You guessed: {}", guess);

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}

// call!(guessing_game);

// %% [markdown]
// ## Variables and Mutability

// %%
fn variables() {
    let mut x = 5;
    println!("the value of x is: {x}");

    x = 6;
    println!("the value of x is: {x}");

    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
    println!("{THREE_HOURS_IN_SECONDS}");
}

call!(variables);

// %% [markdown]
// Shadowing enables us to change the type of an existing variable and reuse the same name.

// %%
fn shadowing() {
    let x = 5;
    let x = x + 1;

    {
        let x = x * 2;
        println!("the value of x in the inner scope is: {x}");
    }

    println!("the value of x is: {x}");
}

call!(shadowing);

// %% [markdown]
// ## Data Types

// %%
fn tuples() {
    let tup = (500, 6.4, 1);
    let (x, y, z) = tup;
    println!("the value of x, y, z are: {x}, {y}, {z}");

    let five_hundred = tup.0;
    let six_point_four = tup.1;
    let one = tup.2;
    println!("{five_hundred} {six_point_four} {one}");
}

call!(tuples);

// %%
fn arrays() {
    let a = [1, 2, 3, 4, 5];

    let first = a[0];
    let second = a[1];
    println!("{first} {second}");

    let a = [3; 5];
    println!("{a:?}");
}

call!(arrays);

// %% [markdown]
// ## Statements and Expressions

// %%
fn expressions() {
    let y = {
        let x = 3;
        x + 1
    };

    println!("the value of y is: {y}");
}

call!(expressions);

// %% [markdown]
// ## Control Flow

// %%
fn if_expr() {
    let number = 3;

    if number < 5 {
        println!("condition was true");
    } else {
        println!("condition was false");
    }
}

call!(if_expr);

// %%
fn else_if_expr() {
    let number = 6;

    if number % 4 == 0 {
        println!("number is divisible by 4");
    } else if number % 3 == 0 {
        println!("number is divisible by 3");
    } else if number % 2 == 0 {
        println!("number is divisible by 2");
    } else {
        println!("number is not divisible by 4, 3, or 2");
    }
}

call!(else_if_expr);

// %%
fn let_if_expr() {
    let condition = true;
    let number = if condition { 5 } else { 6 };

    println!("the value of number is: {number}");
}

call!(let_if_expr);

// %%
fn loop_expr() {
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("the result is {result}");
}

call!(loop_expr);

// %%
fn loop_labels_expr() {
    let mut count = 0;

    'counting_up: loop {
        println!("count = {count}");
        let mut remaining = 10;

        loop {
            println!("remaining = {remaining}");

            if remaining == 9 {
                break;
            }

            if count == 2 {
                break 'counting_up;
            }

            remaining -= 1;
        }

        count += 1;
    }

    println!("end count = {count}");
}

call!(loop_labels_expr);

// %%
fn while_expr() {
    let mut number = 3;

    while number != 0 {
        println!("{number}!");
        number -= 1;
    }

    println!("LIFTOFF!!!");
}

call!(while_expr);

// %%
fn for_expr() {
    let a = [10, 20, 30, 40, 50];
    let mut index = 0;

    while index < 5 {
        println!("the value is: {}", a[index]);
        index += 1;
    }

    for element in a {
        println!("the value is: {element}");
    }

    for number in (1..4).rev() {
        println!("{number}!");
    }

    println!("LIFTOFF!!!");
}

call!(for_expr);

// %%
