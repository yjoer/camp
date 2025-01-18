// %%
macro_rules! call {
    ($func_name:ident $(, $args:expr)*) => {
        $func_name($($args),*)
    };
}

// %%
fn mutate_strings() {
    let mut s = String::from("hello");
    s.push_str(", world!");

    println!("{s}");
}

call!(mutate_strings);

// %% [markdown]
// ## Move Semantics

// %%
fn move_semantics() {
    let s1 = String::from("hello");
    let s2 = s1;

    // println!("{s1}, world!");
    println!("{s2}, world!");
}

call!(move_semantics);

// %% [markdown]
// ## Cloning and Copying

// %%
fn clone_heap_data() {
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {s1}, s2 = {s2}");
}

call!(clone_heap_data);

// %%
fn copy_stack_data() {
    let x = 5;
    let y = x;

    println!("x = {x}, y = {y}");
}

call!(copy_stack_data);

// %% [markdown]
// ## Ownership and Functions

// %%
fn makes_copy(int: i32) {
    println!("{int}");
}

fn takes_ownership(str: String) {
    println!("{str}");
}

fn gives_ownership() -> String {
    let str = String::from("hello");
    str
}

fn takes_and_gives_ownership(str: String) -> String {
    str
}

fn fn_ownership() {
    let x = 5;
    makes_copy(x);
    println!("{x}");

    let s = String::from("hello");
    takes_ownership(s);

    let s1 = gives_ownership();
    println!("{s1}");

    let s2 = String::from("hello");
    let s2 = takes_and_gives_ownership(s2);
    println!("{s2}");
}

call!(fn_ownership);

// %% [markdown]
// ## References and Borrowing

// %%
fn calculate_length(str: &String) -> usize {
    str.len()
}

fn references() {
    let s = String::from("hello");
    let len = calculate_length(&s);

    println!("the length of '{s}' is {len}");
}

call!(references);

// %%
fn change(str: &mut String) {
    str.push_str(", world");
}

fn mutable_references() {
    let mut s = String::from("hello");
    change(&mut s);

    println!("{s}");
}

call!(mutable_references);

// %% [markdown]
// ## Slices

// %%
fn first_word(str: &String) -> &str {
    let bytes = str.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &str[0..i];
        }
    }

    &str[..]
}

fn slices() {
    let s = String::from("hello world");
    let word = first_word(&s);

    println!("the first word is: {word}");
}

call!(slices);

// %%
