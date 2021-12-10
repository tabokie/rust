// Test the behavior of `0 - 1` when overflow checks are disabled.
#![feature(never_type)]

#![deny(unreachable_code)]
#![allow(dead_code, invalid_value)]

enum Void {}

fn foo() -> Void {
    unsafe { std::mem::MaybeUninit::<Void>::uninit().assume_init() }
}

fn a() {
    if true {
        foo();
    }
    println!("I'm alive!");
}

fn b() {
    match true {
        true => {
            foo();
        }
        _ => {
            return;
        }
    }
    println!("I should be dead though."); // FIXME: false negative.
}

fn c() {
    match true {
        true => {
            foo();
            println!("I'm dead."); //~ ERROR unreachable expression
        }
        _ => {}
    }
}

fn main() {}
