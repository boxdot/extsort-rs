extern crate byteorder;
extern crate extsort;
#[macro_use]
extern crate failure;
extern crate memmap;
extern crate rand;
extern crate stderrlog;

use failure::Error;
use memmap::Mmap;

use std::env;
use std::fs::{File, OpenOptions};
use std::io;

fn main() -> Result<(), Error> {
    stderrlog::new()
        .module(module_path!())
        .verbosity(5)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();

    let mut args = env::args().skip(1);
    let filename = args.next()
        .ok_or_else(|| format_err!("Usage: external-sample-sort <filename>"))?;

    let file = OpenOptions::new().read(true).write(true).open(&filename)?;
    let file_mmap = unsafe { Mmap::map(&file)? };
    let file_data = &file_mmap[..];

    let out_filename = format!("{}.sorted", filename);
    extsort::extsort(&file_mmap[..], &out_filename)?;

    // test
    // sort in memory
    let num_elements = file_data.len() / extsort::ELEMENT_SIZE;

    let elements: Result<Vec<_>, io::Error> = (0..num_elements)
        .map(|index| extsort::read_element(&file_data, index))
        .collect();
    let mut elements = elements?;
    elements.sort();

    let sorted_file = File::open(out_filename)?;
    let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
    let sorted_data = &sorted_mmap[..];
    let sorted_elements: Result<Vec<_>, io::Error> = (0..num_elements)
        .map(|index| extsort::read_element(&sorted_data, index))
        .collect();
    let sorted_elements = sorted_elements?;

    assert_eq!(elements, sorted_elements);

    Ok(())
}
