extern crate byteorder;
extern crate extsort;
#[macro_use]
extern crate failure;
extern crate env_logger;
extern crate memmap;
extern crate rand;

use byteorder::{ByteOrder, LittleEndian};
use failure::Error;
use memmap::Mmap;

use std::env;
use std::fs::{File, OpenOptions};

const ELEMENT_SIZE: usize = 8; // size of u64 in bytes

pub fn read_element(data: &[u8], index: usize) -> u64 {
    let buf = &data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE];
    LittleEndian::read_u64(buf)
}

fn write_element(data: &mut [u8], index: usize, value: u64) {
    let buf = &mut data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE];
    LittleEndian::write_u64(buf, value)
}

fn main() -> Result<(), Error> {
    env_logger::init();

    let mut args = env::args().skip(1);
    let filename = args.next()
        .ok_or_else(|| format_err!("Usage: external-sample-sort <filename>"))?;

    let file = OpenOptions::new().read(true).write(true).open(&filename)?;
    let file_mmap = unsafe { Mmap::map(&file)? };
    let file_data = &file_mmap[..];

    let out_filename = format!("{}.sorted", filename);
    extsort::extsort(&file_mmap[..], read_element, write_element, &out_filename)?;

    // test
    // sort in memory
    let num_elements = file_data.len() / ELEMENT_SIZE;

    let mut elements: Vec<_> = (0..num_elements)
        .map(|index| read_element(&file_data, index))
        .collect();
    elements.sort();

    let sorted_file = File::open(out_filename)?;
    let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
    let sorted_data = &sorted_mmap[..];
    let sorted_elements: Vec<_> = (0..num_elements)
        .map(|index| read_element(&sorted_data, index))
        .collect();

    assert_eq!(elements, sorted_elements);

    Ok(())
}
