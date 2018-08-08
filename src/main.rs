extern crate byteorder;
extern crate extsort;
#[macro_use]
extern crate failure;
extern crate env_logger;
extern crate memmap;
extern crate rand;

use byteorder::{ByteOrder, LittleEndian};
use extsort::Record;
use failure::Error;
use memmap::Mmap;

use std::env;
use std::fs::{File, OpenOptions};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RecordU64(u64);

impl Record for RecordU64 {
    const SIZE_IN_BYTES: usize = 8;

    fn from_bytes(data: &[u8]) -> Self {
        RecordU64(LittleEndian::read_u64(data))
    }

    fn to_bytes(&self, data: &mut [u8]) {
        LittleEndian::write_u64(data, self.0)
    }
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
    extsort::extsort_with_filename::<RecordU64>(&file_mmap[..], &out_filename)?;

    // test
    // sort in memory
    let num_elements = file_data.len() / RecordU64::SIZE_IN_BYTES;

    let mut elements: Vec<_> = (0..num_elements)
        .map(|index| RecordU64::from_bytes(&file_data[index * RecordU64::SIZE_IN_BYTES..]))
        .collect();
    elements.sort();

    let sorted_file = File::open(out_filename)?;
    let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
    let sorted_data = &sorted_mmap[..];
    let sorted_elements: Vec<_> = (0..num_elements)
        .map(|index| RecordU64::from_bytes(&sorted_data[index * RecordU64::SIZE_IN_BYTES..]))
        .collect();

    assert_eq!(elements, sorted_elements);

    Ok(())
}
