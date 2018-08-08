extern crate byteorder;
extern crate env_logger;
extern crate extsort;
extern crate memmap;
extern crate rand;
extern crate tempfile;

use byteorder::{ByteOrder, LittleEndian};
use extsort::Record;
use memmap::Mmap;
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng, StdRng};
use std::io::{BufWriter, Write};
use tempfile::{tempfile, NamedTempFile};

use std::fs::File;
use std::io;
use std::slice;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RecordU64(u64);

impl extsort::Record for RecordU64 {
    const SIZE_IN_BYTES: usize = 8;

    fn from_bytes(data: &[u8]) -> Self {
        RecordU64(LittleEndian::read_u64(data))
    }

    fn to_bytes(&self, data: &mut [u8]) {
        LittleEndian::write_u64(data, self.0)
    }
}

const SEED: [u8; 32] = *b"f16d09be9dafef9145da6d151913f288";

pub fn ref_slice<A>(s: &A) -> &[A] {
    unsafe { slice::from_raw_parts(s, 1) }
}

fn gen_constant_file(size: usize, value: u8) -> io::Result<File> {
    let mut buf_writer = BufWriter::new(tempfile()?);
    for _ in 0..size {
        buf_writer.write(ref_slice(&value))?;
    }
    Ok(buf_writer.into_inner()?)
}

fn gen_random_file(size: usize) -> io::Result<File> {
    let mut buf_writer = BufWriter::new(tempfile()?);
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    for byte in rng.sample_iter(&Uniform::new(0, 255)).take(size) {
        buf_writer.write(ref_slice(&byte))?;
    }
    Ok(buf_writer.into_inner()?)
}

fn run<F>(generate_file: F) -> Result<(), io::Error>
where
    F: FnOnce() -> io::Result<File>,
{
    let file = generate_file()?;
    let file_mmap = unsafe { Mmap::map(&file)? };
    let file_data = &file_mmap[..];

    // sort externally
    let out_file = NamedTempFile::new()?;
    let out_filename = out_file.path().to_str().unwrap();
    extsort::extsort_with_filename::<RecordU64>(&file_mmap[..], out_filename)?;

    // sort in memory
    let num_elements = file_data.len() / RecordU64::SIZE_IN_BYTES;
    let mut elements: Vec<_> = (0..num_elements)
        .map(|index| RecordU64::from_bytes(&file_data[index * RecordU64::SIZE_IN_BYTES..]))
        .collect();
    elements.sort();

    // read sorted file
    let sorted_file = File::open(out_filename)?;
    let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
    let sorted_data = &sorted_mmap[..];
    let sorted_elements: Vec<_> = (0..num_elements)
        .map(|index| RecordU64::from_bytes(&sorted_data[index * RecordU64::SIZE_IN_BYTES..]))
        .collect();

    // compare
    assert_eq!(elements, sorted_elements);

    Ok(())
}

#[test]
fn test_extsort_10mb_random() {
    let _ = env_logger::try_init();
    let res = run(|| gen_random_file(10 * 1024 * 1024));
    assert!(res.is_ok());
}

#[test]
fn test_extsort_10mb_constant() {
    let _ = env_logger::try_init();
    let res = run(|| gen_constant_file(10 * 1024 * 1024, 0));
    assert!(res.is_ok());
}
