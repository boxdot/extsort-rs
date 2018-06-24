extern crate byteorder;
extern crate extsort;
extern crate failure;
extern crate memmap;
extern crate rand;
extern crate stderrlog;
extern crate tempfile;

use byteorder::{ByteOrder, LittleEndian};
use failure::Error;
use memmap::Mmap;
use rand::distributions::Uniform;
use rand::Rng;
use rand::{SeedableRng, StdRng};
use std::io::{BufWriter, Write};
use tempfile::{tempfile, NamedTempFile};

use std::fs::File;
use std::io;
use std::slice;

const ELEMENT_SIZE: usize = 8; // size of u64 in bytes
const SEED: [u8; 32] = *b"f16d09be9dafef9145da6d151913f288";

pub fn read_element(data: &[u8], index: usize) -> u64 {
    let buf = &data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE];
    LittleEndian::read_u64(buf)
}

fn write_element(data: &mut [u8], index: usize, value: u64) {
    let buf = &mut data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE];
    LittleEndian::write_u64(buf, value)
}

pub fn ref_slice<A>(s: &A) -> &[A] {
    unsafe { slice::from_raw_parts(s, 1) }
}

fn gen_random_file(size: usize) -> io::Result<File> {
    let mut buf_writer = BufWriter::new(tempfile()?);
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    for byte in rng.sample_iter(&Uniform::new(0, 255)).take(size) {
        buf_writer.write(ref_slice(&byte))?;
    }
    Ok(buf_writer.into_inner()?)
}

#[test]
fn test_extsort_10mb_random() {
    let run = || -> Result<(), Error> {
        stderrlog::new().verbosity(5).init().unwrap();

        let file = gen_random_file(10 * 1024 * 1024)?;
        let file_mmap = unsafe { Mmap::map(&file)? };
        let file_data = &file_mmap[..];

        // sort externally
        let out_file = NamedTempFile::new()?;
        let out_filename = out_file.path().to_str().unwrap();
        extsort::extsort(&file_mmap[..], read_element, write_element, out_filename)?;

        // sort in memory
        let num_elements = file_data.len() / ELEMENT_SIZE;
        let mut elements: Vec<_> = (0..num_elements)
            .map(|index| read_element(&file_data, index))
            .collect();
        elements.sort();

        // read sorted file
        let sorted_file = File::open(out_filename)?;
        let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
        let sorted_data = &sorted_mmap[..];
        let sorted_elements: Vec<_> = (0..num_elements)
            .map(|index| read_element(&sorted_data, index))
            .collect();

        // compare
        assert_eq!(elements, sorted_elements);

        Ok(())
    };
    assert!(run().is_ok());
}
