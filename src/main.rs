#[cfg(test)]
#[macro_use]
extern crate proptest;

#[macro_use]
extern crate failure;
#[macro_use]
extern crate log;
extern crate byteorder;
extern crate memmap;
extern crate rand;
extern crate stderrlog;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::Error;
use memmap::{Mmap, MmapMut};
use rand::{seq, SeedableRng, StdRng};

use std::env;
use std::fs::{File, OpenOptions};
use std::io;
use std::mem;
use std::ops;

struct ExtSortOptions {
    seed: [u8; 32],
    block_size: usize,
    oversampling_factor: usize,
    tmp_suffix: String,
}

impl ExtSortOptions {
    fn get_tmp_filename<S: AsRef<str>>(&self, filename: S) -> String {
        format!("{}{}", filename.as_ref(), self.tmp_suffix)
    }
}

impl Default for ExtSortOptions {
    fn default() -> Self {
        Self {
            seed: *b"f16d09be9defef9145d36d151913f288",
            block_size: 500 * 1024 * 1024, // 500 MB
            oversampling_factor: 10,
            tmp_suffix: ".tmp".into(),
        }
    }
}

type ElementType = u64;
const ELEMENT_SIZE: usize = mem::size_of::<ElementType>();

fn read_element(data: &[u8], index: usize) -> io::Result<ElementType> {
    (&data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE]).read_u64::<LittleEndian>()
}

fn write_element(data: &mut [u8], index: usize, value: ElementType) -> io::Result<()> {
    (&mut data[index * ELEMENT_SIZE..index * ELEMENT_SIZE + ELEMENT_SIZE])
        .write_u64::<LittleEndian>(value)
}

fn find_partition(value: u64, samples: &Vec<u64>) -> usize {
    lower_bound(&samples[..], value)
}

fn prefix_sum<T: Default + ops::AddAssign + Copy>(v: &Vec<T>) -> Vec<T> {
    v.iter()
        .scan(T::default(), |acc, &value| {
            let res = Some(*acc);
            *acc += value;
            res
        })
        .collect()
}

fn lower_bound<T: PartialOrd>(slice: &[T], value: T) -> usize {
    let mut s = slice;
    while !s.is_empty() {
        let mid = s.len() / 2;
        s = if s[mid] < value {
            &s[mid + 1..]
        } else {
            &s[..mid]
        };
    }
    (s.as_ptr() as usize - slice.as_ptr() as usize) / mem::size_of::<T>()
}

fn extsort<P: AsRef<str>>(data: &[u8], out_filename: P) -> Result<(), Error> {
    info!("extsort on data of size: {}", data.len());

    let options = ExtSortOptions::default();
    let file_size = data.len();

    let num_elements = file_size / ELEMENT_SIZE;
    let num_samples =
        options.oversampling_factor * (file_size + (options.block_size - 1)) / options.block_size;

    info!("sampling sequence of {} pivot(s)", num_samples);
    let mut rng: StdRng = SeedableRng::from_seed(options.seed);
    let mut sample_indices = seq::sample_indices(&mut rng, num_elements, num_samples);
    sample_indices.sort();
    let samples: Result<Vec<u64>, io::Error> = sample_indices
        .into_iter()
        .map(|index| -> Result<u64, io::Error> { read_element(data, index) })
        .collect();
    let mut samples = samples?;
    samples.sort();

    info!("pivots: {:?}", samples);

    let mut counters = vec![0usize; num_samples + 1];
    for i in 0..num_elements {
        let value = read_element(data, i)?;
        let part = find_partition(value, &samples);
        counters[part] += 1;
    }
    info!("counters: {:?}", counters);

    let mut positions = prefix_sum(&counters);
    info!("positions: {:?}", positions);

    let tmp_filename = options.get_tmp_filename(out_filename.as_ref());
    info!("writing blocks to temporary file: {}", tmp_filename);
    let tmp_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(tmp_filename)?;
    tmp_file.set_len(file_size as u64)?;
    let mut tmp_mmap = unsafe { MmapMut::map_mut(&tmp_file)? };
    let tmp_data = &mut tmp_mmap[..];

    let mut output_indices = positions.clone();
    for i in 0..num_elements {
        let value = read_element(data, i)?;
        let part = find_partition(value, &samples);
        write_element(tmp_data, output_indices[part], value)?;
        output_indices[part] += 1;
    }

    info!("writing result file: {}", out_filename.as_ref());
    let out_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(out_filename.as_ref())?;
    out_file.set_len(file_size as u64)?;
    let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };
    let out_data = &mut out_mmap[..];

    positions.push(num_elements);
    let blocks = positions.iter().zip(positions.iter().skip(1));

    for (&start, &end) in blocks {
        info!(
            "writing block {:#10} - {:#10}: {:#10} elements",
            start,
            end,
            end - start
        );

        // Optimize large blocks with same constant value
        if (end - start) * ELEMENT_SIZE > options.block_size {
            let first_element = read_element(tmp_data, start)?;
            let last_element = read_element(tmp_data, end)?;
            if first_element == last_element {
                for i in start..end {
                    write_element(out_data, i, first_element)?;
                }
                continue;
            }

            warn!("Large block: {}", end - start);
        }

        let partition: Result<Vec<u64>, io::Error> = (start..end)
            .map(|index| read_element(tmp_data, index))
            .collect();
        let mut partition = partition?;
        partition.sort();

        for (i, value) in partition.into_iter().enumerate() {
            write_element(out_data, i + start, value)?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    stderrlog::new()
        .module(module_path!())
        .verbosity(3)
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
    extsort(&file_mmap[..], &out_filename)?;

    // test
    // sort in memory
    let num_elements = file_data.len() / ELEMENT_SIZE;

    let elements: Result<Vec<_>, io::Error> = (0..num_elements)
        .map(|index| read_element(&file_data, index))
        .collect();
    let mut elements = elements?;
    elements.sort();

    let sorted_file = File::open(out_filename)?;
    let sorted_mmap = unsafe { Mmap::map(&sorted_file)? };
    let sorted_data = &sorted_mmap[..];
    let sorted_elements: Result<Vec<_>, io::Error> = (0..num_elements)
        .map(|index| read_element(&sorted_data, index))
        .collect();
    let sorted_elements = sorted_elements?;

    assert_eq!(elements, sorted_elements);

    Ok(())
}

#[cfg(test)]
mod test {
    use super::lower_bound;
    use proptest::prelude::*;

    fn lower_bound_reference(v: &Vec<usize>, value: usize) -> usize {
        v.iter()
            .enumerate()
            .find(|(_, &value_i)| value_i >= value)
            .map(|(i, _)| i)
            .unwrap_or(v.len())
    }

    proptest! {
        #[test]
        fn test_lower_bound(
            ref v in prop::collection::vec(50..100usize, 0..100),
            value in 40..110usize
        ) {
            let mut v = v.clone();
            v.sort();
            assert_eq!(lower_bound(&v, value), lower_bound_reference(&v, value));
        }
    }
}
