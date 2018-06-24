#[cfg(test)]
#[macro_use]
extern crate proptest;

extern crate failure;
#[macro_use]
extern crate log;
extern crate byteorder;
extern crate memmap;
extern crate rand;
extern crate stderrlog;

use lower_bound::lower_bound;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::Error;
use memmap::MmapMut;
use rand::{seq, SeedableRng, StdRng};

use std::fs::OpenOptions;
use std::io;
use std::mem;
use std::ops;

mod lower_bound;

pub struct ExtSortOptions {
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
pub const ELEMENT_SIZE: usize = mem::size_of::<ElementType>();

pub fn read_element(data: &[u8], index: usize) -> io::Result<ElementType> {
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

pub fn extsort<P: AsRef<str>>(data: &[u8], out_filename: P) -> Result<(), Error> {
    trace!("extsort on data of size: {}", data.len());

    let options = ExtSortOptions::default();
    let file_size = data.len();

    let num_elements = file_size / ELEMENT_SIZE;
    let num_samples =
        options.oversampling_factor * (file_size + (options.block_size - 1)) / options.block_size;

    trace!("sampling sequence of {} pivot(s)", num_samples);
    let mut rng: StdRng = SeedableRng::from_seed(options.seed);
    let mut sample_indices = seq::sample_indices(&mut rng, num_elements, num_samples);
    sample_indices.sort();
    let samples: Result<Vec<u64>, io::Error> = sample_indices
        .into_iter()
        .map(|index| -> Result<u64, io::Error> { read_element(data, index) })
        .collect();
    let mut samples = samples?;
    samples.sort();

    trace!("pivots: {:?}", samples);

    let mut counters = vec![0usize; num_samples + 1];
    for i in 0..num_elements {
        let value = read_element(data, i)?;
        let part = find_partition(value, &samples);
        counters[part] += 1;
    }
    trace!("counters: {:?}", counters);

    let mut positions = prefix_sum(&counters);
    trace!("positions: {:?}", positions);

    let tmp_filename = options.get_tmp_filename(out_filename.as_ref());
    trace!("writing blocks to temporary file: {}", tmp_filename);
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

    trace!("writing result file: {}", out_filename.as_ref());
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
        trace!(
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

            warn!(
                "large block {:#10} - {:#10}: {:#10} elements",
                start,
                end,
                end - start
            );
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
