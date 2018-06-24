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
extern crate tempfile;

use lower_bound::lower_bound;

use failure::Error;
use memmap::MmapMut;
use rand::{seq, SeedableRng, StdRng};
use tempfile::tempfile;

use std::fs::OpenOptions;
use std::mem;
use std::ops;

mod lower_bound;

pub struct ExtSortOptions {
    pub seed: [u8; 32],
    pub block_size: usize,
    pub oversampling_factor: usize,
}

impl Default for ExtSortOptions {
    fn default() -> Self {
        Self {
            seed: *b"f16d09be9defef9145d36d151913f288",
            block_size: 500 * 1024 * 1024, // 500 MB
            oversampling_factor: 10,
        }
    }
}

pub fn extsort<S, T, FnRead, FnWrite>(
    data: &[u8],
    read_element: FnRead,
    write_element: FnWrite,
    out_filename: S,
) -> Result<(), Error>
where
    S: AsRef<str>,
    T: Clone + Ord,
    FnRead: Fn(&[u8], usize) -> T,
    FnWrite: Fn(&mut [u8], usize, T),
{
    extsort_with_options(
        data,
        read_element,
        write_element,
        out_filename,
        &ExtSortOptions::default(),
    )
}

pub fn extsort_with_options<S, T, FnRead, FnWrite>(
    data: &[u8],
    read_element: FnRead,
    write_element: FnWrite,
    out_filename: S,
    options: &ExtSortOptions,
) -> Result<(), Error>
where
    S: AsRef<str>,
    T: Clone + Ord,
    FnRead: Fn(&[u8], usize) -> T,
    FnWrite: Fn(&mut [u8], usize, T),
{
    trace!("extsort on data of size: {}", data.len());

    let element_size = mem::size_of::<T>();
    let num_elements = data.len() / element_size;
    let num_samples =
        options.oversampling_factor * (data.len() + (options.block_size - 1)) / options.block_size;

    trace!("sampling sequence of {} pivot(s)", num_samples);
    let mut rng: StdRng = SeedableRng::from_seed(options.seed);
    let mut sample_indices = seq::sample_indices(&mut rng, num_elements, num_samples);
    sample_indices.sort();
    let mut samples: Vec<T> = sample_indices
        .into_iter()
        .map(|index| read_element(data, index))
        .collect();
    samples.sort();

    let mut counters = vec![0usize; num_samples + 1];
    for i in 0..num_elements {
        let value = read_element(data, i);
        let part = find_partition(&value, &samples);
        counters[part] += 1;
    }
    trace!("counters: {:?}", counters);

    let mut positions = prefix_sum(&counters);
    trace!("positions: {:?}", positions);

    trace!("writing blocks to temporary file");
    let tmp_file = tempfile()?;
    tmp_file.set_len(data.len() as u64)?;
    let mut tmp_mmap = unsafe { MmapMut::map_mut(&tmp_file)? };
    let tmp_data = &mut tmp_mmap[..];

    let mut output_indices = positions.clone();
    for i in 0..num_elements {
        let value = read_element(data, i);
        let part = find_partition(&value, &samples);
        write_element(tmp_data, output_indices[part], value);
        output_indices[part] += 1;
    }

    trace!("writing result file: {}", out_filename.as_ref());
    let out_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(out_filename.as_ref())?;
    out_file.set_len(data.len() as u64)?;
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
        if (end - start) * element_size > options.block_size {
            let first_element = read_element(tmp_data, start);
            let last_element = read_element(tmp_data, end);
            if first_element == last_element {
                for i in start..end {
                    write_element(out_data, i, first_element.clone());
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

        let mut partition: Vec<T> = (start..end)
            .map(|index| read_element(tmp_data, index))
            .collect();
        partition.sort();

        for (i, value) in partition.into_iter().enumerate() {
            write_element(out_data, i + start, value);
        }
    }

    Ok(())
}

fn find_partition<T: Ord>(value: &T, samples: &Vec<T>) -> usize {
    lower_bound(&samples[..], &value)
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
