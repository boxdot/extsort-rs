#[cfg(test)]
#[macro_use]
extern crate proptest;

extern crate failure;
#[macro_use]
extern crate log;
extern crate byteorder;
extern crate memmap;
extern crate rand;
extern crate tempfile;

use lower_bound::lower_bound;

use failure::Error;
use memmap::MmapMut;
use rand::{seq, SeedableRng, StdRng};
use tempfile::tempfile;

use std::fs::OpenOptions;
use std::io;
use std::mem;
use std::ops;

mod lower_bound;

/// A record is a comparable chunk of memory of constant size.
///
/// It is used by `extsort` for sorting data in an arbitrary bytes slice.
pub trait Record: Ord {
    /// Size of the record in bytes.
    const SIZE_IN_BYTES: usize;
    /// Creates the record from a bytes slice `data`.
    ///
    /// The implementation must not read more than `SIZE_IN_BYTES` bytes from `data`.
    fn from_bytes(data: &[u8]) -> Self;
    /// Writes this record to the bytes slice `data`.
    ///
    // The implementation must not write more than `SIZE_IN_BYTES` bytes to `data`.
    fn to_bytes(&self, data: &mut [u8]);
}

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

pub fn extsort<T: Record, W: io::Write>(data: &[u8], writer: &mut W) -> Result<(), Error> {
    extsort_with_options::<T, W>(data, writer, &ExtSortOptions::default())
}

pub fn extsort_with_filename<T: Record>(data: &[u8], out_filename: &str) -> Result<(), Error> {
    let out_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(out_filename)?;
    out_file.set_len(data.len() as u64)?;
    let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };
    let out_data = &mut out_mmap[..];
    let mut writer = io::Cursor::new(out_data);

    extsort_with_options::<T, io::Cursor<&mut [u8]>>(data, &mut writer, &ExtSortOptions::default())
}

pub fn extsort_with_options<T: Record, W: io::Write>(
    data: &[u8],
    writer: &mut W,
    options: &ExtSortOptions,
) -> Result<(), Error> {
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
        .map(|index| T::from_bytes(&data[index * T::SIZE_IN_BYTES..]))
        .collect();
    samples.sort();

    let mut counters = vec![0usize; num_samples + 1];
    for i in 0..num_elements {
        let value = T::from_bytes(&data[i * T::SIZE_IN_BYTES..]);
        let part = find_partition(&value, &samples);
        counters[part] += 1;
    }
    trace!("counters: {:?}", counters);

    let mut positions = prefix_sum(counters);
    trace!("positions: {:?}", positions);

    trace!("writing blocks to temporary file");
    let tmp_file = tempfile()?;
    tmp_file.set_len(data.len() as u64)?;
    let mut tmp_mmap = unsafe { MmapMut::map_mut(&tmp_file)? };
    let tmp_data = &mut tmp_mmap[..];

    let mut output_indices = positions.clone();
    for i in 0..num_elements {
        let value = T::from_bytes(&data[i * T::SIZE_IN_BYTES..]);
        let part = find_partition(&value, &samples);
        value.to_bytes(&mut tmp_data[output_indices[part] * T::SIZE_IN_BYTES..]);
        output_indices[part] += 1;
    }

    positions.push(num_elements);
    let blocks = positions.iter().zip(positions.iter().skip(1));

    let mut buf = Vec::new();
    buf.resize(T::SIZE_IN_BYTES, 0u8);

    for (&start, &end) in blocks {
        trace!(
            "writing block {:#10} - {:#10}: {:#10} elements",
            start,
            end,
            end - start
        );

        // Optimize large blocks with same constant value
        if (end - start) * element_size > options.block_size {
            let first_element = T::from_bytes(&tmp_data[start * T::SIZE_IN_BYTES..]);
            let last_element = T::from_bytes(&tmp_data[end * T::SIZE_IN_BYTES..]);
            if first_element == last_element {
                for _ in start..end {
                    first_element.to_bytes(&mut buf[..]);
                    writer.write(&buf)?;
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
            .map(|index| T::from_bytes(&tmp_data[index * T::SIZE_IN_BYTES..]))
            .collect();
        partition.sort();

        for value in partition.into_iter() {
            value.to_bytes(&mut buf);
            writer.write(&buf)?;
        }
    }

    Ok(())
}

fn find_partition<T: Ord>(value: &T, samples: &[T]) -> usize {
    lower_bound(samples, &value)
}

fn prefix_sum<I, T>(iterable: I) -> Vec<T>
where
    I: IntoIterator<Item = T>,
    T: Default + ops::AddAssign + Copy,
{
    iterable
        .into_iter()
        .scan(T::default(), |acc, value| {
            let res = Some(*acc);
            *acc += value;
            res
        })
        .collect()
}
