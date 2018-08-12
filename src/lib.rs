#[cfg(test)]
#[macro_use]
extern crate proptest;

#[macro_use]
extern crate log;
extern crate byteorder;
extern crate memmap;
extern crate rand;
extern crate tempfile;

use lower_bound::lower_bound;

use memmap::{Mmap, MmapMut};
use rand::{seq, SeedableRng, StdRng};
use tempfile::tempfile;

use std::fs::OpenOptions;
use std::io;
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

pub fn extsort<T: Record>(data: &mut [u8]) -> io::Result<()> {
    extsort_with_options::<T>(data, &ExtSortOptions::default())
}

pub fn extsort_with_writer<T: Record, W: io::Write>(data: &[u8], writer: &mut W) -> io::Result<()> {
    extsort_with_writer_and_options::<T, W>(data, writer, &ExtSortOptions::default())
}

pub fn extsort_with_filename<T: Record>(data: &[u8], out_filename: &str) -> io::Result<()> {
    let out_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(out_filename)?;
    out_file.set_len(data.len() as u64)?;
    let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };
    let out_data = &mut out_mmap[..];
    let mut writer = io::Cursor::new(out_data);
    extsort_with_writer_and_options::<T, _>(data, &mut writer, &ExtSortOptions::default())
}

pub fn extsort_with_options<T: Record>(
    data: &mut [u8],
    options: &ExtSortOptions,
) -> io::Result<()> {
    trace!("extsort inplace on data of size: {}", data.len());
    let (tmp_data, blocks) = partition::<T>(data, options)?;
    let mut writer = io::Cursor::new(data);
    merge_blocks::<T, _>(&tmp_data, &blocks[..], options.block_size, &mut writer)
}

pub fn extsort_with_writer_and_options<T: Record, W: io::Write>(
    data: &[u8],
    writer: &mut W,
    options: &ExtSortOptions,
) -> io::Result<()> {
    trace!("extsort on data of size: {}", data.len());
    let (tmp_data, blocks) = partition::<T>(data, options)?;
    merge_blocks::<T, W>(&tmp_data, &blocks[..], options.block_size, writer)
}

fn partition<T: Record>(
    data: &[u8],
    options: &ExtSortOptions,
) -> io::Result<(Mmap, Vec<(usize, usize)>)> {
    let num_elements = data.len() / T::SIZE_IN_BYTES;
    let num_samples =
        options.oversampling_factor * (data.len() + (options.block_size - 1)) / options.block_size;

    trace!("sampling sequence of {} pivot(s)", num_samples);
    let mut rng: StdRng = SeedableRng::from_seed(options.seed);
    let mut sample_indices =
        seq::sample_indices(&mut rng, num_elements, num_samples.min(num_elements));
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
    let mut tmp_data = unsafe { MmapMut::map_mut(&tmp_file)? };

    let mut output_indices = positions.clone();
    for i in 0..num_elements {
        let value = T::from_bytes(&data[i * T::SIZE_IN_BYTES..]);
        let part = find_partition(&value, &samples);
        value.to_bytes(&mut tmp_data[output_indices[part] * T::SIZE_IN_BYTES..]);
        output_indices[part] += 1;
    }
    positions.push(num_elements);

    let blocks: Vec<(usize, usize)> = positions
        .iter()
        .cloned()
        .zip(positions.iter().cloned().skip(1))
        .collect();

    tmp_data.make_read_only().map(|mmap| (mmap, blocks))
}

fn merge_blocks<T: Record, W: io::Write>(
    tmp_data: &[u8],
    blocks: &[(usize, usize)],
    block_size: usize,
    writer: &mut W,
) -> io::Result<()> {
    let mut buf = Vec::new();
    buf.resize(T::SIZE_IN_BYTES, 0u8);

    for &(start, end) in blocks {
        trace!(
            "writing block {:#10} - {:#10}: {:#10} elements",
            start,
            end,
            end - start
        );

        // Optimize large blocks with same constant value
        if (end - start) * T::SIZE_IN_BYTES > block_size {
            let first_element = T::from_bytes(&tmp_data[start * T::SIZE_IN_BYTES..]);
            let last_element = T::from_bytes(&tmp_data[end * T::SIZE_IN_BYTES..]);
            if first_element == last_element {
                for _ in start..end {
                    first_element.to_bytes(&mut buf[..]);
                    writer.write_all(&buf)?;
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

        for value in partition {
            value.to_bytes(&mut buf);
            writer.write_all(&buf)?;
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
        }).collect()
}
