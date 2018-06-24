use std::mem;

pub fn lower_bound<T: Ord>(slice: &[T], value: &T) -> usize {
    let mut s = slice;
    while !s.is_empty() {
        let mid = s.len() / 2;
        s = if &s[mid] < value {
            &s[mid + 1..]
        } else {
            &s[..mid]
        };
    }
    (s.as_ptr() as usize - slice.as_ptr() as usize) / mem::size_of::<T>()
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
            assert_eq!(lower_bound(&v, &value), lower_bound_reference(&v, value));
        }
    }
}
