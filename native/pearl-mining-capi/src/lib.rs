//! C ABI for Pearl mining BLAKE3 keyed-merkle operations.
//!
//! Exposes the keyed-BLAKE3 hashing, B-seed expansion, and keyed-merkle
//! root/proof primitives the miner needs on the host side (the A/B matrix
//! commitments and inclusion proofs). The miner calls these via P/Invoke as
//! `pearl_mining_capi`.
//!
//! All exported symbols are `extern "C"` and use only POD types; the `cdylib`
//! build produces `libpearl_mining_capi.so`.
//!
//! # Memory ownership
//!
//! Functions that allocate an output buffer hand ownership to the caller, who
//! MUST release it with `pearl_capi_free_buffer` / `pearl_capi_free_u32_buffer`
//! / `pearl_capi_free_string` as documented per function.
//!
//! # Errors
//!
//! On error: returns non-zero, sets output pointers to NULL / length 0, and
//! writes a NUL-terminated message to `*err_msg_ptr` (caller frees with
//! `pearl_capi_free_string`). On success: returns 0, `err_msg_ptr` is left
//! untouched.

use std::collections::BTreeSet;
use std::os::raw::{c_char, c_int};
use pearl_blake3::{pad_to_chunk_boundary, padded_chunk_len, Blake3Hasher, MerkleTree};

// Error codes mirror the layout used by `libpearl_gemm_capi.so` so the C#
// side can keep its existing `CheckResult(int)` helper.
const PEARL_CAPI_OK: c_int = 0;
const PEARL_CAPI_ERR_BAD_ARG: c_int = 1;
const PEARL_CAPI_ERR_INTERNAL: c_int = 2;

fn set_err(err_msg_ptr: *mut *mut c_char, msg: impl Into<String>) {
    if err_msg_ptr.is_null() {
        return;
    }
    let s = msg.into();
    let bytes = s.into_bytes();
    let mut v: Vec<u8> = Vec::with_capacity(bytes.len() + 1);
    v.extend_from_slice(&bytes);
    v.push(0);
    let boxed = v.into_boxed_slice();
    let raw = Box::into_raw(boxed) as *mut c_char;
    unsafe { *err_msg_ptr = raw };
}

fn expand_bseed_int7_into(bseed: &[u8; 32], out: &mut [u8]) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(bseed);
    hasher.finalize_xof().fill(out);
    map_xof_bytes_to_int7(out);
}

fn expand_bseed_int7_range_into(bseed: &[u8; 32], byte_offset: u64, out: &mut [u8]) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(bseed);
    let mut reader = hasher.finalize_xof();
    reader.set_position(byte_offset);
    reader.fill(out);
    map_xof_bytes_to_int7(out);
}

fn map_xof_bytes_to_int7(out: &mut [u8]) {
    // Map each XOF byte b -> ((b % 127) as i8 - 63) as u8, in place.
    // `b % 127` is in [0, 126] so the `as i8` cast is lossless; subtracting
    // 63 yields i8 in [-63, 63]; the final `as u8` reinterprets the bit
    // pattern (matches C# `unchecked((byte)mapped)`).
    for slot in out {
        *slot = (((*slot % 127) as i8) - 63) as u8;
    }
}

/// Free a byte buffer previously allocated and returned by one of the
/// `pearl_capi_*` functions in this library. `len` must be the exact length
/// returned alongside the pointer; the `Box::from_raw_parts` call below
/// relies on it.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_free_buffer(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(ptr, len);
    let _ = Box::from_raw(slice as *mut [u8]);
}

/// Free a NUL-terminated error string previously produced by this library.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // Strings are allocated as Box<[u8]> with a trailing NUL; we recover
    // the length by walking until NUL. (We don't store the length anywhere
    // C-visible.) This is safe because we always wrote a NUL.
    let mut len = 0usize;
    while *ptr.add(len) != 0 {
        len += 1;
    }
    len += 1; // include NUL
    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, len);
    let _ = Box::from_raw(slice as *mut [u8]);
}

/// Library version. Bump when the ABI changes incompatibly.
///
/// History:
///   1 — initial release.
///   2 — added `pearl_capi_merkle_root_and_proof`.
#[no_mangle]
pub extern "C" fn pearl_capi_version() -> u32 {
    2
}

/// Diagnostic: BLAKE3 keyed hash of an arbitrary byte buffer.
///
/// Mirrors `pearl-blake3::blake3_digest(data, Some(key))` — used by Akoya.Miner
/// at trigger time to recompute, host-side, the hash that the on-device
/// `tensor_hash` kernel SHOULD produce. Lets us discriminate between
/// "kernel hashed different bytes" and "kernel hash is stale/contaminated".
///
/// `data_ptr` may be NULL iff `data_len == 0`.
/// `key_ptr` and `out` MUST be 32-byte buffers.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_blake3_keyed(
    data_ptr: *const u8,
    data_len: usize,
    key_ptr: *const u8,
    out: *mut u8,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || out.is_null() {
            return Err("key_ptr and out must be non-null".into());
        }
        if data_len > 0 && data_ptr.is_null() {
            return Err("data_ptr null with non-zero data_len".into());
        }
        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);
        let data: &[u8] = if data_len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(data_ptr, data_len)
        };
        let digest = pearl_blake3::blake3_digest(data, Some(key));
        std::ptr::copy_nonoverlapping(digest.as_ptr(), out, 32);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during blake3_keyed"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Compute the BLAKE3 keyed Merkle root of `data`.
///
/// `data_ptr` / `data_len` — raw matrix bytes (padded to chunk boundary by caller).
/// `key_ptr` — 32-byte key.  `out_root` — caller-allocated 32-byte buffer.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_root(
    data_ptr: *const u8,
    data_len: usize,
    key_ptr: *const u8,
    out_root: *mut u8,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || out_root.is_null() {
            return Err("key_ptr and out_root must be non-null".into());
        }
        if data_len > 0 && data_ptr.is_null() {
            return Err("data_ptr null with non-zero data_len".into());
        }
        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);
        let data: &[u8] = if data_len == 0 { &[] } else {
            std::slice::from_raw_parts(data_ptr, data_len)
        };
        let padded = pearl_blake3::pad_to_chunk_boundary(data);
        let tree = pearl_blake3::MerkleTree::new(&padded, key);
        let root = tree.root();
        std::ptr::copy_nonoverlapping(root.as_ptr(), out_root, 32);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_root"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Combined BSeed XOF expansion + int7 mapping + keyed BLAKE3 Merkle root.
///
/// This is the fused hot-path entry used by the pool's `ShareVerifier`
/// (`hashb_derivation` step). It replaces three separate C# stages —
///   1. BLAKE3 XOF expansion of `bseed` to `n*k` bytes,
///   2. per-byte mapping `b -> ((b % 127) as i8 - 63) as u8` into a row-major B
///      buffer padded to the next 1024-byte chunk boundary,
///   3. keyed BLAKE3 Merkle root over the padded B buffer
/// — with one FFI call, eliminating the ~1 MiB managed allocation pair and
/// running the merkle construction in rayon-parallel Rust.
///
/// Byte-for-byte equivalent to the managed implementation in
/// `BSeedExpander.ExpandAndComputeHashB`.
///
/// * `bseed_ptr`  — pointer to a 32-byte BSeed (XOF input).
/// * `n`, `k`     — matrix dimensions (typical: n=8192, k=128).
/// * `key_ptr`    — pointer to the 32-byte job key.
/// * `out_root`   — caller-allocated 32-byte buffer for the Merkle root.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_expand_and_merkle(
    bseed_ptr: *const u8,
    n: usize,
    k: usize,
    key_ptr: *const u8,
    out_root: *mut u8,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if bseed_ptr.is_null() || key_ptr.is_null() || out_root.is_null() {
            return Err("bseed_ptr, key_ptr and out_root must be non-null".into());
        }
        if n == 0 || k == 0 {
            return Err("n and k must be non-zero".into());
        }
        let total = n.checked_mul(k).ok_or_else(|| "n*k overflows".to_string())?;

        let mut bseed = [0u8; 32];
        std::ptr::copy_nonoverlapping(bseed_ptr, bseed.as_mut_ptr(), 32);
        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);

        // Allocate the padded B buffer up-front; XOF directly into the first
        // `total` bytes, then map in place. Tail bytes [total..padded_len) stay
        // zero from `vec![0u8; ...]`, which matches the managed pad.
        let padded_len = pearl_blake3::padded_chunk_len(total);
        let mut padded = vec![0u8; padded_len];

        expand_bseed_int7_into(&bseed, &mut padded[..total]);

        // Keyed BLAKE3 Merkle root over the padded B buffer.
        let tree = pearl_blake3::MerkleTree::new(&padded, key);
        let root = tree.root();
        std::ptr::copy_nonoverlapping(root.as_ptr(), out_root, 32);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_expand_and_merkle"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// BSeed XOF expansion + int7 mapping into a caller-owned row-major B buffer.
///
/// Byte-for-byte equivalent to the expansion half of
/// `pearl_capi_bseed_expand_and_merkle`, without padding or Merkle hashing.
/// This is used by miners that need the expanded B matrix for GPU upload.
///
/// * `bseed_ptr` — pointer to a 32-byte BSeed (XOF input).
/// * `n`, `k`    — matrix dimensions; output length is exactly `n*k`.
/// * `out_b`     — caller-owned buffer with at least `out_b_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_expand_raw(
    bseed_ptr: *const u8,
    n: usize,
    k: usize,
    out_b: *mut u8,
    out_b_len: usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if bseed_ptr.is_null() || out_b.is_null() {
            return Err("bseed_ptr and out_b must be non-null".into());
        }
        if n == 0 || k == 0 {
            return Err("n and k must be non-zero".into());
        }
        let total = n.checked_mul(k).ok_or_else(|| "n*k overflows".to_string())?;
        if out_b_len < total {
            return Err(format!("out_b_len ({}) < n*k ({})", out_b_len, total));
        }

        let mut bseed = [0u8; 32];
        std::ptr::copy_nonoverlapping(bseed_ptr, bseed.as_mut_ptr(), 32);
        let out = std::slice::from_raw_parts_mut(out_b, total);
        expand_bseed_int7_into(&bseed, out);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_expand_raw"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// BSeed XOF range expansion + int7 mapping into a caller-owned buffer.
///
/// This is equivalent to expanding the full conceptual BSeed stream and
/// copying `out_b_len` bytes starting at `byte_offset`, but uses BLAKE3 XOF
/// seeking instead of materialising the skipped prefix.
///
/// * `bseed_ptr`   — pointer to a 32-byte BSeed (XOF input).
/// * `byte_offset` — byte offset into the conceptual raw B expansion.
/// * `out_b`       — caller-owned output buffer.
/// * `out_b_len`   — number of bytes to write.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_expand_range_raw(
    bseed_ptr: *const u8,
    byte_offset: u64,
    out_b: *mut u8,
    out_b_len: usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if bseed_ptr.is_null() {
            return Err("bseed_ptr must be non-null".into());
        }
        if out_b_len > 0 && out_b.is_null() {
            return Err("out_b must be non-null when out_b_len is non-zero".into());
        }
        if out_b_len == 0 {
            return Ok(());
        }

        let mut bseed = [0u8; 32];
        std::ptr::copy_nonoverlapping(bseed_ptr, bseed.as_mut_ptr(), 32);
        let out = std::slice::from_raw_parts_mut(out_b, out_b_len);
        expand_bseed_int7_range_into(&bseed, byte_offset, out);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_expand_range_raw"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Verify a multi-leaf Merkle proof against an expected root.
///
/// Builds a `MerkleProof` from the provided fields, calls `compute_root`,
/// and compares with `expected_root`.  Returns 0 if the proof is valid.
///
/// * `leaf_data_ptr` — pointer to `leaf_count` contiguous 1024-byte chunks.
/// * `leaf_indices_ptr` — pointer to `leaf_count` u32 values (chunk indices).
/// * `siblings_ptr` — pointer to `sibling_count` contiguous 32-byte digests.
/// * `key_ptr`, `expected_root_ptr` — 32-byte buffers.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_verify_proof(
    leaf_data_ptr: *const u8,
    leaf_indices_ptr: *const u32,
    leaf_count: usize,
    total_leaves: usize,
    siblings_ptr: *const u8,
    sibling_count: usize,
    key_ptr: *const u8,
    expected_root_ptr: *const u8,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || expected_root_ptr.is_null() {
            return Err("key/root pointers must be non-null".into());
        }
        if leaf_count == 0 {
            return Err("leaf_count must be > 0".into());
        }
        if leaf_data_ptr.is_null() || leaf_indices_ptr.is_null() {
            return Err("leaf_data/indices pointers must be non-null".into());
        }
        if sibling_count > 0 && siblings_ptr.is_null() {
            return Err("siblings_ptr null with non-zero sibling_count".into());
        }

        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);
        let mut expected_root = [0u8; 32];
        std::ptr::copy_nonoverlapping(expected_root_ptr, expected_root.as_mut_ptr(), 32);

        // Reconstruct leaf_data
        let leaf_data: Vec<[u8; 1024]> = (0..leaf_count)
            .map(|i| {
                let mut chunk = [0u8; 1024];
                std::ptr::copy_nonoverlapping(leaf_data_ptr.add(i * 1024), chunk.as_mut_ptr(), 1024);
                chunk
            })
            .collect();

        // Reconstruct leaf_indices
        let leaf_indices: Vec<usize> = (0..leaf_count)
            .map(|i| *leaf_indices_ptr.add(i) as usize)
            .collect();

        // Reconstruct siblings
        let siblings: Vec<[u8; 32]> = (0..sibling_count)
            .map(|i| {
                let mut dig = [0u8; 32];
                std::ptr::copy_nonoverlapping(siblings_ptr.add(i * 32), dig.as_mut_ptr(), 32);
                dig
            })
            .collect();

        let proof = pearl_blake3::MerkleProof {
            leaf_data,
            leaf_indices,
            total_leaves,
            root: expected_root,
            siblings,
        };

        match proof.compute_root(key) {
            Some(root) if root == expected_root => Ok(()),
            Some(root) => {
                let expected_hex: String = expected_root.iter().map(|b| format!("{:02x}", b)).collect();
                let actual_hex: String = root.iter().map(|b| format!("{:02x}", b)).collect();
                Err(format!("root mismatch: expected={}, computed={}", expected_hex, actual_hex))
            }
            None => Err("proof verification failed (invalid structure or siblings)".into()),
        }
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_verify_proof"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Verify audit_proof v1 sibling paths against an expected root.
///
/// This is intentionally separate from `pearl_capi_merkle_verify_proof`.
/// The generic proof verifier consumes the crate's compact multi-leaf proof
/// format: sorted unique leaves with merged sibling paths. The pool/miner
/// audit wire format is different by design: K independent leaf-major paths,
/// duplicates allowed, with `siblings = K * log2(total_leaves)` digests.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_verify_audit_paths(
    leaf_data_ptr: *const u8,
    leaf_indices_ptr: *const u32,
    leaf_count: usize,
    total_leaves: usize,
    siblings_ptr: *const u8,
    sibling_count: usize,
    key_ptr: *const u8,
    expected_root_ptr: *const u8,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || expected_root_ptr.is_null() {
            return Err("key/root pointers must be non-null".into());
        }
        if leaf_count == 0 {
            return Err("leaf_count must be > 0".into());
        }
        if leaf_data_ptr.is_null() || leaf_indices_ptr.is_null() {
            return Err("leaf_data/indices pointers must be non-null".into());
        }
        if sibling_count > 0 && siblings_ptr.is_null() {
            return Err("siblings_ptr null with non-zero sibling_count".into());
        }
        if total_leaves == 0 {
            return Err("total_leaves must be > 0".into());
        }
        if !total_leaves.is_power_of_two() {
            return Err(format!("total_leaves ({}) must be a power of two", total_leaves));
        }

        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);
        let mut expected_root = [0u8; 32];
        std::ptr::copy_nonoverlapping(expected_root_ptr, expected_root.as_mut_ptr(), 32);

        let levels = if total_leaves <= 1 {
            0
        } else {
            total_leaves.trailing_zeros() as usize
        };
        let expected_siblings = leaf_count
            .checked_mul(levels)
            .ok_or_else(|| "leaf_count * levels overflows".to_string())?;
        if sibling_count != expected_siblings {
            return Err(format!(
                "wrong sibling count: expected={}, actual={}",
                expected_siblings, sibling_count
            ));
        }

        let hasher = Blake3Hasher::with_key(key);
        for i in 0..leaf_count {
            let leaf_idx = *leaf_indices_ptr.add(i) as usize;
            if leaf_idx >= total_leaves {
                return Err(format!(
                    "leaf index {} out of bounds for total_leaves={}",
                    leaf_idx, total_leaves
                ));
            }

            let mut leaf = [0u8; 1024];
            std::ptr::copy_nonoverlapping(leaf_data_ptr.add(i * 1024), leaf.as_mut_ptr(), 1024);

            let mut node = if total_leaves == 1 {
                hasher.hash(&leaf)
            } else {
                hasher.chunk_cv(&leaf, leaf_idx as u64)
            };

            let mut idx = leaf_idx;
            let block_base = i * levels;
            for level in 0..levels {
                let mut sibling = [0u8; 32];
                std::ptr::copy_nonoverlapping(
                    siblings_ptr.add((block_base + level) * 32),
                    sibling.as_mut_ptr(),
                    32,
                );

                let (left, right) = if idx & 1 == 0 {
                    (node, sibling)
                } else {
                    (sibling, node)
                };
                node = if level == levels - 1 {
                    hasher.root_cv(&left, &right)
                } else {
                    hasher.parent_cv(&left, &right)
                };
                idx >>= 1;
            }

            if node != expected_root {
                let expected_hex: String = expected_root.iter().map(|b| format!("{:02x}", b)).collect();
                let actual_hex: String = node.iter().map(|b| format!("{:02x}", b)).collect();
                return Err(format!(
                    "audit path root mismatch at opening {} leaf {}: expected={}, computed={}",
                    i, leaf_idx, expected_hex, actual_hex
                ));
            }
        }

        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_verify_audit_paths"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Compute the keyed BLAKE3 Merkle root **and** a multi-leaf inclusion proof
/// for the given rows in a single tree build.
///
/// This is the fused replacement for `Akoya.Crypto.Blake3.MerkleRoot` +
/// `Akoya.Crypto.MerkleProofBuilder.BuildProof` on the miner hot path: the
/// caller used to scan the same ~32 MB matrix three times (root, A-proof,
/// B-proof — and effectively rebuild the tree for each). With this we build
/// the tree once per matrix and emit both outputs.
///
/// # Inputs
///
/// * `data_ptr` / `data_len` — raw row-major matrix bytes (unpadded; this
///   function applies `pad_to_chunk_boundary` internally to mirror what the
///   C# code does).
/// * `key_ptr` — 32-byte job key.
/// * `row_indices_ptr` / `row_indices_len` — winning matrix row indices.
/// * `row_width` — bytes per row (= `k` for A or B^T).
///
/// # Outputs (all required to be non-null)
///
/// * `out_root` — caller-allocated 32-byte buffer, receives the root.
/// * `out_total_leaves` — total number of 1024-byte chunks in the padded
///   matrix.
/// * `out_leaf_data` / `out_leaf_count` — leaf chunks (`leaf_count * 1024`
///   bytes contiguous). Free with `pearl_capi_free_buffer`, passing
///   `len = leaf_count * 1024`.
/// * `out_leaf_indices` / `out_leaf_indices_len` — chunk indices, one
///   `uint32` per leaf, little-endian native. Free with
///   `pearl_capi_free_buffer`, passing `len = leaf_indices_len * 4`. (We
///   reuse the same byte-buffer freer.)
/// * `out_siblings` / `out_sibling_count` — sibling digests
///   (`sibling_count * 32` bytes contiguous). Free with
///   `pearl_capi_free_buffer`, passing `len = sibling_count * 32`. May be
///   zero-length (then `*out_siblings` is set to NULL).
///
/// Returns 0 on success, non-zero on error (with `*err_msg_ptr` set;
/// caller frees with `pearl_capi_free_string`). On error, no output buffer
/// is allocated.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_root_and_proof(
    data_ptr: *const u8,
    data_len: usize,
    key_ptr: *const u8,
    row_indices_ptr: *const u32,
    row_indices_len: usize,
    row_width: usize,
    out_root: *mut u8,
    out_total_leaves: *mut u32,
    out_leaf_data: *mut *mut u8,
    out_leaf_count: *mut usize,
    out_leaf_indices: *mut *mut u32,
    out_leaf_indices_len: *mut usize,
    out_siblings: *mut *mut u8,
    out_sibling_count: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    // Zero output pointers up-front so partial-failure paths leave the
    // caller in a consistent state.
    if !out_leaf_data.is_null()       { *out_leaf_data       = std::ptr::null_mut(); }
    if !out_leaf_count.is_null()      { *out_leaf_count      = 0; }
    if !out_leaf_indices.is_null()    { *out_leaf_indices    = std::ptr::null_mut(); }
    if !out_leaf_indices_len.is_null(){ *out_leaf_indices_len= 0; }
    if !out_siblings.is_null()        { *out_siblings        = std::ptr::null_mut(); }
    if !out_sibling_count.is_null()   { *out_sibling_count   = 0; }
    if !out_total_leaves.is_null()    { *out_total_leaves    = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || out_root.is_null()
            || out_total_leaves.is_null()
            || out_leaf_data.is_null() || out_leaf_count.is_null()
            || out_leaf_indices.is_null() || out_leaf_indices_len.is_null()
            || out_siblings.is_null() || out_sibling_count.is_null()
        {
            return Err("required out-parameter pointer was null".into());
        }
        if data_len > 0 && data_ptr.is_null() {
            return Err("data_ptr null with non-zero data_len".into());
        }
        if row_indices_len > 0 && row_indices_ptr.is_null() {
            return Err("row_indices_ptr null with non-zero row_indices_len".into());
        }
        if row_width == 0 {
            return Err("row_width must be > 0".into());
        }
        if row_indices_len == 0 {
            return Err("row_indices_len must be > 0".into());
        }

        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);

        let data: &[u8] = if data_len == 0 { &[] } else {
            std::slice::from_raw_parts(data_ptr, data_len)
        };
        let row_indices: Vec<usize> = std::slice::from_raw_parts(row_indices_ptr, row_indices_len)
            .iter()
            .map(|&r| r as usize)
            .collect();

        // Mirror the C# `BuildProof`: pad to 1024-byte chunk boundary, then
        // compute `(total_leaves, num_cols=row_width)` and derive the
        // chunk-index set from the row indices. `num_rows` is whatever
        // `data_len / row_width` works out to — we just need it for the
        // shape-tuple, not for correctness of the chunk math.
        let padded = pearl_blake3::pad_to_chunk_boundary(data);
        let tree = pearl_blake3::MerkleTree::new(&padded, key);
        let total_leaves = tree.num_leaves();
        let num_rows = if row_width == 0 { 0 } else { data.len() / row_width };

        // Bounds-check row indices to avoid an out-of-bounds chunk lookup.
        if let Some(&bad) = row_indices.iter().find(|&&r| r >= num_rows.max(1)) {
            return Err(format!(
                "row index {} out of bounds (num_rows={}, data_len={}, row_width={})",
                bad, num_rows, data.len(), row_width
            ));
        }

        let leaf_indices = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(
            &row_indices,
            (num_rows.max(1), row_width),
        );
        let proof = tree.get_multileaf_proof(&leaf_indices);

        // ---- Emit root and counts ----
        std::ptr::copy_nonoverlapping(proof.root.as_ptr(), out_root, 32);
        *out_total_leaves = total_leaves as u32;

        // ---- Pack leaf_data: leaf_count × 1024 contiguous bytes ----
        let leaf_count = proof.leaf_data.len();
        if leaf_count > 0 {
            let mut buf = vec![0u8; leaf_count * 1024];
            for (i, chunk) in proof.leaf_data.iter().enumerate() {
                buf[i * 1024..(i + 1) * 1024].copy_from_slice(chunk);
            }
            let boxed = buf.into_boxed_slice();
            *out_leaf_count = leaf_count;
            *out_leaf_data  = Box::into_raw(boxed) as *mut u8;
        }

        // ---- Pack leaf_indices: leaf_count × u32 ----
        {
            let mut idx_buf: Vec<u32> = proof.leaf_indices.iter().map(|&i| i as u32).collect();
            idx_buf.shrink_to_fit();
            let len = idx_buf.len();
            let ptr = idx_buf.as_mut_ptr();
            // Box-from-raw uses [u8] in `pearl_capi_free_buffer`; we layer a
            // u32 slice over the same allocation by forgetting the Vec and
            // exposing its byte length to the caller for freeing.
            std::mem::forget(idx_buf);
            *out_leaf_indices_len = len;
            *out_leaf_indices     = ptr;
        }

        // ---- Pack siblings: sibling_count × 32 bytes ----
        let sibling_count = proof.siblings.len();
        if sibling_count > 0 {
            let mut buf = vec![0u8; sibling_count * 32];
            for (i, sib) in proof.siblings.iter().enumerate() {
                buf[i * 32..(i + 1) * 32].copy_from_slice(sib);
            }
            let boxed = buf.into_boxed_slice();
            *out_sibling_count = sibling_count;
            *out_siblings      = Box::into_raw(boxed) as *mut u8;
        }

        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_root_and_proof"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Free a u32 buffer previously allocated by
/// `pearl_capi_merkle_root_and_proof` (the `out_leaf_indices` output).
/// `len` is the number of u32 elements (not bytes).
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_free_u32_buffer(ptr: *mut u32, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // Reconstitute the Vec exactly as it was leaked from. The capacity must
    // equal len because we called `shrink_to_fit` before forgetting.
    let _ = Vec::from_raw_parts(ptr, len, len);
}

// ============================================================================
// Merkle-tree handle CAPI (build once, prove many)
// ============================================================================
//
// `pearl_capi_merkle_root_and_proof` builds the full Merkle tree on every
// call. For the miner's B matrix that's 64 MiB of keyed-BLAKE3 per share
// (~130 ms on a desktop CPU), but the tree is constant per σ — only the
// inclusion proof for the trigger's column indices changes. Splitting the
// build/proof phases lets the miner cache the tree at σ install and pay
// only the proof-extraction cost (~µs) at trigger time.
//
// Ownership: `pearl_capi_merkle_build_tree` boxes a `MerkleTreeCtx` and
// returns the raw pointer. The caller MUST eventually call
// `pearl_capi_merkle_tree_free` on it. Calling `_proof_for_handle` does NOT
// transfer ownership; the same handle can be proved against many times.

#[allow(non_camel_case_types)]
struct MerkleTreeCtx {
    tree: MerkleTree,
    num_rows: usize,
    row_width: usize,
}

struct BSeedMerkleTreeCtx {
    bseed: [u8; 32],
    layers: Vec<Vec<[u8; 32]>>,
    root: [u8; 32],
    num_rows: usize,
    row_width: usize,
    data_len: usize,
}

fn expand_bseed_leaf(bseed: &[u8; 32], data_len: usize, leaf_idx: usize) -> [u8; 1024] {
    let mut leaf = [0u8; 1024];
    let offset = leaf_idx.saturating_mul(1024);
    if offset < data_len {
        let valid = (data_len - offset).min(1024);
        expand_bseed_int7_range_into(bseed, offset as u64, &mut leaf[..valid]);
    }
    leaf
}

fn build_bseed_merkle_ctx(
    leaf_cvs: &[[u8; 32]],
    bseed: [u8; 32],
    key: [u8; 32],
    num_rows: usize,
    row_width: usize,
) -> Result<BSeedMerkleTreeCtx, String> {
    let data_len = num_rows
        .checked_mul(row_width)
        .ok_or_else(|| "num_rows * row_width overflowed".to_string())?;
    let expected_leaves = padded_chunk_len(data_len) / blake3::CHUNK_LEN;
    if expected_leaves == 0 {
        return Err("expected at least one leaf".into());
    }
    if leaf_cvs.len() != expected_leaves {
        return Err(format!(
            "leaf_cvs count mismatch: got {}, expected {} for data_len={} row_width={}",
            leaf_cvs.len(), expected_leaves, data_len, row_width
        ));
    }

    let single_leaf = if expected_leaves == 1 {
        Some(expand_bseed_leaf(&bseed, data_len, 0))
    } else {
        None
    };
    let (layers, root) = build_merkle_layers_from_leaf_cvs(
        leaf_cvs,
        key,
        single_leaf.as_ref(),
    )?;

    Ok(BSeedMerkleTreeCtx {
        bseed,
        layers,
        root,
        num_rows,
        row_width,
        data_len,
    })
}

fn build_merkle_layers_from_leaf_cvs(
    leaf_cvs: &[[u8; 32]],
    key: [u8; 32],
    single_leaf_data: Option<&[u8; blake3::CHUNK_LEN]>,
) -> Result<(Vec<Vec<[u8; 32]>>, [u8; 32]), String> {
    if leaf_cvs.is_empty() {
        return Err("leaf_cvs must be non-empty".into());
    }

    let hasher = Blake3Hasher::with_key(key);
    let mut layers = vec![leaf_cvs.to_vec()];

    while layers.last().map(|l| l.len()).unwrap_or(0) > 2 {
        let prev = layers.last().unwrap();
        let next: Vec<[u8; 32]> = prev
            .chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    hasher.parent_cv(&pair[0], &pair[1])
                } else {
                    pair[0]
                }
            })
            .collect();
        layers.push(next);
    }

    let last = layers.last().unwrap();
    let root = match last.len() {
        1 => {
            let leaf = single_leaf_data.ok_or_else(||
                "single-leaf Merkle root requires the selected leaf data".to_string())?;
            hasher.hash(leaf)
        }
        2 => hasher.root_cv(&last[0], &last[1]),
        _ => return Err("invalid Merkle layer shape".into()),
    };
    if last.len() == 2 {
        layers.push(vec![root]);
    }

    Ok((layers, root))
}

fn merkle_siblings_from_layers(
    layers: &[Vec<[u8; 32]>],
    total_leaves: usize,
    leaf_indices: &[usize],
) -> Result<Vec<[u8; 32]>, String> {
    if leaf_indices.is_empty() {
        return Err("leaf_indices must be non-empty".into());
    }

    let mut siblings = Vec::new();
    let mut current_set: BTreeSet<usize> = leaf_indices.iter().copied().collect();
    let mut level_len = total_leaves;
    let mut level = 0usize;

    while level_len > 1 && !current_set.is_empty() {
        let level_nodes = layers
            .get(level)
            .ok_or_else(|| format!("missing merkle layer {}", level))?;
        for &idx in &current_set {
            if idx % 2 == 1 {
                if !current_set.contains(&(idx - 1)) {
                    siblings.push(level_nodes[idx - 1]);
                }
            } else if !current_set.contains(&(idx + 1)) && idx + 1 < level_len {
                siblings.push(level_nodes[idx + 1]);
            }
        }

        current_set = current_set.iter().map(|&idx| idx / 2).collect();
        level_len = level_len.div_ceil(2);
        level += 1;
    }

    Ok(siblings)
}

fn bseed_merkle_audit_paths(ctx: &BSeedMerkleTreeCtx, leaf_indices: &[usize]) -> Result<Vec<u8>, String> {
    let total_leaves = ctx.layers[0].len();
    if !(total_leaves.is_power_of_two() || total_leaves == 0) {
        return Err(format!(
            "total_leaves ({}) must be a power of two for audit_proof v1",
            total_leaves
        ));
    }
    if leaf_indices.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(&bad) = leaf_indices.iter().find(|&&i| i >= total_leaves.max(1)) {
        return Err(format!("leaf index {} out of bounds (total_leaves={})", bad, total_leaves));
    }

    let levels = if total_leaves <= 1 {
        0
    } else {
        total_leaves.trailing_zeros() as usize
    };
    let mut out = vec![0u8; leaf_indices.len() * levels * 32];

    for (k, &start_idx) in leaf_indices.iter().enumerate() {
        let mut idx = start_idx;
        let block_base = k * levels * 32;
        for level in 0..levels {
            let sibling = ctx.layers[level][idx ^ 1];
            let dst_off = block_base + level * 32;
            out[dst_off..dst_off + 32].copy_from_slice(&sibling);
            idx >>= 1;
        }
    }

    Ok(out)
}

/// Build the keyed-BLAKE3 Merkle tree over `data` once and return an opaque
/// handle that can be reused across many `pearl_capi_merkle_proof_for_handle`
/// calls. Also writes the 32-byte root and the total leaf count.
///
/// On success: writes `*out_handle`, `*out_root` (32 B), `*out_total_leaves`.
/// On error: sets `*out_handle = NULL` and writes a message via `err_msg_ptr`.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_build_tree(
    data_ptr: *const u8,
    data_len: usize,
    key_ptr: *const u8,
    row_width: usize,
    out_handle: *mut *mut std::ffi::c_void,
    out_root: *mut u8,
    out_total_leaves: *mut u32,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_handle.is_null() { *out_handle = std::ptr::null_mut(); }
    if !out_total_leaves.is_null() { *out_total_leaves = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if key_ptr.is_null() || out_handle.is_null() || out_root.is_null() || out_total_leaves.is_null() {
            return Err("required out-parameter pointer was null".into());
        }
        if data_len > 0 && data_ptr.is_null() {
            return Err("data_ptr null with non-zero data_len".into());
        }
        if row_width == 0 {
            return Err("row_width must be > 0".into());
        }

        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);

        let data: &[u8] = if data_len == 0 { &[] } else {
            std::slice::from_raw_parts(data_ptr, data_len)
        };
        let num_rows = if row_width == 0 { 0 } else { data.len() / row_width };

        let padded = pad_to_chunk_boundary(data);
        let tree = MerkleTree::new(&padded, key);

        std::ptr::copy_nonoverlapping(tree.root().as_ptr(), out_root, 32);
        *out_total_leaves = tree.num_leaves() as u32;

        let ctx = Box::new(MerkleTreeCtx { tree, num_rows, row_width });
        *out_handle = Box::into_raw(ctx) as *mut std::ffi::c_void;
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_build_tree"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Build a BSeed-backed Merkle handle from precomputed 1024-byte leaf CVs.
///
/// `leaf_cvs_ptr` must point to `total_leaves * 32` bytes produced by the
/// native/GPU tensor-hash leaf stage under `key_ptr`. The handle retains only
/// Merkle CV layers plus BSeed; proof leaf bytes are regenerated on demand.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_merkle_build_tree_from_leaf_cvs(
    leaf_cvs_ptr: *const u8,
    leaf_cvs_len: usize,
    bseed_ptr: *const u8,
    key_ptr: *const u8,
    num_rows: usize,
    row_width: usize,
    out_handle: *mut *mut std::ffi::c_void,
    out_root: *mut u8,
    out_total_leaves: *mut u32,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_handle.is_null() { *out_handle = std::ptr::null_mut(); }
    if !out_total_leaves.is_null() { *out_total_leaves = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if leaf_cvs_ptr.is_null() || bseed_ptr.is_null() || key_ptr.is_null()
            || out_handle.is_null() || out_root.is_null() || out_total_leaves.is_null()
        {
            return Err("required pointer was null".into());
        }
        if leaf_cvs_len == 0 || leaf_cvs_len % 32 != 0 {
            return Err("leaf_cvs_len must be a non-zero multiple of 32".into());
        }
        if num_rows == 0 || row_width == 0 {
            return Err("num_rows and row_width must be > 0".into());
        }

        let leaf_bytes = std::slice::from_raw_parts(leaf_cvs_ptr, leaf_cvs_len);
        let mut leaf_cvs = Vec::<[u8; 32]>::with_capacity(leaf_cvs_len / 32);
        for chunk in leaf_bytes.chunks_exact(32) {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(chunk);
            leaf_cvs.push(digest);
        }

        let mut bseed = [0u8; 32];
        std::ptr::copy_nonoverlapping(bseed_ptr, bseed.as_mut_ptr(), 32);
        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);

        let ctx = build_bseed_merkle_ctx(&leaf_cvs, bseed, key, num_rows, row_width)?;

        std::ptr::copy_nonoverlapping(ctx.root.as_ptr(), out_root, 32);
        *out_total_leaves = ctx.layers[0].len() as u32;
        *out_handle = Box::into_raw(Box::new(ctx)) as *mut std::ffi::c_void;
        Ok(())
    }));

    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_merkle_build_tree_from_leaf_cvs"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Build a Merkle inclusion proof from precomputed leaf CVs plus the selected
/// 1024-byte leaf chunks.
///
/// This is the A-side fast path for miners: GPU tensor_hash produces the
/// keyed BLAKE3 leaf CVs for the full A matrix, while the miner copies back
/// only the opened A chunks. Rust reconstructs the CV tree, emits the same
/// proof shape as `pearl_capi_merkle_root_and_proof`, and never needs the full
/// matrix bytes.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_proof_from_leaf_cvs(
    leaf_cvs_ptr: *const u8,
    leaf_cvs_len: usize,
    leaf_data_ptr: *const u8,
    leaf_data_len: usize,
    key_ptr: *const u8,
    row_indices_ptr: *const u32,
    row_indices_len: usize,
    num_rows: usize,
    row_width: usize,
    out_root: *mut u8,
    out_total_leaves: *mut u32,
    out_leaf_indices: *mut *mut u32,
    out_leaf_indices_len: *mut usize,
    out_siblings: *mut *mut u8,
    out_sibling_count: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_leaf_indices.is_null() { *out_leaf_indices = std::ptr::null_mut(); }
    if !out_leaf_indices_len.is_null() { *out_leaf_indices_len = 0; }
    if !out_siblings.is_null() { *out_siblings = std::ptr::null_mut(); }
    if !out_sibling_count.is_null() { *out_sibling_count = 0; }
    if !out_total_leaves.is_null() { *out_total_leaves = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if leaf_cvs_ptr.is_null() || leaf_data_ptr.is_null() || key_ptr.is_null()
            || row_indices_ptr.is_null() || out_root.is_null()
            || out_total_leaves.is_null() || out_leaf_indices.is_null()
            || out_leaf_indices_len.is_null() || out_siblings.is_null()
            || out_sibling_count.is_null()
        {
            return Err("required pointer was null".into());
        }
        if leaf_cvs_len == 0 || leaf_cvs_len % 32 != 0 {
            return Err("leaf_cvs_len must be a non-zero multiple of 32".into());
        }
        if leaf_data_len == 0 || leaf_data_len % blake3::CHUNK_LEN != 0 {
            return Err("leaf_data_len must be a non-zero multiple of 1024".into());
        }
        if row_indices_len == 0 {
            return Err("row_indices_len must be > 0".into());
        }
        if num_rows == 0 || row_width == 0 {
            return Err("num_rows and row_width must be > 0".into());
        }

        let data_len = num_rows
            .checked_mul(row_width)
            .ok_or_else(|| "num_rows * row_width overflowed".to_string())?;
        let expected_leaves = padded_chunk_len(data_len) / blake3::CHUNK_LEN;
        if expected_leaves == 0 {
            return Err("expected at least one leaf".into());
        }
        if leaf_cvs_len / 32 != expected_leaves {
            return Err(format!(
                "leaf_cvs count mismatch: got {}, expected {} for data_len={} row_width={}",
                leaf_cvs_len / 32, expected_leaves, data_len, row_width
            ));
        }

        let leaf_bytes = std::slice::from_raw_parts(leaf_cvs_ptr, leaf_cvs_len);
        let mut leaf_cvs = Vec::<[u8; 32]>::with_capacity(expected_leaves);
        for chunk in leaf_bytes.chunks_exact(32) {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(chunk);
            leaf_cvs.push(digest);
        }

        let leaf_data = std::slice::from_raw_parts(leaf_data_ptr, leaf_data_len);

        let mut key = [0u8; 32];
        std::ptr::copy_nonoverlapping(key_ptr, key.as_mut_ptr(), 32);

        let row_indices: Vec<usize> = std::slice::from_raw_parts(row_indices_ptr, row_indices_len)
            .iter()
            .map(|&r| r as usize)
            .collect();
        if let Some(&bad) = row_indices.iter().find(|&&r| r >= num_rows) {
            return Err(format!(
                "row index {} out of bounds (num_rows={}, row_width={})",
                bad, num_rows, row_width
            ));
        }

        let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(
            &row_indices,
            (num_rows, row_width),
        );
        if leaf_data_len != leaf_indices.len() * blake3::CHUNK_LEN {
            return Err(format!(
                "leaf_data_len mismatch: got {}, expected {} for {} opened leaves",
                leaf_data_len,
                leaf_indices.len() * blake3::CHUNK_LEN,
                leaf_indices.len()
            ));
        }
        if let Some(&bad) = leaf_indices.iter().find(|&&i| i >= expected_leaves) {
            return Err(format!("leaf index {} out of bounds (total_leaves={})", bad, expected_leaves));
        }

        let hasher = Blake3Hasher::with_key(key);
        for (i, &leaf_idx) in leaf_indices.iter().enumerate() {
            let chunk = &leaf_data[i * blake3::CHUNK_LEN..(i + 1) * blake3::CHUNK_LEN];
            let cv = hasher.chunk_cv(chunk, leaf_idx as u64);
            if cv != leaf_cvs[leaf_idx] {
                return Err(format!("leaf data/CV mismatch at opened leaf {}", leaf_idx));
            }
        }

        let single_leaf = if expected_leaves == 1 {
            let mut leaf = [0u8; blake3::CHUNK_LEN];
            leaf.copy_from_slice(&leaf_data[..blake3::CHUNK_LEN]);
            Some(leaf)
        } else {
            None
        };
        let (layers, root) = build_merkle_layers_from_leaf_cvs(
            &leaf_cvs,
            key,
            single_leaf.as_ref(),
        )?;

        std::ptr::copy_nonoverlapping(root.as_ptr(), out_root, 32);
        *out_total_leaves = expected_leaves as u32;

        {
            let mut idx_buf: Vec<u32> = leaf_indices.iter().map(|&i| i as u32).collect();
            idx_buf.shrink_to_fit();
            let len = idx_buf.len();
            let ptr = idx_buf.as_mut_ptr();
            std::mem::forget(idx_buf);
            *out_leaf_indices_len = len;
            *out_leaf_indices = ptr;
        }

        let siblings = merkle_siblings_from_layers(&layers, expected_leaves, &leaf_indices)?;
        let sibling_count = siblings.len();
        if sibling_count > 0 {
            let mut buf = vec![0u8; sibling_count * 32];
            for (i, sib) in siblings.iter().enumerate() {
                buf[i * 32..(i + 1) * 32].copy_from_slice(sib);
            }
            let boxed = buf.into_boxed_slice();
            *out_sibling_count = sibling_count;
            *out_siblings = Box::into_raw(boxed) as *mut u8;
        }

        Ok(())
    }));

    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_proof_from_leaf_cvs"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Extract an inclusion proof against a previously built Merkle tree.
/// Output buffer ownership mirrors `pearl_capi_merkle_root_and_proof`:
/// `leaf_data` and `siblings` via `pearl_capi_free_buffer`, `leaf_indices`
/// via `pearl_capi_free_u32_buffer`.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_proof_for_handle(
    handle: *mut std::ffi::c_void,
    row_indices_ptr: *const u32,
    row_indices_len: usize,
    out_leaf_data: *mut *mut u8,
    out_leaf_count: *mut usize,
    out_leaf_indices: *mut *mut u32,
    out_leaf_indices_len: *mut usize,
    out_siblings: *mut *mut u8,
    out_sibling_count: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_leaf_data.is_null()       { *out_leaf_data       = std::ptr::null_mut(); }
    if !out_leaf_count.is_null()      { *out_leaf_count      = 0; }
    if !out_leaf_indices.is_null()    { *out_leaf_indices    = std::ptr::null_mut(); }
    if !out_leaf_indices_len.is_null(){ *out_leaf_indices_len= 0; }
    if !out_siblings.is_null()        { *out_siblings        = std::ptr::null_mut(); }
    if !out_sibling_count.is_null()   { *out_sibling_count   = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null()
            || out_leaf_data.is_null() || out_leaf_count.is_null()
            || out_leaf_indices.is_null() || out_leaf_indices_len.is_null()
            || out_siblings.is_null() || out_sibling_count.is_null()
        {
            return Err("required out-parameter pointer was null".into());
        }
        if row_indices_len == 0 {
            return Err("row_indices_len must be > 0".into());
        }
        if row_indices_ptr.is_null() {
            return Err("row_indices_ptr null with non-zero row_indices_len".into());
        }

        let ctx = &*(handle as *const MerkleTreeCtx);

        let row_indices: Vec<usize> = std::slice::from_raw_parts(row_indices_ptr, row_indices_len)
            .iter()
            .map(|&r| r as usize)
            .collect();

        if let Some(&bad) = row_indices.iter().find(|&&r| r >= ctx.num_rows.max(1)) {
            return Err(format!(
                "row index {} out of bounds (num_rows={}, row_width={})",
                bad, ctx.num_rows, ctx.row_width
            ));
        }

        let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(
            &row_indices,
            (ctx.num_rows.max(1), ctx.row_width),
        );
        let proof = ctx.tree.get_multileaf_proof(&leaf_indices);

        // ---- Pack leaf_data: leaf_count × 1024 contiguous bytes ----
        let leaf_count = proof.leaf_data.len();
        if leaf_count > 0 {
            let mut buf = vec![0u8; leaf_count * 1024];
            for (i, chunk) in proof.leaf_data.iter().enumerate() {
                buf[i * 1024..(i + 1) * 1024].copy_from_slice(chunk);
            }
            let boxed = buf.into_boxed_slice();
            *out_leaf_count = leaf_count;
            *out_leaf_data  = Box::into_raw(boxed) as *mut u8;
        }

        // ---- Pack leaf_indices: leaf_count × u32 ----
        {
            let mut idx_buf: Vec<u32> = proof.leaf_indices.iter().map(|&i| i as u32).collect();
            idx_buf.shrink_to_fit();
            let len = idx_buf.len();
            let ptr = idx_buf.as_mut_ptr();
            std::mem::forget(idx_buf);
            *out_leaf_indices_len = len;
            *out_leaf_indices     = ptr;
        }

        // ---- Pack siblings: sibling_count × 32 bytes ----
        let sibling_count = proof.siblings.len();
        if sibling_count > 0 {
            let mut buf = vec![0u8; sibling_count * 32];
            for (i, sib) in proof.siblings.iter().enumerate() {
                buf[i * 32..(i + 1) * 32].copy_from_slice(sib);
            }
            let boxed = buf.into_boxed_slice();
            *out_sibling_count = sibling_count;
            *out_siblings      = Box::into_raw(boxed) as *mut u8;
        }

        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_proof_for_handle"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Extract an inclusion proof against a BSeed-backed CV-layer handle.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_merkle_proof_for_handle(
    handle: *mut std::ffi::c_void,
    row_indices_ptr: *const u32,
    row_indices_len: usize,
    out_leaf_data: *mut *mut u8,
    out_leaf_count: *mut usize,
    out_leaf_indices: *mut *mut u32,
    out_leaf_indices_len: *mut usize,
    out_siblings: *mut *mut u8,
    out_sibling_count: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_leaf_data.is_null()       { *out_leaf_data       = std::ptr::null_mut(); }
    if !out_leaf_count.is_null()      { *out_leaf_count      = 0; }
    if !out_leaf_indices.is_null()    { *out_leaf_indices    = std::ptr::null_mut(); }
    if !out_leaf_indices_len.is_null(){ *out_leaf_indices_len= 0; }
    if !out_siblings.is_null()        { *out_siblings        = std::ptr::null_mut(); }
    if !out_sibling_count.is_null()   { *out_sibling_count   = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null()
            || out_leaf_data.is_null() || out_leaf_count.is_null()
            || out_leaf_indices.is_null() || out_leaf_indices_len.is_null()
            || out_siblings.is_null() || out_sibling_count.is_null()
        {
            return Err("required out-parameter pointer was null".into());
        }
        if row_indices_len == 0 {
            return Err("row_indices_len must be > 0".into());
        }
        if row_indices_ptr.is_null() {
            return Err("row_indices_ptr null with non-zero row_indices_len".into());
        }

        let ctx = &*(handle as *const BSeedMerkleTreeCtx);
        let row_indices: Vec<usize> = std::slice::from_raw_parts(row_indices_ptr, row_indices_len)
            .iter()
            .map(|&r| r as usize)
            .collect();

        if let Some(&bad) = row_indices.iter().find(|&&r| r >= ctx.num_rows.max(1)) {
            return Err(format!(
                "row index {} out of bounds (num_rows={}, row_width={})",
                bad, ctx.num_rows, ctx.row_width
            ));
        }

        let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(
            &row_indices,
            (ctx.num_rows.max(1), ctx.row_width),
        );
        if let Some(&bad) = leaf_indices.iter().find(|&&i| i >= ctx.layers[0].len()) {
            return Err(format!("leaf index {} out of bounds (total_leaves={})", bad, ctx.layers[0].len()));
        }

        let leaf_count = leaf_indices.len();
        if leaf_count > 0 {
            let mut buf = vec![0u8; leaf_count * 1024];
            for (i, &leaf_idx) in leaf_indices.iter().enumerate() {
                let leaf = expand_bseed_leaf(&ctx.bseed, ctx.data_len, leaf_idx);
                buf[i * 1024..(i + 1) * 1024].copy_from_slice(&leaf);
            }
            let boxed = buf.into_boxed_slice();
            *out_leaf_count = leaf_count;
            *out_leaf_data = Box::into_raw(boxed) as *mut u8;
        }

        {
            let mut idx_buf: Vec<u32> = leaf_indices.iter().map(|&i| i as u32).collect();
            idx_buf.shrink_to_fit();
            let len = idx_buf.len();
            let ptr = idx_buf.as_mut_ptr();
            std::mem::forget(idx_buf);
            *out_leaf_indices_len = len;
            *out_leaf_indices = ptr;
        }

        let siblings = merkle_siblings_from_layers(&ctx.layers, ctx.layers[0].len(), &leaf_indices)?;
        let sibling_count = siblings.len();
        if sibling_count > 0 {
            let mut buf = vec![0u8; sibling_count * 32];
            for (i, sib) in siblings.iter().enumerate() {
                buf[i * 32..(i + 1) * 32].copy_from_slice(sib);
            }
            let boxed = buf.into_boxed_slice();
            *out_sibling_count = sibling_count;
            *out_siblings = Box::into_raw(boxed) as *mut u8;
        }

        Ok(())
    }));

    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_merkle_proof_for_handle"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Open K independent, leaf-major Merkle audit paths against a previously
/// built handle. Output is a single byte buffer of `leaf_indices_len ×
/// levels × 32` bytes, where `levels = log2(total_leaves)`; openings are
/// concatenated in caller-supplied order and within each opening the
/// siblings are emitted leaf→root.
///
/// **v1 spec:** `total_leaves` must be a power of two. Non-pow2 trees are
/// rejected with `PEARL_CAPI_ERR_BAD_ARG`.
///
/// Caller MUST free `*out_siblings` with `pearl_capi_free_buffer` using
/// `*out_sibling_bytes` as the length.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_audit_paths_for_handle(
    handle: *mut std::ffi::c_void,
    leaf_indices_ptr: *const u32,
    leaf_indices_len: usize,
    out_siblings: *mut *mut u8,
    out_sibling_bytes: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_siblings.is_null() { *out_siblings = std::ptr::null_mut(); }
    if !out_sibling_bytes.is_null() { *out_sibling_bytes = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null() || out_siblings.is_null() || out_sibling_bytes.is_null() {
            return Err("required out-parameter pointer was null".into());
        }
        if leaf_indices_len > 0 && leaf_indices_ptr.is_null() {
            return Err("leaf_indices_ptr null with non-zero leaf_indices_len".into());
        }

        let ctx = &*(handle as *const MerkleTreeCtx);

        let leaf_indices: Vec<usize> = if leaf_indices_len == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(leaf_indices_ptr, leaf_indices_len)
                .iter()
                .map(|&i| i as usize)
                .collect()
        };

        let buf = ctx.tree.get_audit_paths(&leaf_indices).map_err(|e| e.to_string())?;
        let len = buf.len();
        if len > 0 {
            let boxed = buf.into_boxed_slice();
            *out_sibling_bytes = len;
            *out_siblings = Box::into_raw(boxed) as *mut u8;
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during merkle_audit_paths_for_handle"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Open audit paths against a BSeed-backed CV-layer handle.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_merkle_audit_paths_for_handle(
    handle: *mut std::ffi::c_void,
    leaf_indices_ptr: *const u32,
    leaf_indices_len: usize,
    out_siblings: *mut *mut u8,
    out_sibling_bytes: *mut usize,
    err_msg_ptr: *mut *mut c_char,
) -> c_int {
    if !out_siblings.is_null() { *out_siblings = std::ptr::null_mut(); }
    if !out_sibling_bytes.is_null() { *out_sibling_bytes = 0; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null() || out_siblings.is_null() || out_sibling_bytes.is_null() {
            return Err("required out-parameter pointer was null".into());
        }
        if leaf_indices_len > 0 && leaf_indices_ptr.is_null() {
            return Err("leaf_indices_ptr null with non-zero leaf_indices_len".into());
        }

        let ctx = &*(handle as *const BSeedMerkleTreeCtx);
        let leaf_indices: Vec<usize> = if leaf_indices_len == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(leaf_indices_ptr, leaf_indices_len)
                .iter()
                .map(|&i| i as usize)
                .collect()
        };

        let buf = bseed_merkle_audit_paths(ctx, &leaf_indices)?;
        let len = buf.len();
        if len > 0 {
            let boxed = buf.into_boxed_slice();
            *out_sibling_bytes = len;
            *out_siblings = Box::into_raw(boxed) as *mut u8;
        }
        Ok(())
    }));

    match result {
        Ok(Ok(())) => PEARL_CAPI_OK,
        Ok(Err(e)) => { set_err(err_msg_ptr, e); PEARL_CAPI_ERR_BAD_ARG }
        Err(_)     => { set_err(err_msg_ptr, "panic during bseed_merkle_audit_paths_for_handle"); PEARL_CAPI_ERR_INTERNAL }
    }
}

/// Free a BSeed-backed Merkle tree handle previously returned by
/// `pearl_capi_bseed_merkle_build_tree_from_leaf_cvs`. Safe to call with NULL.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_bseed_merkle_tree_free(handle: *mut std::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let _ = Box::from_raw(handle as *mut BSeedMerkleTreeCtx);
}

/// Free a Merkle tree handle previously returned by
/// `pearl_capi_merkle_build_tree`. Safe to call with `NULL`.
#[no_mangle]
pub unsafe extern "C" fn pearl_capi_merkle_tree_free(handle: *mut std::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let _ = Box::from_raw(handle as *mut MerkleTreeCtx);
}
