// Akoya.Mining — P/Invoke surface for libpearl_mining_capi.so.
//
// The library is built from py-pearl-mining with the `capi` cargo
// feature:
//
//   cd py-pearl-mining
//   cargo build --release --no-default-features --features capi
//   # → target/release/libpearl_mining.so (we typically rename to
//   #   libpearl_mining_capi.so to disambiguate from the Python build)
//
// Caller sets AKOYA_PEARL_MINING_LIB to the absolute path of the .so;
// if unset we fall back to "libpearl_mining_capi.so" on PATH.
//
// All entry points return int (0 = success, non-zero = error). On error
// the optional `err_msg` out-pointer is set to a NUL-terminated string
// owned by the library; caller MUST free with FreeString.

using System.Runtime.InteropServices;

namespace Akoya.Mining;

public static unsafe partial class PearlMiningNative
{
    public const string Lib = "pearl_mining_capi";

    [LibraryImport(Lib, EntryPoint = "pearl_capi_version")]
    public static partial uint Version();

    // The V1 plain-proof pack / verify / diagnose entry points were removed
    // in V2: the pool now constructs the block proof from the typed
    // ShareSubmission, so the miner no longer packs or self-verifies one.
    // Buffer/string free helpers remain because MerkleRootAndProof still
    // uses them.

    [LibraryImport(Lib, EntryPoint = "pearl_capi_free_buffer")]
    public static partial void FreeBuffer(byte* ptr, nuint len);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_free_string")]
    public static partial void FreeString(byte* ptr);

    /// <summary>
    /// Diagnostic: BLAKE3-keyed hash of an arbitrary byte buffer. Mirrors
    /// `pearl-blake3::blake3_digest(data, Some(key))`. Used to recompute,
    /// host-side, the hash that an on-device `tensor_hash` SHOULD produce.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_blake3_keyed")]
    public static partial int Blake3Keyed(
        byte* dataPtr, nuint dataLen,
        byte* keyPtr,             // 32
        byte* outDigest,          // 32
        byte** errMsgPtr);

    /// <summary>
    /// Compute the BLAKE3 keyed Merkle root using the canonical Rust implementation.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_root")]
    public static partial int MerkleRoot(
        byte* dataPtr, nuint dataLen,
        byte* keyPtr,             // 32
        byte* outRoot,            // 32
        byte** errMsgPtr);

    /// <summary>
    /// Fused BSeed XOF expansion, int7 mapping, and keyed-BLAKE3 Merkle root.
    /// Used by pool-side share verification for the HashB derivation path.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_and_merkle")]
    public static partial int BSeedExpandAndMerkle(
        byte* bseedPtr,            // 32
        nuint n,
        nuint k,
        byte* keyPtr,              // 32
        byte* outRoot,             // 32
        byte** errMsgPtr);

    /// <summary>
    /// BSeed XOF expansion and int7 mapping into a caller-owned row-major B buffer.
    /// Does not pad or compute a Merkle root. Intended for miner GPU upload.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_raw")]
    public static partial int BSeedExpandRaw(
        byte* bseedPtr,            // 32
        nuint n,
        nuint k,
        byte* outBPtr,             // n*k bytes
        nuint outBLen,
        byte** errMsgPtr);

    /// <summary>
    /// BSeed XOF range expansion and int7 mapping into a caller-owned buffer.
    /// Equivalent to slicing the raw full expansion at <paramref name="byteOffset"/>,
    /// without materialising skipped bytes.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_range_raw")]
    public static partial int BSeedExpandRangeRaw(
        byte* bseedPtr,            // 32
        ulong byteOffset,
        byte* outBPtr,
        nuint outBLen,
        byte** errMsgPtr);

    /// <summary>
    /// Verify a multi-leaf Merkle proof using the canonical Rust implementation.
    /// Returns 0 if the proof is valid, non-zero with error message on failure.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_verify_proof")]
    public static partial int MerkleVerifyProof(
        byte* leafDataPtr,        // leaf_count * 1024 bytes contiguous
        uint* leafIndicesPtr,     // leaf_count u32 values
        nuint leafCount,
        nuint totalLeaves,
        byte* siblingsPtr,        // sibling_count * 32 bytes contiguous
        nuint siblingCount,
        byte* keyPtr,             // 32
        byte* expectedRootPtr,    // 32
        byte** errMsgPtr);

    /// <summary>
    /// Fused replacement for <c>Blake3.MerkleRoot</c> + <c>MerkleProofBuilder.BuildProof</c>:
    /// builds the keyed BLAKE3 Merkle tree once and emits both the root and a
    /// multi-leaf inclusion proof for the requested rows. Free
    /// <paramref name="outLeafData"/> and <paramref name="outSiblings"/> with
    /// <see cref="FreeBuffer"/>; free <paramref name="outLeafIndices"/> with
    /// <see cref="FreeU32Buffer"/>.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_root_and_proof")]
    public static partial int MerkleRootAndProof(
        byte* dataPtr, nuint dataLen,
        byte* keyPtr,                  // 32
        uint* rowIndicesPtr, nuint rowIndicesLen,
        nuint rowWidth,
        byte* outRoot,                 // caller-alloc 32
        uint* outTotalLeaves,
        byte** outLeafData, nuint* outLeafCount,
        uint** outLeafIndices, nuint* outLeafIndicesLen,
        byte** outSiblings, nuint* outSiblingCount,
        byte** errMsgPtr);

    /// <summary>
    /// Build an inclusion proof from precomputed full-matrix leaf CVs and
    /// caller-owned selected 1024-byte leaf data. Used by the miner's A-side
    /// share path to avoid copying and hashing the full A matrix on CPU.
    /// Free <paramref name="outLeafIndices"/> with <see cref="FreeU32Buffer"/>
    /// and <paramref name="outSiblings"/> with <see cref="FreeBuffer"/>.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_proof_from_leaf_cvs")]
    public static partial int MerkleProofFromLeafCvs(
        byte* leafCvsPtr, nuint leafCvsLen,
        byte* leafDataPtr, nuint leafDataLen,
        byte* keyPtr,                  // 32
        uint* rowIndicesPtr, nuint rowIndicesLen,
        nuint numRows,
        nuint rowWidth,
        byte* outRoot,                 // caller-alloc 32
        uint* outTotalLeaves,
        uint** outLeafIndices, nuint* outLeafIndicesLen,
        byte** outSiblings, nuint* outSiblingCount,
        byte** errMsgPtr);

    /// <summary>
    /// Free a <c>uint*</c> buffer allocated by
    /// <see cref="MerkleRootAndProof"/>. <paramref name="len"/> is the number
    /// of u32 elements (not bytes).
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_free_u32_buffer")]
    public static partial void FreeU32Buffer(uint* ptr, nuint len);

    /// <summary>
    /// Build the keyed-BLAKE3 Merkle tree over <paramref name="dataPtr"/>
    /// once and return an opaque handle that can be reused across many
    /// <see cref="MerkleProofForHandle"/> calls. Used to amortize the
    /// O(data_len) BLAKE3 cost when the tree is constant per-σ (B matrix).
    /// Caller MUST free the handle with <see cref="MerkleTreeFree"/>.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_build_tree")]
    public static partial int MerkleBuildTree(
        byte* dataPtr, nuint dataLen,
        byte* keyPtr,                  // 32
        nuint rowWidth,
        void** outHandle,
        byte* outRoot,                 // caller-alloc 32
        uint* outTotalLeaves,
        byte** errMsgPtr);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_merkle_build_tree_from_leaf_cvs")]
    public static partial int BSeedMerkleBuildTreeFromLeafCvs(
        byte* leafCvsPtr, nuint leafCvsLen,
        byte* bseedPtr,                 // 32
        byte* keyPtr,                   // 32
        nuint numRows,
        nuint rowWidth,
        void** outHandle,
        byte* outRoot,                  // caller-alloc 32
        uint* outTotalLeaves,
        byte** errMsgPtr);

    /// <summary>
    /// Extract an inclusion proof against a tree previously built with
    /// <see cref="MerkleBuildTree"/>. Does NOT free the handle. Output
    /// buffer ownership mirrors <see cref="MerkleRootAndProof"/>.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_proof_for_handle")]
    public static partial int MerkleProofForHandle(
        void* handle,
        uint* rowIndicesPtr, nuint rowIndicesLen,
        byte** outLeafData, nuint* outLeafCount,
        uint** outLeafIndices, nuint* outLeafIndicesLen,
        byte** outSiblings, nuint* outSiblingCount,
        byte** errMsgPtr);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_merkle_proof_for_handle")]
    public static partial int BSeedMerkleProofForHandle(
        void* handle,
        uint* rowIndicesPtr, nuint rowIndicesLen,
        byte** outLeafData, nuint* outLeafCount,
        uint** outLeafIndices, nuint* outLeafIndicesLen,
        byte** outSiblings, nuint* outSiblingCount,
        byte** errMsgPtr);

    /// <summary>
    /// Open K independent audit paths against a previously built handle.
    /// Output buffer is a single block of <c>leafIndicesLen × levels × 32</c>
    /// bytes (leaf→root within each opening, openings in caller order).
    /// Caller MUST free <paramref name="outSiblings"/> via
    /// <see cref="FreeBuffer"/> using <paramref name="outSiblingBytes"/> as
    /// the length.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_audit_paths_for_handle")]
    public static partial int MerkleAuditPathsForHandle(
        void* handle,
        uint* leafIndicesPtr, nuint leafIndicesLen,
        byte** outSiblings, nuint* outSiblingBytes,
        byte** errMsgPtr);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_merkle_audit_paths_for_handle")]
    public static partial int BSeedMerkleAuditPathsForHandle(
        void* handle,
        uint* leafIndicesPtr, nuint leafIndicesLen,
        byte** outSiblings, nuint* outSiblingBytes,
        byte** errMsgPtr);

    /// <summary>
    /// Free a Merkle tree handle previously returned by
    /// <see cref="MerkleBuildTree"/>. Safe to call with <see cref="IntPtr.Zero"/>.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "pearl_capi_merkle_tree_free")]
    public static partial void MerkleTreeFree(void* handle);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_merkle_tree_free")]
    public static partial void BSeedMerkleTreeFree(void* handle);
}
