// BLAKE3 implementation — single-chunk hash, XOF (extended output),
// keyed hash, and Merkle tree construction.
//
// The core compress() function is spec-faithful per the BLAKE3 reference.
// Original scope was ≤1024-byte inputs for commitment-key derivation;
// extended for BSeedExpander (XOF) and HashB (keyed Merkle tree).
//
// Reference: https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs

using System.Buffers;
using System.Buffers.Binary;

namespace Akoya.Crypto;

public static class Blake3
{
    public const int DigestSize = 32;
    public const int ChunkLen = 1024;
    private const int BlockLen = 64;

    private const uint FlagChunkStart = 1 << 0;
    private const uint FlagChunkEnd   = 1 << 1;
    private const uint FlagParent     = 1 << 2;
    private const uint FlagRoot       = 1 << 3;
    private const uint FlagKeyedHash  = 1 << 4;

    internal static readonly uint[] IV =
    {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
    };

    private static readonly int[] MsgPermutation =
    { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 };

    /// <summary>Hash arbitrary-length input ≤ 1024 bytes. Returns 32 bytes.</summary>
    public static byte[] Hash(ReadOnlySpan<byte> input)
    {
        if (input.Length > ChunkLen)
            throw new NotSupportedException("Blake3.Hash only supports inputs ≤ 1024 bytes. Use HashLarge for bigger inputs.");

        Span<uint> chainingValue = stackalloc uint[8];
        IV.AsSpan().CopyTo(chainingValue);
        Span<uint> blockWords = stackalloc uint[16];

        int totalBlocks = Math.Max(1, (input.Length + BlockLen - 1) / BlockLen);

        for (int blocksConsumed = 0; blocksConsumed < totalBlocks; blocksConsumed++)
        {
            int offset = blocksConsumed * BlockLen;
            int blockBytes = Math.Min(BlockLen, input.Length - offset);
            LoadBlock(input.Slice(offset, blockBytes), blockWords);

            uint flags = 0;
            if (blocksConsumed == 0) flags |= FlagChunkStart;
            bool isLast = blocksConsumed == totalBlocks - 1;
            if (isLast) flags |= FlagChunkEnd | FlagRoot;

            Span<uint> compressed = stackalloc uint[16];
            Compress(chainingValue, blockWords, 0, (uint)blockBytes, flags, compressed);

            if (isLast)
                return WordsToBytes(compressed[..8]);

            compressed[..8].CopyTo(chainingValue);
        }
        throw new InvalidOperationException("unreachable");
    }

    /// <summary>
    /// Random-access BLAKE3 XOF read at an arbitrary byte offset. Equivalent
    /// to <c>Xof(input, fullBuffer); fullBuffer[byteOffset..byteOffset+output.Length].CopyTo(output)</c>
    /// but skips all preceding XOF blocks — O(output.Length / 64) compressions,
    /// independent of <paramref name="byteOffset"/>.
    ///
    /// Used by AuditProofVerifier (and the audit_proof spec's pool-side leaf
    /// re-derivation) to read a single 1024-byte leaf at offset
    /// <c>leaf_idx * 1024</c> without materialising the full B matrix.
    /// </summary>
    public static void XofAt(ReadOnlySpan<byte> input, ulong byteOffset, Span<byte> output)
    {
        if (input.Length > ChunkLen)
            throw new NotSupportedException("Blake3.XofAt only supports inputs ≤ 1024 bytes.");

        Span<uint> chainingValue = stackalloc uint[8];
        IV.AsSpan().CopyTo(chainingValue);
        Span<uint> blockWords = stackalloc uint[16];

        int totalBlocks = Math.Max(1, (input.Length + BlockLen - 1) / BlockLen);
        int lastBlockOffset = (totalBlocks - 1) * BlockLen;
        int lastBlockLen = Math.Min(BlockLen, input.Length - lastBlockOffset);

        for (int i = 0; i < totalBlocks - 1; i++)
        {
            int offset = i * BlockLen;
            int blockBytes = Math.Min(BlockLen, input.Length - offset);
            LoadBlock(input.Slice(offset, blockBytes), blockWords);

            uint flags = 0;
            if (i == 0) flags |= FlagChunkStart;

            Span<uint> compressed = stackalloc uint[16];
            Compress(chainingValue, blockWords, 0, (uint)blockBytes, flags, compressed);
            compressed[..8].CopyTo(chainingValue);
        }

        LoadBlock(input.Slice(lastBlockOffset, lastBlockLen), blockWords);
        uint lastFlags = FlagChunkEnd | FlagRoot;
        if (totalBlocks == 1) lastFlags |= FlagChunkStart;

        // Each XOF compression with counter=c emits 64 bytes spanning
        // logical output range [c*64 .. (c+1)*64). Seek by skipping
        // (byteOffset / 64) compressions and discarding (byteOffset % 64)
        // bytes from the first compression we keep.
        ulong counter = byteOffset / 64;
        int skipInFirst = (int)(byteOffset % 64);

        Span<uint> xofOut = stackalloc uint[16];
        Span<byte> blockBuf = stackalloc byte[64];
        int written = 0;

        while (written < output.Length)
        {
            Compress(chainingValue, blockWords, counter, (uint)lastBlockLen, lastFlags, xofOut);

            for (int w = 0; w < 16; w++)
                BinaryPrimitives.WriteUInt32LittleEndian(blockBuf.Slice(w * 4, 4), xofOut[w]);

            int take = Math.Min(64 - skipInFirst, output.Length - written);
            blockBuf.Slice(skipInFirst, take).CopyTo(output.Slice(written, take));
            written += take;
            skipInFirst = 0;
            counter++;
        }
    }

    /// <summary>
    /// BLAKE3 XOF (extended output function). Hashes <paramref name="input"/>
    /// then produces <paramref name="outputLength"/> bytes of deterministic output.
    /// Used by BSeedExpander to expand a 32-byte seed into n×k bytes.
    /// </summary>
    public static void Xof(ReadOnlySpan<byte> input, Span<byte> output)
    {
        if (input.Length > ChunkLen)
            throw new NotSupportedException("Blake3.Xof only supports inputs ≤ 1024 bytes.");

        Span<uint> chainingValue = stackalloc uint[8];
        IV.AsSpan().CopyTo(chainingValue);
        Span<uint> blockWords = stackalloc uint[16];

        int totalBlocks = Math.Max(1, (input.Length + BlockLen - 1) / BlockLen);
        int lastBlockOffset = (totalBlocks - 1) * BlockLen;
        int lastBlockLen = Math.Min(BlockLen, input.Length - lastBlockOffset);

        // Process non-last blocks to build chaining value
        for (int i = 0; i < totalBlocks - 1; i++)
        {
            int offset = i * BlockLen;
            int blockBytes = Math.Min(BlockLen, input.Length - offset);
            LoadBlock(input.Slice(offset, blockBytes), blockWords);

            uint flags = 0;
            if (i == 0) flags |= FlagChunkStart;

            Span<uint> compressed = stackalloc uint[16];
            Compress(chainingValue, blockWords, 0, (uint)blockBytes, flags, compressed);
            compressed[..8].CopyTo(chainingValue);
        }

        // Prepare the last block (ROOT flagged) — we'll re-compress with varying counters
        LoadBlock(input.Slice(lastBlockOffset, lastBlockLen), blockWords);
        uint lastFlags = FlagChunkEnd | FlagRoot;
        if (totalBlocks == 1) lastFlags |= FlagChunkStart;

        // XOF: re-run the last compression with incrementing counter to produce 64 bytes each
        Span<uint> xofOut = stackalloc uint[16];
        Span<byte> wordBuf = stackalloc byte[4];
        int written = 0;
        ulong counter = 0;

        while (written < output.Length)
        {
            Compress(chainingValue, blockWords, counter, (uint)lastBlockLen, lastFlags, xofOut);

            // Each counter produces 64 bytes (16 × u32)
            int remaining = output.Length - written;
            int toCopy = Math.Min(64, remaining);
            for (int w = 0; w < (toCopy + 3) / 4; w++)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(wordBuf, xofOut[w]);
                int bytesFromWord = Math.Min(4, remaining - w * 4);
                if (bytesFromWord > 0)
                    wordBuf[..bytesFromWord].CopyTo(output.Slice(written + w * 4, bytesFromWord));
            }

            written += toCopy;
            counter++;
        }
    }

    /// <summary>
    /// Keyed BLAKE3 hash for data ≤ 1024 bytes (single chunk).
    /// Used by the Merkle tree single-chunk-root path.
    /// </summary>
    public static byte[] KeyedHash(ReadOnlySpan<byte> key, ReadOnlySpan<byte> input)
    {
        if (key.Length != DigestSize) throw new ArgumentException("Key must be 32 bytes.");
        if (input.Length > ChunkLen)
            throw new NotSupportedException("Blake3.KeyedHash only supports inputs ≤ 1024 bytes.");

        Span<uint> chainingValue = stackalloc uint[8];
        LoadKeyWords(key, chainingValue);
        Span<uint> blockWords = stackalloc uint[16];

        int totalBlocks = Math.Max(1, (input.Length + BlockLen - 1) / BlockLen);

        for (int i = 0; i < totalBlocks; i++)
        {
            int offset = i * BlockLen;
            int blockBytes = Math.Min(BlockLen, input.Length - offset);
            LoadBlock(input.Slice(offset, blockBytes), blockWords);

            uint flags = FlagKeyedHash;
            if (i == 0) flags |= FlagChunkStart;
            bool isLast = i == totalBlocks - 1;
            if (isLast) flags |= FlagChunkEnd | FlagRoot;

            Span<uint> compressed = stackalloc uint[16];
            Compress(chainingValue, blockWords, 0, (uint)blockBytes, flags, compressed);

            if (isLast)
                return WordsToBytes(compressed[..8]);

            compressed[..8].CopyTo(chainingValue);
        }
        throw new InvalidOperationException("unreachable");
    }

    /// <summary>
    /// Compute the chaining value (CV) for a single 1024-byte chunk in keyed mode.
    /// Non-root: returns 32 bytes (first 8 words of last compress).
    /// </summary>
    public static byte[] ChunkCv(ReadOnlySpan<byte> chunk, ulong chunkIndex, ReadOnlySpan<byte> key)
    {
        if (key.Length != DigestSize) throw new ArgumentException("Key must be 32 bytes.");
        if (chunk.Length is <= 0 or > ChunkLen) throw new ArgumentOutOfRangeException(nameof(chunk));

        Span<uint> chainingValue = stackalloc uint[8];
        LoadKeyWords(key, chainingValue);
        Span<uint> blockWords = stackalloc uint[16];

        int totalBlocks = Math.Max(1, (chunk.Length + BlockLen - 1) / BlockLen);

        for (int i = 0; i < totalBlocks; i++)
        {
            int offset = i * BlockLen;
            int blockBytes = Math.Min(BlockLen, chunk.Length - offset);
            LoadBlock(chunk.Slice(offset, blockBytes), blockWords);

            uint flags = FlagKeyedHash;
            if (i == 0) flags |= FlagChunkStart;
            if (i == totalBlocks - 1) flags |= FlagChunkEnd;

            Span<uint> compressed = stackalloc uint[16];
            Compress(chainingValue, blockWords, chunkIndex, (uint)blockBytes, flags, compressed);

            if (i == totalBlocks - 1)
                return WordsToBytes(compressed[..8]);

            compressed[..8].CopyTo(chainingValue);
        }
        throw new InvalidOperationException("unreachable");
    }

    /// <summary>Compute a non-root parent CV from two child CVs (keyed mode).</summary>
    public static byte[] ParentCv(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, ReadOnlySpan<byte> key)
    {
        var result = new byte[DigestSize];
        ParentCv(left, right, key, result);
        return result;
    }

    /// <summary>Compute a non-root parent CV into <paramref name="output"/>.</summary>
    public static void ParentCv(
        ReadOnlySpan<byte> left,
        ReadOnlySpan<byte> right,
        ReadOnlySpan<byte> key,
        Span<byte> output)
    {
        if (output.Length < DigestSize) throw new ArgumentException("Output must be at least 32 bytes.", nameof(output));
        Span<uint> words = stackalloc uint[16];
        CompressParent(left, right, key, FlagParent | FlagKeyedHash, words);
        WriteWordsToBytes(words[..8], output);
    }

    /// <summary>Compute the root CV from two child CVs (keyed mode). Sets ROOT flag.</summary>
    public static byte[] RootCv(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, ReadOnlySpan<byte> key)
    {
        var result = new byte[DigestSize];
        RootCv(left, right, key, result);
        return result;
    }

    /// <summary>Compute the root CV into <paramref name="output"/>. Sets ROOT flag.</summary>
    public static void RootCv(
        ReadOnlySpan<byte> left,
        ReadOnlySpan<byte> right,
        ReadOnlySpan<byte> key,
        Span<byte> output)
    {
        if (output.Length < DigestSize) throw new ArgumentException("Output must be at least 32 bytes.", nameof(output));
        Span<uint> words = stackalloc uint[16];
        CompressParent(left, right, key, FlagParent | FlagRoot | FlagKeyedHash, words);
        WriteWordsToBytes(words[..8], output);
    }

    /// <summary>
    /// Build a BLAKE3 keyed Merkle tree root from arbitrary-length data.
    /// Data is split into 1024-byte chunks, zero-padded to chunk boundary.
    /// Matches the pool's Blake3MerkleHasher construction exactly.
    /// </summary>
    public static byte[] MerkleRoot(ReadOnlySpan<byte> data, ReadOnlySpan<byte> key)
    {
        if (key.Length != DigestSize) throw new ArgumentException("Key must be 32 bytes.");

        // Pad to chunk boundary
        int paddedLen = ((data.Length + ChunkLen - 1) / ChunkLen) * ChunkLen;
        if (paddedLen == 0) paddedLen = ChunkLen;
        int totalLeaves = paddedLen / ChunkLen;

        byte[]? rented = null;
        ReadOnlySpan<byte> paddedData;
        if (paddedLen == data.Length)
        {
            paddedData = data;
        }
        else
        {
            rented = ArrayPool<byte>.Shared.Rent(paddedLen);
            rented.AsSpan(0, paddedLen).Clear();
            data.CopyTo(rented);
            paddedData = rented.AsSpan(0, paddedLen);
        }

        try
        {
            if (totalLeaves == 1)
                return KeyedHash(key, paddedData[..ChunkLen]);

            // Compute leaf CVs
            var cvs = new byte[totalLeaves][];
            for (int i = 0; i < totalLeaves; i++)
                cvs[i] = ChunkCv(paddedData.Slice(i * ChunkLen, ChunkLen), (ulong)i, key);

            // Build binary tree bottom-up
            while (cvs.Length > 2)
            {
                int nextLen = (cvs.Length + 1) / 2;
                var next = new byte[nextLen][];
                for (int i = 0; i < cvs.Length; i += 2)
                {
                    next[i / 2] = (i + 1 < cvs.Length)
                        ? ParentCv(cvs[i], cvs[i + 1], key)
                        : cvs[i];
                }
                cvs = next;
            }

            return RootCv(cvs[0], cvs[1], key);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<byte>.Shared.Return(rented, clearArray: true);
        }
    }

    // ---- Internal compression ----

    private static void CompressParent(
        ReadOnlySpan<byte> left, ReadOnlySpan<byte> right,
        ReadOnlySpan<byte> key, uint flags, Span<uint> output)
    {
        if (left.Length != DigestSize) throw new ArgumentException("Left must be 32 bytes.");
        if (right.Length != DigestSize) throw new ArgumentException("Right must be 32 bytes.");
        if (key.Length != DigestSize) throw new ArgumentException("Key must be 32 bytes.");

        Span<uint> cv = stackalloc uint[8];
        LoadKeyWords(key, cv);
        Span<uint> blockWords = stackalloc uint[16];
        for (int i = 0; i < 8; i++)
            blockWords[i] = BinaryPrimitives.ReadUInt32LittleEndian(left.Slice(i * 4, 4));
        for (int i = 0; i < 8; i++)
            blockWords[8 + i] = BinaryPrimitives.ReadUInt32LittleEndian(right.Slice(i * 4, 4));

        Compress(cv, blockWords, 0, BlockLen, flags, output);
    }

    internal static void Compress(
        ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block, ulong counter,
        uint blockLen, uint flags, Span<uint> outWords)
    {
        Span<uint> state = stackalloc uint[16];
        state[ 0] = cv[0]; state[ 1] = cv[1]; state[ 2] = cv[2]; state[ 3] = cv[3];
        state[ 4] = cv[4]; state[ 5] = cv[5]; state[ 6] = cv[6]; state[ 7] = cv[7];
        state[ 8] = IV[0]; state[ 9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
        state[12] = (uint)(counter & 0xFFFFFFFFu);
        state[13] = (uint)(counter >> 32);
        state[14] = blockLen;
        state[15] = flags;

        Span<uint> m = stackalloc uint[16];
        block.CopyTo(m);

        for (int round = 0; round < 7; round++)
        {
            G(state, 0, 4,  8, 12, m[0],  m[1]);
            G(state, 1, 5,  9, 13, m[2],  m[3]);
            G(state, 2, 6, 10, 14, m[4],  m[5]);
            G(state, 3, 7, 11, 15, m[6],  m[7]);
            G(state, 0, 5, 10, 15, m[8],  m[9]);
            G(state, 1, 6, 11, 12, m[10], m[11]);
            G(state, 2, 7,  8, 13, m[12], m[13]);
            G(state, 3, 4,  9, 14, m[14], m[15]);

            if (round < 6)
            {
                Span<uint> permuted = stackalloc uint[16];
                for (int i = 0; i < 16; i++) permuted[i] = m[MsgPermutation[i]];
                permuted.CopyTo(m);
            }
        }

        for (int i = 0; i < 8; i++)
        {
            outWords[i]     = state[i] ^ state[i + 8];
            outWords[i + 8] = state[i + 8] ^ cv[i];
        }
    }

    private static void G(Span<uint> s, int a, int b, int c, int d, uint mx, uint my)
    {
        s[a] = s[a] + s[b] + mx;
        s[d] = RotR(s[d] ^ s[a], 16);
        s[c] = s[c] + s[d];
        s[b] = RotR(s[b] ^ s[c], 12);
        s[a] = s[a] + s[b] + my;
        s[d] = RotR(s[d] ^ s[a], 8);
        s[c] = s[c] + s[d];
        s[b] = RotR(s[b] ^ s[c], 7);
    }

    private static uint RotR(uint x, int n) => (x >> n) | (x << (32 - n));

    private static void LoadBlock(ReadOnlySpan<byte> data, Span<uint> words)
    {
        words.Clear();
        for (int i = 0; i < Math.Min(data.Length / 4, 16); i++)
            words[i] = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(i * 4, 4));
        // Handle trailing bytes (< 4)
        int fullWords = data.Length / 4;
        int tail = data.Length % 4;
        if (tail > 0 && fullWords < 16)
        {
            Span<byte> tmp = stackalloc byte[4];
            tmp.Clear();
            data.Slice(fullWords * 4, tail).CopyTo(tmp);
            words[fullWords] = BinaryPrimitives.ReadUInt32LittleEndian(tmp);
        }
    }

    private static void LoadKeyWords(ReadOnlySpan<byte> key, Span<uint> words)
    {
        for (int i = 0; i < 8; i++)
            words[i] = BinaryPrimitives.ReadUInt32LittleEndian(key.Slice(i * 4, 4));
    }

    private static byte[] WordsToBytes(ReadOnlySpan<uint> words)
    {
        var result = new byte[words.Length * 4];
        WriteWordsToBytes(words, result);
        return result;
    }

    private static void WriteWordsToBytes(ReadOnlySpan<uint> words, Span<byte> destination)
    {
        if (destination.Length < words.Length * 4)
            throw new ArgumentException("Destination is too small.", nameof(destination));
        for (int i = 0; i < words.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(i * 4, 4), words[i]);
    }
}
