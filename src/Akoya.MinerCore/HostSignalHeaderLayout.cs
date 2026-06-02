// HostSignalHeaderLayout — managed mirror of the C++ HostSignalHeader
// struct in miner/pearl-gemm/csrc/gemm/host_signal_header.hpp.
//
// The kernel writes this struct directly into the pinned host buffer via
// UVA. We only need to *read* it after a sync, so the layout below is
// derived by hand from the C++ definition. Verified by checking that
// `GetHostSignalHeaderSize()` (which rounds up to 128) is at least
// `sizeof(HostSignalHeader)`.
//
// Layout (no padding within fields; cute::array<T,N> is just T[N]):
//   offset  size  field
//        0     4  status                    (HostSignalStatus enum, int)
//        4    12  gridDim[3]                u32
//       16    12  blockDim[3]               u32
//       28    12  blockIdx[3]               u32
//       40    12  tileCoord[3]              u32
//       52    12  threadIdx[3]              u32
//       64     2  num_registers_per_thread  u16
//       66   256  thread_rows[256]          u8
//      322   256  thread_cols[256]          u8
//      578     2  (alignment padding to 4-byte boundary for MMASize)
//      580    12  mma_size                  3 × i32
//      592    12  mma_tile_size             3 × i32
//      604    32  target[8]                 u32
//   total: 636 (rounded up to 128 by GetHostSignalHeaderSize → 640)
//
// The host_signal_header_size constant on the C side is
//   ((sizeof(HostSignalHeader) + 127) / 128) * 128.
// We never assume a specific value — Akoya.PearlGemm.PearlGemmNative
// .GetHostSignalHeaderSize() is queried at runtime.

namespace Akoya.MinerCore;

public static class HostSignalHeaderLayout
{
    public const int OFF_STATUS                   = 0;
    public const int OFF_GRID_DIM                 = 4;
    public const int OFF_BLOCK_DIM                = 16;
    public const int OFF_BLOCK_IDX                = 28;
    public const int OFF_TILE_COORD               = 40;
    public const int OFF_THREAD_IDX               = 52;
    public const int OFF_NUM_REGISTERS_PER_THREAD = 64;
    public const int OFF_THREAD_ROWS              = 66;
    public const int OFF_THREAD_COLS              = 322;
    public const int OFF_MMA_SIZE                 = 580;       // padded +2 for 4-byte align
    public const int OFF_MMA_TILE_SIZE            = 592;
    public const int OFF_TARGET                   = 604;

    public const int MAX_NUM_REGISTERS_PER_THREAD = 256;

    public readonly record struct ProofTileIndices(ulong[] ARowIndices, ulong[] BColumnIndices);

    /// <summary>
    /// Mirrors <c>pearl_gemm.helpers.extract_indices(header)</c>:
    ///   row_tile_coord = tileCoord[0] * mma_tile_size.m
    ///   col_tile_coord = tileCoord[1] * mma_tile_size.n
    ///   thread_rows = sorted(set(thread_rows[:num_registers_per_thread]))
    ///   A_row_indices    = [row_tile_coord + r for r in thread_rows]
    ///   B_column_indices = [col_tile_coord + c for c in thread_cols]
    /// </summary>
    public static ProofTileIndices ExtractIndices(ReadOnlySpan<byte> header)
    {
        // Read u32 LE little-endian on x86_64 — we just trust BitConverter.
        uint tileRow  = BitConverter.ToUInt32(header.Slice(OFF_TILE_COORD + 0, 4));
        uint tileCol  = BitConverter.ToUInt32(header.Slice(OFF_TILE_COORD + 4, 4));
        int  mmaTileH = BitConverter.ToInt32 (header.Slice(OFF_MMA_TILE_SIZE + 0, 4));
        int  mmaTileW = BitConverter.ToInt32 (header.Slice(OFF_MMA_TILE_SIZE + 4, 4));

        ushort numRegs = BitConverter.ToUInt16(header.Slice(OFF_NUM_REGISTERS_PER_THREAD, 2));
        if (numRegs == 0 || numRegs > MAX_NUM_REGISTERS_PER_THREAD)
            throw new InvalidDataException($"num_registers_per_thread out of range: {numRegs}");

        var rows = new SortedSet<byte>();
        var cols = new SortedSet<byte>();
        for (int i = 0; i < numRegs; i++)
        {
            rows.Add(header[OFF_THREAD_ROWS + i]);
            cols.Add(header[OFF_THREAD_COLS + i]);
        }

        ulong rowOff = (ulong)tileRow * (ulong)mmaTileH;
        ulong colOff = (ulong)tileCol * (ulong)mmaTileW;

        var aRows = new ulong[rows.Count];
        var bCols = new ulong[cols.Count];
        int ai = 0; foreach (var r in rows) aRows[ai++] = rowOff + r;
        int bi = 0; foreach (var c in cols) bCols[bi++] = colOff + c;

        return new ProofTileIndices(aRows, bCols);
    }

    public static (uint TileRow, uint TileCol, int MmaTileM, int MmaTileN, ushort NumRegs)
        DebugInfo(ReadOnlySpan<byte> header)
        => (BitConverter.ToUInt32(header.Slice(OFF_TILE_COORD + 0, 4)),
            BitConverter.ToUInt32(header.Slice(OFF_TILE_COORD + 4, 4)),
            BitConverter.ToInt32 (header.Slice(OFF_MMA_TILE_SIZE + 0, 4)),
            BitConverter.ToInt32 (header.Slice(OFF_MMA_TILE_SIZE + 4, 4)),
            BitConverter.ToUInt16(header.Slice(OFF_NUM_REGISTERS_PER_THREAD, 2)));
}
