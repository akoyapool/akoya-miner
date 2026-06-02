#pragma once

#include <cute/container/array.hpp>
#include "blake3/blake3_constants.hpp"

enum HostSignalStatus { kSignalIdle = 0, kSignalTriggered = 1 };

// Struct to keep sync between different blocks found all on device memory
struct __align__(8) HostSignalSync {
  int global_lock = 0;
  HostSignalStatus status = kSignalIdle;
};

struct MMASize {
  int m;
  int n;
  int k;
};

static constexpr int MAX_NUM_REGISTERS_PER_THREAD = 256;

struct __align__(128) HostSignalHeader {
  HostSignalStatus status;
  cute::array<uint32_t, 3> gridDim;
  cute::array<uint32_t, 3> blockDim;
  cute::array<uint32_t, 3> blockIdx;
  cute::array<uint32_t, 3> tileCoord;
  cute::array<uint32_t, 3> threadIdx;
  uint16_t num_registers_per_thread;

  // Notice only first num_registers_per_thread elements are valid
  cute::array<uint8_t, MAX_NUM_REGISTERS_PER_THREAD> thread_rows;
  cute::array<uint8_t, MAX_NUM_REGISTERS_PER_THREAD> thread_cols;
  MMASize mma_size;
  MMASize mma_tile_size;
  cute::array<uint32_t, blake3::CHAINING_VALUE_SIZE_U32> target;

  CUTE_HOST_DEVICE bool block_in_bounds() const {
    uint32_t a_row_offset = tileCoord[0] * mma_tile_size.m;
    uint32_t b_col_offset = tileCoord[1] * mma_tile_size.n;
    uint32_t max_thread_row =
        static_cast<uint32_t>(thread_rows[num_registers_per_thread - 1]);
    uint32_t max_thread_col =
        static_cast<uint32_t>(thread_cols[num_registers_per_thread - 1]);
    return (a_row_offset + max_thread_row < (uint32_t)mma_size.m) &&
           (b_col_offset + max_thread_col < (uint32_t)mma_size.n);
  }
};

static constexpr int host_signal_header_size =
    ((sizeof(HostSignalHeader) + 128 - 1) / 128) * 128;
