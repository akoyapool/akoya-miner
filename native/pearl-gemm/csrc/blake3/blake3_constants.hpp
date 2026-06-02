#pragma once

#include <cstdint>

using u32 = uint32_t;

namespace blake3 {
// BLAKE3 constants
constexpr u32 CHAINING_VALUE_SIZE = 32;
constexpr u32 CHAINING_VALUE_SIZE_U32 = CHAINING_VALUE_SIZE / sizeof(u32);  // 8
constexpr u32 KEY_SIZE = CHAINING_VALUE_SIZE;
constexpr u32 CHUNK_SIZE = 1024;
constexpr u32 MSG_BLOCK_SIZE = 64;
constexpr u32 MSG_BLOCK_SIZE_U32 = MSG_BLOCK_SIZE / sizeof(u32);  // 16
// Admissible values for the flags field:
constexpr u32 CHUNK_START = 1 << 0;
constexpr u32 CHUNK_END = 1 << 1;
constexpr u32 PARENT = 1 << 2;
constexpr u32 ROOT = 1 << 3;
constexpr u32 KEYED_HASH = 1 << 4;
constexpr u32 DERIVE_KEY_CONTEXT = 1 << 5;
constexpr u32 DERIVE_KEY_MATERIAL = 1 << 6;

constexpr u32 IV0 = 0x6A09E667;
constexpr u32 IV1 = 0xBB67AE85;
constexpr u32 IV2 = 0x3C6EF372;
constexpr u32 IV3 = 0xA54FF53A;
constexpr u32 IV4 = 0x510e527f;
constexpr u32 IV5 = 0x9b05688c;
constexpr u32 IV6 = 0x1f83d9ab;
constexpr u32 IV7 = 0x5be0cd19;

}  // namespace blake3