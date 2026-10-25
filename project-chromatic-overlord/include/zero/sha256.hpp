#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
namespace zero {
// SHA-256 with streaming input; used only for model identity, never search scoring.
class Sha256 {
    std::array<uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t,64> buffer_{};
    uint64_t bytes_=0;
    size_t used_=0;
    void block(const uint8_t* data);
public:
    void update(const void* data,size_t count);
    std::string hex_digest() const;
};
}
