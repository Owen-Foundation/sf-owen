#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Decode LEB128 signed integers from buffer
int decode_leb128_i16(const uint8_t* in, uint32_t in_len, int16_t* out, uint32_t count) {
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    
    while (out_pos < count && in_pos < in_len) {
        uint32_t result = 0;
        uint32_t shift = 0;
        uint8_t byte;
        
        do {
            if (in_pos >= in_len) return -1;
            byte = in[in_pos++];
            result |= (uint32_t)(byte & 0x7f) << (shift % 32);
            shift += 7;
        } while (byte & 0x80);
        
        if (shift < 32 && (byte & 0x40)) {
            result |= ~((1U << shift) - 1);
        }
        
        out[out_pos++] = (int16_t)result;
    }
    
    return (out_pos == count) ? 0 : -2;
}

int decode_leb128_i32(const uint8_t* in, uint32_t in_len, int32_t* out, uint32_t count) {
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    
    while (out_pos < count && in_pos < in_len) {
        uint32_t result = 0;
        uint32_t shift = 0;
        uint8_t byte;
        
        do {
            if (in_pos >= in_len) return -1;
            byte = in[in_pos++];
            result |= (uint32_t)(byte & 0x7f) << (shift % 32);
            shift += 7;
        } while (byte & 0x80);
        
        if (shift < 32 && (byte & 0x40)) {
            result |= ~((1U << shift) - 1);
        }
        
        out[out_pos++] = (int32_t)result;
    }
    
    return (out_pos == count) ? 0 : -2;
}
