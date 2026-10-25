#include "zero/sha256.hpp"
#include <algorithm>
#include <bit>
#include <iomanip>
#include <sstream>
namespace zero {
void Sha256::block(const uint8_t* p) {
    static constexpr uint32_t k[]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64]{};
    for(int i=0;i<16;++i)w[i]=(uint32_t(p[4*i])<<24)|(uint32_t(p[4*i+1])<<16)|(uint32_t(p[4*i+2])<<8)|uint32_t(p[4*i+3]);
    for(int i=16;i<64;++i) {
        auto s0=std::rotr(w[i-15],7)^std::rotr(w[i-15],18)^(w[i-15]>>3);
        auto s1=std::rotr(w[i-2],17)^std::rotr(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    auto [a,b,c,d,e,f,g,h]=state_;
    for(int i=0;i<64;++i) {
        auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);
        auto t1=h+s1+((e&f)^((~e)&g))+k[i]+w[i];
        auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);
        auto t2=s0+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
    state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
}
void Sha256::update(const void* input,size_t count) {
    auto p=static_cast<const uint8_t*>(input);bytes_+=uint64_t(count);
    while(count) {
        const auto n=std::min(count,buffer_.size()-used_);
        std::copy_n(p,n,buffer_.begin()+used_);used_+=n;p+=n;count-=n;
        if(used_==64){block(buffer_.data());used_=0;}
    }
}
std::string Sha256::hex_digest() const {
    auto copy=*this;const uint64_t bits=bytes_*8;uint8_t pad=0x80;copy.update(&pad,1);pad=0;
    while(copy.used_!=56)copy.update(&pad,1);
    uint8_t length[8]{};for(int i=0;i<8;++i)length[7-i]=uint8_t(bits>>(i*8));copy.update(length,8);
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto word:copy.state_)out<<std::setw(8)<<word;
    return out.str();
}
}
