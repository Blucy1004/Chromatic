#include "zero/profile.hpp"
#include "zero/network.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace zero {
std::vector<float> Network::Layer::apply(const std::vector<float>& x) const {
    std::vector<float> y=bias;
    for(int o=0;o<out;++o) {const float* w=weights.data()+size_t(o)*in;float s=y[o];for(int i=0;i<in;++i)s+=w[i]*x[i];y[o]=s;}
    return y;
}
void Network::load(const std::string& path) {
    static_assert(sizeof(float)==4);
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open network: "+path);
    char magic[8];f.read(magic,8);if(std::string(magic,8)!="ZERO0001")throw std::runtime_error("invalid network magic");
    auto number=[&](){uint32_t n=0;f.read(reinterpret_cast<char*>(&n),4);if(!f)throw std::runtime_error("truncated network header");return n;};
    auto inputs=number(),width=number(),blocks=number(),actions=number();
    if(inputs!=INPUTS||actions!=ACTIONS||width<1||width>1024||blocks>16)throw std::runtime_error("unsupported network dimensions");
    Network candidate;
    auto layer=[&](int in,int out){Layer l;l.in=in;l.out=out;l.weights.resize(size_t(in)*out);l.bias.resize(out);f.read(reinterpret_cast<char*>(l.weights.data()),std::streamsize(l.weights.size()*4));f.read(reinterpret_cast<char*>(l.bias.data()),std::streamsize(l.bias.size()*4));if(!f)throw std::runtime_error("truncated network weights");for(float v:l.weights)if(!std::isfinite(v))throw std::runtime_error("nonfinite weight");for(float v:l.bias)if(!std::isfinite(v))throw std::runtime_error("nonfinite bias");return l;};
    candidate.input_=layer(INPUTS,int(width));
    for(uint32_t i=0;i<blocks;++i)candidate.residual_.push_back(layer(int(width),int(width)));
    candidate.policy_=layer(int(width),ACTIONS);candidate.wdl_=layer(int(width),3);
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing network bytes");
    candidate.loaded_=true;*this=std::move(candidate);
}
Evaluation Network::evaluate(const Board& b) const {
    profile::Timer timer(profile::neural_ns);if(profile::active)++profile::neural_calls;
    Evaluation e;
    if(!loaded_) { // Explicit uniform baseline for rule tests and pre-bootstrap play.
        e.wdl={0.25f,0.5f,0.25f};return e;
    }
    auto features=b.features();std::vector<float>x(features.begin(),features.end());x=input_.apply(x);for(float& v:x)v=std::max(0.0f,v);
    for(const auto& layer:residual_) {auto y=layer.apply(x);for(size_t i=0;i<x.size();++i)x[i]=std::max(0.0f,x[i]+y[i]);}
    auto p=policy_.apply(x),v=wdl_.apply(x);std::copy(p.begin(),p.end(),e.policy.begin());
    float max=*std::max_element(v.begin(),v.end()),sum=0;for(int i=0;i<3;++i){e.wdl[i]=std::exp(v[i]-max);sum+=e.wdl[i];}for(float& f:e.wdl)f/=sum;
    return e;
}
}
