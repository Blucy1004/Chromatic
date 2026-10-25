#include "zero/network.hpp"
#include "zero/sha256.hpp"
#include <bit>
#include "zero/profile.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#if defined(_M_X64)||defined(__SSE2__)
#include <emmintrin.h>
#define CHROMATIC_SSE2 1
#endif

namespace zero {
float Network::Layer::row(const std::vector<float>& x,int output,InferenceBackend backend) const {
    const float* w=weights.data()+size_t(output)*in;
    float sum=bias[output];
#ifdef CHROMATIC_SSE2
    if(backend==InferenceBackend::SIMD) {
        __m128 partial=_mm_setzero_ps();int i=0;
        for(;i+3<in;i+=4)partial=_mm_add_ps(partial,_mm_mul_ps(_mm_loadu_ps(w+i),_mm_loadu_ps(x.data()+i)));
        float lanes[4];_mm_storeu_ps(lanes,partial);
        for(float value:lanes)sum+=value;
        for(;i<in;++i)sum+=w[i]*x[i];
    } else
#else
    (void)backend;
#endif
    {for(int i=0;i<in;++i)sum+=w[i]*x[i];}
    if(!std::isfinite(sum))throw std::runtime_error("nonfinite neural activation");
    return sum;
}
void Network::Layer::apply(const std::vector<float>& x,std::vector<float>& y,InferenceBackend backend) const {
    y.resize(out);for(int output=0;output<out;++output)y[output]=row(x,output,backend);
}
void Network::load(const std::string& path) {
    static_assert(sizeof(float)==4);
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open network: "+path);
    char magic[8]{};f.read(magic,8);
    if(!f)throw std::runtime_error("truncated network magic");
    if(std::string(magic,8)!="ZERO0001")throw std::runtime_error("incompatible network version or magic");
    auto number=[&](){uint32_t n=0;f.read(reinterpret_cast<char*>(&n),4);if(!f)throw std::runtime_error("truncated network header");return n;};
    auto inputs=number(),width=number(),blocks=number(),actions=number();
    if(inputs!=INPUTS||actions!=ACTIONS||width<1||width>1024||blocks>16)throw std::runtime_error("unsupported network dimensions");
    Network candidate;candidate.backend_=backend_;candidate.sparse_=sparse_;
    auto layer=[&](int in,int out) {
        Layer l;l.in=in;l.out=out;l.weights.resize(size_t(in)*out);l.bias.resize(out);
        f.read(reinterpret_cast<char*>(l.weights.data()),std::streamsize(l.weights.size()*4));
        f.read(reinterpret_cast<char*>(l.bias.data()),std::streamsize(l.bias.size()*4));
        if(!f)throw std::runtime_error("truncated network weights");
        for(float v:l.weights)if(!std::isfinite(v))throw std::runtime_error("nonfinite weight");
        for(float v:l.bias)if(!std::isfinite(v))throw std::runtime_error("nonfinite bias");
        return l;
    };
    candidate.input_=layer(INPUTS,int(width));
    for(uint32_t i=0;i<blocks;++i)candidate.residual_.push_back(layer(int(width),int(width)));
    candidate.policy_=layer(int(width),ACTIONS);candidate.wdl_=layer(int(width),3);
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing network bytes");
    // Identity comes from the actual validated in-memory tensors, not a later
    // reread of a potentially replaced file. ZERO0001 is little-endian FP32.
    static_assert(std::endian::native==std::endian::little);
    Sha256 digest;digest.update("ZERO0001",8);
    const uint32_t header[]={inputs,width,blocks,actions};digest.update(header,sizeof(header));
    auto hash_layer=[&](const Layer& l){digest.update(l.weights.data(),l.weights.size()*sizeof(float));digest.update(l.bias.data(),l.bias.size()*sizeof(float));};
    hash_layer(candidate.input_);for(const auto& l:candidate.residual_)hash_layer(l);
    hash_layer(candidate.policy_);hash_layer(candidate.wdl_);
    candidate.loaded_sha256_=digest.hex_digest();
    candidate.loaded_=true;*this=std::move(candidate);
}
Evaluation Network::evaluate(const Board& b) const {
    Workspace workspace;return evaluate_impl(b,nullptr,workspace);
}
Evaluation Network::evaluate_legal(const Board& b,const std::vector<Move>& moves,Workspace& workspace) const {
    return evaluate_impl(b,sparse_?&moves:nullptr,workspace);
}
Evaluation Network::evaluate_impl(const Board& b,const std::vector<Move>* moves,Workspace& workspace) const {
    profile::Timer timer(profile::neural_ns);if(profile::active)++profile::neural_calls;
    Evaluation e;
    if(!loaded_){e.wdl={.25f,.5f,.25f};return e;}
    auto features=b.features();workspace.input.assign(features.begin(),features.end());
    input_.apply(workspace.input,workspace.a,backend_);
    for(float& value:workspace.a)value=std::max(0.0f,value);
    for(const auto& layer:residual_) {
        layer.apply(workspace.a,workspace.b,backend_);
        for(size_t i=0;i<workspace.a.size();++i)workspace.a[i]=std::max(0.0f,workspace.a[i]+workspace.b[i]);
    }
    if(moves) {
        // This is exact row selection, with the same FP32 arithmetic as the full head.
        // Unselected logits are not probabilities and must never enter a softmax.
        for(auto m:*moves){int index=action(m);e.policy[index]=policy_.row(workspace.a,index,backend_);}
    } else {
        for(int index=0;index<ACTIONS;++index)e.policy[index]=policy_.row(workspace.a,index,backend_);
    }
    wdl_.apply(workspace.a,workspace.wdl,backend_);
    float maximum=*std::max_element(workspace.wdl.begin(),workspace.wdl.end()),sum=0;
    for(int i=0;i<3;++i){e.wdl[i]=std::exp(workspace.wdl[i]-maximum);sum+=e.wdl[i];}
    for(float& value:e.wdl)value/=sum;
    return e;
}
}
