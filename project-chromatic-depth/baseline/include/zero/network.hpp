#pragma once
#include "board.hpp"
#include <string>
namespace zero {
struct Evaluation {std::array<float,ACTIONS> policy{};std::array<float,3> wdl{};};
enum class InferenceBackend {Scalar,SIMD};
class Network {
    struct Layer {
        int in=0,out=0;std::vector<float> weights,bias;
        float row(const std::vector<float>& x,int output,InferenceBackend backend) const;
        void apply(const std::vector<float>& x,std::vector<float>& y,InferenceBackend backend) const;
    };
    Layer input_,policy_,wdl_;
    std::vector<Layer> residual_;
    bool loaded_=false;
    InferenceBackend backend_=InferenceBackend::SIMD;
    bool sparse_=true;
public:
    struct Workspace {std::vector<float> input,a,b,wdl;};
    void load(const std::string& path); // transactional: old model survives invalid files
    bool loaded() const {return loaded_;}
    void set_backend(InferenceBackend backend){backend_=backend;}
    InferenceBackend backend() const{return backend_;}
    void set_sparse(bool enabled){sparse_=enabled;}
    bool sparse() const{return sparse_;}
    Evaluation evaluate(const Board& b) const;
    Evaluation evaluate_legal(const Board& b,const std::vector<Move>& moves,Workspace& workspace) const;
private:
    Evaluation evaluate_impl(const Board& b,const std::vector<Move>* moves,Workspace& workspace) const;
};
}
