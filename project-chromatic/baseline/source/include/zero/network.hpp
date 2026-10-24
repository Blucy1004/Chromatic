#pragma once
#include "board.hpp"
#include <string>
namespace zero {
struct Evaluation {std::array<float,ACTIONS> policy{};std::array<float,3> wdl{};};
class Network {
    struct Layer {int in=0,out=0;std::vector<float> weights,bias;std::vector<float> apply(const std::vector<float>& x) const;};
    Layer input_,policy_,wdl_;
    std::vector<Layer> residual_;
    bool loaded_=false;
public:
    void load(const std::string& path); // transactional: old model survives invalid files
    bool loaded() const {return loaded_;}
    Evaluation evaluate(const Board& b) const;
};
}
