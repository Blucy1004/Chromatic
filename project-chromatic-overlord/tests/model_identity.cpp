#include "zero/sha256.hpp"
#include "zero/network.hpp"
#include <iostream>
#include <stdexcept>
using namespace zero;
void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main(int argc,char** argv){try {
    Sha256 empty;require(empty.hex_digest()=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty vector");
    Sha256 abc;abc.update("a",1);abc.update("bc",2);
    require(abc.hex_digest()=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 abc vector");
    Sha256 million;std::string chunk(1000,'a');for(int i=0;i<1000;++i)million.update(chunk.data(),chunk.size());
    require(million.hex_digest()=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA256 million-a vector");
    Network net;require(!net.loaded()&&net.loaded_sha256().empty(),"uniform engine has no model identity");
    if(argc>1){net.load(argv[1]);auto hash=net.loaded_sha256();require(hash=="f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9","loaded champion tensors match known file SHA256");
        bool rejected=false;try{net.load("missing-identity-fixture.zero");}catch(const std::exception&){rejected=true;}
        require(rejected&&net.loaded_sha256()==hash&&net.loaded(),"failed transactional load preserves model identity");}
    std::cout<<"SHA256 vectors and actual loaded-model identity passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
