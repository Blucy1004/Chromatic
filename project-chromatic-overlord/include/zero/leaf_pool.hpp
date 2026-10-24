#pragma once
#include "network.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
namespace zero {
// Workers only read immutable boards, legal lists and weights. The coordinator
// owns all tree state. Destruction drains pending jobs and joins every worker.
class LeafPool {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::packaged_task<Evaluation(Network::Workspace&)>> jobs_;
    std::vector<std::thread> workers_;
    bool done_=false;
public:
    explicit LeafPool(int count) {
        try {
            for(int i=0;i<count;++i)workers_.emplace_back([this] {
                Network::Workspace workspace;
                for(;;) {
                    std::packaged_task<Evaluation(Network::Workspace&)> job;
                    {
                        std::unique_lock lock(mutex_);
                        ready_.wait(lock,[&]{return done_||!jobs_.empty();});
                        if(jobs_.empty()&&done_)return;
                        job=std::move(jobs_.front());jobs_.pop_front();
                    }
                    job(workspace);
                }
            });
        } catch(...) {close();throw;}
    }
    LeafPool(const LeafPool&)=delete;
    ~LeafPool(){close();}
    void close() {
        {std::lock_guard lock(mutex_);done_=true;}
        ready_.notify_all();
        for(auto& worker:workers_)if(worker.joinable())worker.join();
    }
    std::future<Evaluation> submit(const Network& net,Board board,std::vector<Move> moves) {
        std::packaged_task<Evaluation(Network::Workspace&)> job(
            [&net,board=std::move(board),moves=std::move(moves)](Network::Workspace& w){return net.evaluate_legal(board,moves,w);});
        auto result=job.get_future();
        {std::lock_guard lock(mutex_);jobs_.push_back(std::move(job));}
        ready_.notify_one();return result;
    }
};
}
