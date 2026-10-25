#pragma once
#include "network.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
namespace zero {
// Workers only read immutable boards, legal lists and weights. The coordinator
// owns all tree state. Destruction drains pending jobs and joins every worker.
class LeafPool {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::packaged_task<Evaluation(Network::Workspace&)>> jobs_;
    std::vector<std::thread> workers_;
    bool done_=false;
    bool timed_=false;
    std::atomic<uint64_t> neural_ns_{0};
public:
    explicit LeafPool(int count,bool timed=false):timed_(timed) {
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
    uint64_t neural_ns() const{return neural_ns_.load(std::memory_order_relaxed);}
    LeafPool(const LeafPool&)=delete;
    ~LeafPool(){close();}
    void close() {
        {std::lock_guard lock(mutex_);done_=true;}
        ready_.notify_all();
        for(auto& worker:workers_)if(worker.joinable())worker.join();
    }
    std::future<Evaluation> submit(const Network& net,Board board,std::vector<Move> moves) {
        std::packaged_task<Evaluation(Network::Workspace&)> job(
            [this,&net,board=std::move(board),moves=std::move(moves)](Network::Workspace& w){
                const auto start=timed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                auto result=net.evaluate_legal(board,moves,w);
                if(timed_)neural_ns_.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count()),std::memory_order_relaxed);
                return result;
            });
        auto result=job.get_future();
        {std::lock_guard lock(mutex_);jobs_.push_back(std::move(job));}
        ready_.notify_one();return result;
    }
};
}
