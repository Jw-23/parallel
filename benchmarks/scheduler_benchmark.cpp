#include "parallel.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using parallel::detail::Executor;
using ChunkWork = Executor::ChunkTask;
namespace outcome = boost::outcome_v2;

// A single mutex protects the pending chunks and batch completion. The caller
// also claims work, giving each scheduler the same number of execution threads.
class MutexPool {
public:
    explicit MutexPool(const ChunkWork& work) : work_(work) {}

    ~MutexPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    parallel::Result<void> start(std::size_t background_workers) {
        return parallel::detail::run_safe([&]() -> parallel::Result<void> {
            threads_.reserve(background_workers);
            for (std::size_t i = 0; i < background_workers; ++i) {
                threads_.emplace_back([this] { worker(); });
            }
            return outcome::success();
        });
    }

    void run(std::size_t size, std::size_t grain) {
        std::unique_lock<std::mutex> lock(mutex_);
        next_ = 0;
        size_ = size;
        grain_ = grain;
        remaining_ = size / grain + (size % grain != 0);
        ready_.notify_all();
        while (next_ < size_) {
            run_chunk(lock);
        }
        finished_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    void worker() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [this] { return stopping_ || next_ < size_; });
            if (stopping_) {
                return;
            }
            run_chunk(lock);
        }
    }

    void run_chunk(std::unique_lock<std::mutex>& lock) {
        const auto first = next_;
        const auto last = first + std::min(grain_, size_ - first);
        next_ = last;
        lock.unlock();
        static_cast<void>(work_(first, last));  // The benchmark workload always succeeds.
        lock.lock();
        if (--remaining_ == 0) {
            finished_.notify_one();
        }
    }

    const ChunkWork& work_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::vector<std::thread> threads_;
    std::size_t next_ = 0;
    std::size_t size_ = 0;
    std::size_t grain_ = 1;
    std::size_t remaining_ = 0;
    bool stopping_ = false;
};

enum class Scheduler {
    pipeline_collect,
    fresh_dynamic_chunks,
    reused_dynamic_chunks,
    fresh_mutex_pool,
    reused_mutex_pool,
};

struct Workload {
    const char* name;
    std::array<int, 2> batch_sizes;
    unsigned rounds;
    unsigned heavy_rounds;
};

std::uint64_t compute(std::size_t index, unsigned rounds) noexcept {
    auto value = static_cast<std::uint64_t>(index) + 1;
    for (unsigned round = 0; round < rounds; ++round) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        value *= UINT64_C(2685821657736338717);
    }
    return value;
}

template<class T>
bool succeeded(benchmark::State& state, const parallel::Result<T>& result) {
    if (result) {
        return true;
    }
    state.SkipWithError(result.error().message().c_str());
    return false;
}

void compare_schedulers(benchmark::State& state, Scheduler scheduler, Workload workload) {
    const auto size = static_cast<std::size_t>(state.range(0));
    const auto workers = static_cast<std::size_t>(state.range(1));
    // Match the executor's grain so all schedulers distribute the same chunks.
    const auto grain = std::max(std::size_t{1}, std::min(std::size_t{64}, size / workers / 8));
    const auto heavy_end = workload.heavy_rounds == 0 ? 0 : size / 16;
    const auto calculate = [&](std::size_t index) {
        return compute(index, index < heavy_end ? workload.heavy_rounds : workload.rounds);
    };
    // Build the lazy plan before timing its collection.
    using Index = parallel::Range::value_type;
    const auto pipeline = parallel::Range(static_cast<Index>(size)).into_inter().iter_paral(workers)
        .map([&](const Index& index) { return calculate(static_cast<std::size_t>(index)); });
    std::vector<std::uint64_t> output(scheduler == Scheduler::pipeline_collect ? 0 : size);
    const ChunkWork work = [&](std::size_t first, std::size_t last) -> parallel::Result<void> {
        for (auto i = first; i < last; ++i) {
            output[i] = calculate(i);
        }
        return outcome::success();
    };

    Executor dynamic(workers);
    MutexPool locked(work);
    if (scheduler == Scheduler::reused_dynamic_chunks &&
        !succeeded(state, dynamic.run_chunks(size, [](std::size_t, std::size_t) -> parallel::Result<void> {
            return outcome::success();
        }))) {
        return;
    }
    if (scheduler == Scheduler::reused_mutex_pool && !succeeded(state, locked.start(workers - 1))) {
        return;
    }

    for (auto iteration : state) {
        static_cast<void>(iteration);
        switch (scheduler) {
        case Scheduler::pipeline_collect: {
            auto result = pipeline.collect();
            if (!succeeded(state, result)) {
                return;
            }
            output = std::move(result).assume_value();
            break;
        }
        case Scheduler::fresh_dynamic_chunks: {
            Executor executor(workers);
            if (!succeeded(state, executor.run_chunks(size, work))) {
                return;
            }
            break;
        }
        case Scheduler::reused_dynamic_chunks:
            if (!succeeded(state, dynamic.run_chunks(size, work))) {
                return;
            }
            break;
        case Scheduler::fresh_mutex_pool: {
            MutexPool pool(work);
            if (!succeeded(state, pool.start(workers - 1))) {
                return;
            }
            pool.run(size, grain);
            break;
        }
        case Scheduler::reused_mutex_pool:
            locked.run(size, grain);
            break;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    if (output.size() != size) {
        state.SkipWithError("scheduler produced the wrong number of results");
        return;
    }
    for (std::size_t i = 0; i < size; ++i) {
        if (output[i] != calculate(i)) {
            state.SkipWithError("scheduler produced an incorrect result");
            return;
        }
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

const bool cases_registered = [] {
    const struct {
        const char* name;
        Scheduler type;
    } schedulers[] = {
        {"PipelineCollect", Scheduler::pipeline_collect},
        {"FreshDynamicChunks", Scheduler::fresh_dynamic_chunks},
        {"ReusedDynamicChunks", Scheduler::reused_dynamic_chunks},
        {"FreshMutexPool", Scheduler::fresh_mutex_pool},
        {"ReusedMutexPool", Scheduler::reused_mutex_pool},
    };
    const Workload workloads[] = {
        {"Uniform", {4096, 65536}, 24, 0},
        {"Uneven", {4096, 65536}, 8, 256},
        {"ShortTasks", {128, 4096}, 4, 0},
        {"LongTasks", {8, 32}, 1u << 20, 0},
    };
    for (const auto& scheduler : schedulers) {
        for (const auto& workload : workloads) {
            const auto name = std::string("compare_schedulers/") + scheduler.name + '/' + workload.name;
            auto* test = benchmark::RegisterBenchmark(name.c_str(), compare_schedulers, scheduler.type, workload);
            test->ArgNames({"tasks", "workers"})->UseRealTime()->Unit(benchmark::kMicrosecond);
            for (int size : workload.batch_sizes) {
                for (int workers : {2, 4, 8}) {
                    test->Args({size, workers});
                }
            }
        }
    }
    return true;
}();

}  // namespace
