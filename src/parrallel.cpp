#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace parallel {
namespace {

class ParallelErrorCategory : public std::error_category {
public:
    const char* name() const noexcept override { return "parallel"; }

    std::string message(int error) const override {
        if (error == static_cast<int>(Errc::unexpected_exception)) {
            return "unexpected exception during pipeline execution";
        }
        return "unknown parallel error";
    }
};

}  // namespace

ErrorCode make_error_code(Errc error) noexcept {
    static const ParallelErrorCategory category;
    return {static_cast<int>(error), category};
}

namespace detail {

namespace {

thread_local bool may_cache_team = true;

class Batch {
public:
    Batch(std::size_t size, std::size_t workers, const Executor::ChunkTask& run)
        : size_(size), grain_(std::max(std::size_t{1}, std::min(std::size_t{64}, size / workers / 8))),
          tail_begin_(grain_ >= 16 ? size - grain_ - size % grain_ : size), run_(run) {}

    bool has_work() const noexcept {
        return !cancelled_.load(std::memory_order_relaxed) && next_.load(std::memory_order_relaxed) < size_;
    }

    bool has_single_item_chunks() const noexcept { return grain_ == 1; }

    void work() noexcept {
        const auto status = run_safe([&]() -> Result<void> {
            const auto prefix = consume(tail_begin_, grain_);
            if (!prefix || tail_begin_ == size_) {
                return prefix;
            }
            // Workers can share the final full chunk and any remainder.
            return consume(size_, grain_ / 4);
        });
        if (!status && !cancelled_.exchange(true, std::memory_order_relaxed)) {
            error_ = status.assume_error();
        }
    }

    // The caller reads the error after every participating worker finishes.
    Result<void> result() const {
        if (cancelled_.load(std::memory_order_relaxed)) {
            return outcome::failure(error_);
        }
        return outcome::success();
    }

private:
    Result<void> consume(std::size_t end, std::size_t grain) {
        while (!cancelled_.load(std::memory_order_relaxed)) {
            auto first = next_.load(std::memory_order_relaxed);
            if (first >= end) {
                break;
            }
            const auto last = first + std::min(grain, end - first);
            if (next_.compare_exchange_weak(first, last, std::memory_order_relaxed) &&
                !cancelled_.load(std::memory_order_relaxed)) {
                const auto chunk = run_(first, last);
                if (!chunk) {
                    return chunk.as_failure();
                }
            }
        }
        return outcome::success();
    }

    const std::size_t size_;
    const std::size_t grain_;
    const std::size_t tail_begin_;
    const Executor::ChunkTask& run_;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> cancelled_{false};
    ErrorCode error_;
};

}  // namespace

struct Executor::Team {
    ~Team() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    Result<void> run(Batch& task, std::size_t background_workers) {
        // Grow the pool before publishing work. A failed launch runs no chunk of this batch.
        if (threads.size() < background_workers) {
            threads.reserve(background_workers);
            while (threads.size() < background_workers) {
                threads.emplace_back([this] { worker(); });
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            batch = &task;
            worker_budget = background_workers;
        }
        if (background_workers > 1 && task.has_single_item_chunks()) {
            ready.notify_all();
        } else {
            ready.notify_one();
        }
        task.work();
        {
            std::unique_lock<std::mutex> lock(mutex);
            // Only workers that took the batch can still be using it.
            batch = nullptr;
            finished.wait(lock, [&] { return active == 0; });
        }
        return task.result();
    }

private:
    void worker() {
        may_cache_team = false;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] {
                return stopping || (batch && active < worker_budget && batch->has_work());
            });
            if (stopping) {
                return;
            }
            auto* task = batch;
            ++active;
            const bool wake_next = active < worker_budget && !task->has_single_item_chunks();
            lock.unlock();
            if (wake_next && task->has_work()) {
                ready.notify_one();
            }
            task->work();
            lock.lock();
            if (--active == 0 && !batch) {
                finished.notify_one();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable finished;
    std::vector<std::thread> threads;
    Batch* batch = nullptr;
    std::size_t worker_budget = 0;
    std::size_t active = 0;
    bool stopping = false;
};

struct Executor::TeamCache {
    ~TeamCache() { may_cache_team = false; }

    std::unique_ptr<Team> team;
    std::size_t worker_limit = 0;
    bool leased = false;
};

Executor::Executor(std::size_t workers, Reuse reuse) noexcept
    : worker_limit_(workers == 0 ? std::max(1u, std::thread::hardware_concurrency()) : workers),
      reuse_(reuse) {}

Executor::~Executor() {
    if (cache_) {
        cache_->team = std::move(team_);
        cache_->leased = false;
    }
}

void Executor::acquire_team() {
    // Leave the cache with its current evaluation during a nested collection.
    if (reuse_ == Reuse::calling_thread && may_cache_team) {
        static thread_local TeamCache local;
        if (!local.leased) {
            if (local.worker_limit != worker_limit_) {
                local.team.reset();
                local.worker_limit = worker_limit_;
            }
            local.leased = true;
            cache_ = &local;
            team_ = std::move(local.team);
        }
    }
    if (!team_) {
        team_ = std::make_unique<Team>();
    }
}

Result<void> Executor::run_chunks(std::size_t size, const ChunkTask& run) {
    if (size == 0) {
        return outcome::success();
    }
    const auto workers = std::min(worker_limit_, size);
    return run_safe([&]() -> Result<void> {
        if (workers == 1) {
            return run(0, size);
        }
        if (!team_) {
            acquire_team();
        }
        Batch batch(size, workers, run);
        return team_->run(batch, workers - 1);
    });
}

}  // namespace detail
}  // namespace parallel
