#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
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

std::size_t divide_up(std::size_t count, std::size_t grain) noexcept {
    return count / grain + (count % grain != 0);
}

std::size_t default_grain(std::size_t size, std::size_t workers) noexcept {
    const auto per_worker = size / workers;
    // Keep small batches fine-grained and grow the prefix grain for large batches.
    return std::max({std::size_t{1}, std::min(std::size_t{64}, per_worker / 8), per_worker / 16});
}

}  // namespace

struct Executor::Batch {
    struct Queue {
        std::atomic<std::size_t> next{0};
        std::size_t first = 0;
        std::size_t count = 0;
        // Keep queue counters on separate cache lines even on C++14 heaps.
        unsigned char padding[128 - sizeof(next) - sizeof(first) - sizeof(count)];
    };

    Batch(std::size_t size, std::size_t workers, ChunkCallback run, std::size_t grain_hint, Schedule schedule)
        : size_(size),
          workers_(workers),
          grain_(grain_hint ? std::min(grain_hint, size)
                            : (schedule == Schedule::static_partition ? size : default_grain(size, workers))),
          tail_base_(grain_hint ? grain_ : std::min(std::size_t{64}, grain_)),
          tail_begin_(tail_base_ >= 16 ? size - tail_base_ - size % tail_base_ : size),
          tail_grain_(std::max(std::size_t{1}, tail_base_ / (grain_hint ? 4 : std::max(std::size_t{4}, workers)))),
          prefix_chunks_(divide_up(tail_begin_, grain_)),
          chunks_(prefix_chunks_ + divide_up(size - tail_begin_, tail_grain_)),
          run_(run),
          static_(schedule == Schedule::static_partition),
          stealing_(schedule == Schedule::work_stealing && workers >= 4 &&
                    size / workers >= 4096 && prefix_chunks_ / workers >= 4) {}

    bool is_static() const noexcept { return static_; }
    bool uses_stealing() const noexcept { return stealing_; }

    void set_queues(Queue* queues) noexcept {
        queues_ = queues;
        // Dense queues benefit from contiguous memory access. For coarser
        // work, round-robin placement spreads neighboring expensive chunks.
        const bool contiguous = prefix_chunks_ / workers_ >= 32;
        const bool share_tail = contiguous || chunks_ - prefix_chunks_ >= 8;
        const auto local_chunks = share_tail ? prefix_chunks_ : chunks_;
        queue_stride_ = contiguous ? 1 : workers_;
        const auto per_worker = local_chunks / workers_;
        const auto extra = local_chunks % workers_;
        for (std::size_t i = 0; i < workers_; ++i) {
            queues[i].next.store(0, std::memory_order_relaxed);
            queues[i].first = contiguous ? i * per_worker + std::min(i, extra) : i;
            queues[i].count = per_worker + (i < extra);
        }
        // A shared tail only becomes eligible once the prefix has been claimed.
        cursor_.next.store(local_chunks, std::memory_order_relaxed);
    }

    bool defer_peer_wakeups() const noexcept { return !static_ && size_ < 8192 && grain_ > 1; }

    std::size_t claimed() const noexcept { return cursor_.next.load(std::memory_order_relaxed); }

    bool has_work() const noexcept {
        return !cancelled_.load(std::memory_order_relaxed) && (stealing_ || claimed() < chunks_);
    }

    void work(std::size_t worker) noexcept {
        const auto status = run_safe([&]() -> Result<void> {
            ErrorCode error;
            if (run_work(worker, error)) {
                return outcome::success();
            }
            return outcome::failure(error);
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
    bool run_work(std::size_t worker, ErrorCode& error) {
        if (static_) {
            return run_static(worker, error);
        }
        if (stealing_) {
            return run_stealing(worker, error);
        }
        return run_dynamic(error);
    }

    bool drain_queue(std::size_t victim, ErrorCode& error) {
        auto& queue = queues_[victim];
        while (!cancelled_.load(std::memory_order_relaxed)) {
            const auto next = queue.next.fetch_add(1, std::memory_order_relaxed);
            if (next >= queue.count || cancelled_.load(std::memory_order_relaxed)) {
                break;
            }
            if (!run_dynamic_chunk(queue.first + next * queue_stride_, error)) {
                return false;
            }
        }
        return true;
    }

    bool run_stealing(std::size_t worker, ErrorCode& error) {
        if (!drain_queue(worker, error)) {
            return false;
        }
        // Queues are populated before publication and only shrink. A single
        // pass after the local queue is enough to find every unclaimed chunk.
        auto victim = worker;
        for (std::size_t visited = 1; visited < workers_; ++visited) {
            if (++victim == workers_) {
                victim = 0;
            }
            auto& queue = queues_[victim];
            if (queue.next.load(std::memory_order_relaxed) < queue.count && !drain_queue(victim, error)) {
                return false;
            }
        }
        return claimed() >= chunks_ || run_dynamic(error);
    }

    bool run_static(std::size_t worker, ErrorCode& error) const {
        const auto width = size_ / workers_;
        const auto extra = size_ % workers_;
        auto first = worker * width + std::min(worker, extra);
        const auto end = first + width + (worker < extra);
        while (first < end && !cancelled_.load(std::memory_order_relaxed)) {
            const auto last = first + std::min(grain_, end - first);
            if (!run_(first, last, error)) {
                return false;
            }
            first = last;
        }
        return true;
    }

    std::size_t take_bounded() noexcept {
        auto chunk = claimed();
        while (chunk < chunks_ && !cursor_.next.compare_exchange_weak(chunk, chunk + 1, std::memory_order_relaxed)) {
        }
        return chunk;
    }

    bool run_dynamic(ErrorCode& error) {
        // Each worker can claim one ticket past the end. Bound the counter for
        // the exceptional case of a range whose chunk count is near SIZE_MAX.
        const bool bounded = chunks_ > std::numeric_limits<std::size_t>::max() - workers_;
        while (!cancelled_.load(std::memory_order_relaxed)) {
            const auto chunk = bounded ? take_bounded() : cursor_.next.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= chunks_ || cancelled_.load(std::memory_order_relaxed)) {
                break;
            }
            if (!run_dynamic_chunk(chunk, error)) {
                return false;
            }
        }
        return true;
    }

    bool run_dynamic_chunk(std::size_t chunk, ErrorCode& error) const {
        if (chunk < prefix_chunks_) {
            const auto first = chunk * grain_;
            return run_(first, first + std::min(grain_, tail_begin_ - first), error);
        }
        // Keep the tail fine-grained even when the default prefix grain grows.
        const auto first = tail_begin_ + (chunk - prefix_chunks_) * tail_grain_;
        return run_(first, first + std::min(tail_grain_, size_ - first), error);
    }

    const std::size_t size_;
    const std::size_t workers_;
    const std::size_t grain_;
    const std::size_t tail_base_;
    const std::size_t tail_begin_;
    const std::size_t tail_grain_;
    const std::size_t prefix_chunks_;
    const std::size_t chunks_;
    const ChunkCallback run_;
    Queue* queues_ = nullptr;
    std::size_t queue_stride_ = 0;
    ErrorCode error_;
    const bool static_;
    const bool stealing_;
    std::atomic<bool> cancelled_{false};
    // Keep the frequently updated ticket away from read-mostly batch state.
    struct alignas(128) ChunkCursor {
        std::atomic<std::size_t> next{0};
    } cursor_;
};

struct Executor::Team {
    ~Team() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        for (const auto& worker : workers) {
            worker->ready.notify_one();
        }
        for (const auto& worker : workers) {
            worker->thread.join();
        }
    }

    std::size_t size() const noexcept { return workers.size(); }

    Result<void> run(Batch& task, std::size_t background_workers) {
        // Grow the pool before publishing work. A failed launch runs no chunk of this batch.
        grow(background_workers);
        if (task.uses_stealing()) {
            const auto count = background_workers + 1;
            if (queue_capacity < count) {
                queues = std::make_unique<Batch::Queue[]>(count);
                queue_capacity = count;
            }
            task.set_queues(queues.get());
        }
        const auto initial = task.defer_peer_wakeups() ? std::size_t{1} : background_workers;
        {
            std::lock_guard<std::mutex> lock(mutex);
            worker_budget = background_workers;
            published = initial;
            last_claim = 0;
            remaining = initial;
            publish(task, 0, initial);
        }
        wake(0, initial);
        task.work(0);
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!task.is_static()) {
                // The caller has drained the queue. Workers that have yet to
                // take this batch can stay idle, even if another worker is busy.
                worker_budget = published;
                for (std::size_t i = 0; i < published; ++i) {
                    if (workers[i]->pending) {
                        workers[i]->pending = nullptr;
                        --remaining;
                    }
                }
            }
            finished.wait(lock, [&] { return remaining == 0; });
        }
        return task.result();
    }

private:
    struct Worker {
        std::condition_variable ready;
        Batch* pending = nullptr;
        std::thread thread;
    };

    void grow(std::size_t count) {
        if (workers.size() >= count) {
            return;
        }
        workers.reserve(count);
        while (workers.size() < count) {
            auto worker = std::make_unique<Worker>();
            const auto index = workers.size() + 1;
            worker->thread = std::thread([this, slot = worker.get(), index] { run_worker(*slot, index); });
            workers.push_back(std::move(worker));
        }
    }

    void publish(Batch& task, std::size_t first, std::size_t end) noexcept {
        for (auto i = first; i < end; ++i) {
            workers[i]->pending = &task;
        }
    }

    void wake(std::size_t first, std::size_t end) noexcept {
        for (auto i = first; i < end; ++i) {
            workers[i]->ready.notify_one();
        }
    }

    void run_worker(Worker& worker, std::size_t index) {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            worker.ready.wait(lock, [&] { return stopping || worker.pending; });
            if (stopping) {
                return;
            }
            auto* task = std::exchange(worker.pending, nullptr);
            if (task->is_static() || task->has_work()) {
                const auto first = published;
                if (published < worker_budget) {
                    // Double participation while chunks are being claimed quickly.
                    // Slow progress releases all remaining workers immediately.
                    const auto claimed = task->claimed();
                    const auto available = worker_budget - published;
                    const auto count = claimed - last_claim >= 2 ? std::min(available, published) : available;
                    last_claim = claimed;
                    remaining += count;
                    published += count;
                    publish(*task, first, published);
                }
                const auto end = published;
                lock.unlock();
                wake(first, end);
                task->work(index);
                lock.lock();
            }
            if (--remaining == 0) {
                finished.notify_one();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable finished;
    std::vector<std::unique_ptr<Worker>> workers;
    std::unique_ptr<Batch::Queue[]> queues;
    std::size_t queue_capacity = 0;
    std::size_t worker_budget = 0;
    std::size_t published = 0;
    std::size_t last_claim = 0;
    std::size_t remaining = 0;
    bool stopping = false;
};

struct Executor::TeamCache {
    TeamCache() { idle.reserve(max_idle_teams); }

    static std::shared_ptr<TeamCache> shared() {
        static std::atomic<bool> shutting_down{false};
        struct Owner {
            std::shared_ptr<TeamCache> cache = std::make_shared<TeamCache>();
            ~Owner() { shutting_down.store(true, std::memory_order_relaxed); }
        };
        // A later static destructor can still run an evaluation with its own team.
        if (shutting_down.load(std::memory_order_relaxed)) {
            return {};
        }
        static const Owner owner;
        return owner.cache;
    }

    std::unique_ptr<Team> take(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        if (idle.empty()) {
            return nullptr;
        }
        auto best = idle.rbegin();
        for (auto it = std::next(best); it != idle.rend(); ++it) {
            const auto size = (*it)->size();
            const auto current = (*best)->size();
            if (current < count ? size > current : (size >= count && size < current)) {
                best = it;
            }
        }
        auto team = std::move(*best);
        idle.erase(std::prev(best.base()));
        return team;
    }

    void put(std::unique_ptr<Team> team) {
        std::unique_ptr<Team> retired;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (idle.size() == max_idle_teams) {
                retired = std::move(idle.front());
                idle.erase(idle.begin());
            }
            idle.push_back(std::move(team));
        }
    }

    static constexpr std::size_t max_idle_teams = 8;
    std::mutex mutex;
    std::vector<std::unique_ptr<Team>> idle;
};

Executor::Executor(std::size_t workers, Reuse reuse) noexcept
    : Executor(workers, Schedule::dynamic, reuse) {}

Executor::Executor(std::size_t workers, Schedule schedule, Reuse reuse) noexcept
    : worker_limit_(workers == 0 ? std::max(1u, std::thread::hardware_concurrency()) : workers),
      reuse_(reuse), schedule_(schedule) {}

Executor::~Executor() {
    if (cache_ && team_ && team_->size() != 0) {
        cache_->put(std::move(team_));
    }
}

void Executor::acquire_team(std::size_t background_workers) {
    if (reuse_ == Reuse::shared) {
        cache_ = TeamCache::shared();
        if (cache_) {
            team_ = cache_->take(background_workers);
        }
    }
    if (!team_) {
        team_ = std::make_unique<Team>();
    }
}

Result<void> Executor::run_chunks(std::size_t size, const ChunkTask& run, std::size_t grain_hint) {
    return run_borrowed_chunks(size, run, grain_hint);
}

Result<void> Executor::submit(std::size_t size, ChunkCallback run, std::size_t grain_hint) {
    if (size == 0) {
        return outcome::success();
    }
    const auto workers = std::min(worker_limit_, size);
    return run_safe([&]() -> Result<void> {
        if (workers == 1) {
            ErrorCode error;
            if (!run(0, size, error)) {
                return outcome::failure(error);
            }
            return outcome::success();
        }
        if (!team_) {
            acquire_team(workers - 1);
        }
        Batch batch(size, workers, run, grain_hint, schedule_);
        return team_->run(batch, workers - 1);
    });
}

}  // namespace detail
}  // namespace parallel
