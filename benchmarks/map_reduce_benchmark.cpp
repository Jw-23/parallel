#include "parallel.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

namespace outcome = boost::outcome_v2;

struct Request {
    std::uint32_t latency_us;
    std::uint32_t response_bytes;
    std::uint16_t status;
    std::uint8_t retries;
};

std::vector<Request> make_requests(std::size_t count) {
    std::vector<Request> requests;
    requests.reserve(count);
    std::uint64_t random = 0xd312f68a547ce209ULL;
    for (std::size_t i = 0; i < count; ++i) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        const auto tail = (random & 31u) == 0;
        const auto status = (random & 63u) == 1 ? std::uint16_t{503} : std::uint16_t{200};
        requests.push_back({
            static_cast<std::uint32_t>(200 + (random >> 8) % (tail ? 600000 : 18000)),
            static_cast<std::uint32_t>(256 + (random >> 32) % 65536),
            status,
            static_cast<std::uint8_t>(status == 503 ? (random >> 6) % 4 : 0),
        });
    }
    return requests;
}

struct NormalizedRequest {
    std::uint32_t latency_us;
    std::uint32_t response_bytes;
    bool failed;
};

NormalizedRequest normalize(const Request& request) noexcept {
    const auto latency = request.latency_us * (std::uint32_t{1} + request.retries);
    return {std::min(latency, std::uint32_t{2000000}), request.response_bytes, request.status >= 500};
}

constexpr std::array<std::uint32_t, 7> latency_limits{{500, 1000, 2500, 5000, 10000, 25000, 100000}};

struct ClassifiedRequest {
    NormalizedRequest request;
    std::size_t latency_bucket;
    std::uint64_t penalty;
};

ClassifiedRequest classify(const NormalizedRequest& request) noexcept {
    const auto bucket = std::lower_bound(latency_limits.begin(), latency_limits.end(), request.latency_us);
    const auto over_budget = request.latency_us - std::min(request.latency_us, std::uint32_t{10000});
    const auto penalty = std::uint64_t{over_budget} * (request.failed ? 8u : 1u);
    return {request, static_cast<std::size_t>(bucket - latency_limits.begin()), penalty};
}

struct RequestSummary {
    std::array<std::uint64_t, latency_limits.size() + 1> latency_histogram{};
    std::uint64_t failures = 0;
    std::uint64_t response_bytes = 0;
    std::uint64_t latency_sum = 0;
    std::uint64_t latency_squared_sum = 0;
    std::uint64_t penalty = 0;
    std::uint32_t max_latency = 0;
};

bool operator==(const RequestSummary& a, const RequestSummary& b) noexcept {
    return a.latency_histogram == b.latency_histogram && a.failures == b.failures &&
           a.response_bytes == b.response_bytes && a.latency_sum == b.latency_sum &&
           a.latency_squared_sum == b.latency_squared_sum && a.penalty == b.penalty &&
           a.max_latency == b.max_latency;
}

void add_request(RequestSummary& total, const ClassifiedRequest& item) noexcept {
    const auto latency = std::uint64_t{item.request.latency_us};
    ++total.latency_histogram[item.latency_bucket];
    total.failures += item.request.failed;
    total.response_bytes += item.request.response_bytes;
    total.latency_sum += latency;
    total.latency_squared_sum += latency * latency;
    total.penalty += item.penalty;
    total.max_latency = std::max(total.max_latency, item.request.latency_us);
}

RequestSummary summarize_request(const ClassifiedRequest& item) noexcept {
    RequestSummary contribution;
    add_request(contribution, item);
    return contribution;
}

RequestSummary merge(RequestSummary total, const RequestSummary& part) noexcept {
    for (std::size_t i = 0; i < total.latency_histogram.size(); ++i) {
        total.latency_histogram[i] += part.latency_histogram[i];
    }
    total.failures += part.failures;
    total.response_bytes += part.response_bytes;
    total.latency_sum += part.latency_sum;
    total.latency_squared_sum += part.latency_squared_sum;
    total.penalty += part.penalty;
    total.max_latency = std::max(total.max_latency, part.max_latency);
    return total;
}

void add_directly(RequestSummary& total, const Request& request) noexcept {
    add_request(total, classify(normalize(request)));
}

void map_and_merge(RequestSummary& total, const Request& request) noexcept {
    total = merge(std::move(total), summarize_request(classify(normalize(request))));
}

template<void (*Add)(RequestSummary&, const Request&)>
parallel::Result<RequestSummary> handwritten_summary(const std::vector<Request>& requests,
                                                     parallel::Executor& executor) {
    const auto count = requests.size();
    const auto workers = executor.worker_limit();
    const auto fold = [&](std::size_t first, std::size_t last) {
        RequestSummary total;
        for (auto i = first; i < last; ++i) {
            Add(total, requests[i]);
        }
        return total;
    };
    if (workers == 1 || count <= 1) {
        return outcome::success(fold(0, count));
    }

    // Match Pipeline::reduce's block size and use the same warmed executor.
    const auto block_size = std::max(std::size_t{2}, count / std::min(workers, count) / 8);
    const auto blocks = count / block_size + (count % block_size != 0);
    std::vector<RequestSummary> partials(blocks);
    const auto completed = executor.run_borrowed_chunks(blocks, [&](std::size_t first, std::size_t last) {
        for (auto block = first; block < last; ++block) {
            const auto begin = block * block_size;
            partials[block] = fold(begin, begin + std::min(block_size, count - begin));
        }
    });
    if (!completed) {
        return completed.as_failure();
    }
    RequestSummary total;
    for (const auto& part : partials) {
        total = merge(std::move(total), part);
    }
    return outcome::success(total);
}

template<class Run>
void measure(benchmark::State& state, const std::vector<Request>& requests, const Run& run) {
    RequestSummary expected;
    for (const auto& request : requests) {
        add_directly(expected, request);
    }
    const auto warm = run();
    if (!warm) {
        state.SkipWithError(warm.error().message().c_str());
        return;
    }
    if (!(warm.assume_value() == expected)) {
        state.SkipWithError("MapReduce summary differs from the serial reference");
        return;
    }

    auto result = warm;
    for (auto iteration : state) {
        static_cast<void>(iteration);
        const auto* input = requests.data();
        benchmark::DoNotOptimize(input);
        result = run();
        if (!result) {
            state.SkipWithError(result.error().message().c_str());
            return;
        }
        benchmark::DoNotOptimize(result.assume_value());
        benchmark::ClobberMemory();
    }
    if (!(result.assume_value() == expected)) {
        state.SkipWithError("MapReduce summary differs from the serial reference");
        return;
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

void pipeline_map_reduce(benchmark::State& state) {
    const auto requests = make_requests(static_cast<std::size_t>(state.range(0)));
    const auto workers = static_cast<std::size_t>(state.range(1));
    parallel::Executor executor(workers);
    const auto plan = parallel::iter_paral(requests, workers)
        .map([](const Request& request) { return normalize(request); })
        .map([](const NormalizedRequest& request) { return classify(request); })
        .map([](const ClassifiedRequest& request) { return summarize_request(request); })
        .reduce(RequestSummary{}, [](RequestSummary total, const RequestSummary& part) {
            return merge(std::move(total), part);
        });
    measure(state, requests, [&] { return plan.collect_one(executor); });
}

template<void (*Add)(RequestSummary&, const Request&)>
void handwritten_map_reduce(benchmark::State& state) {
    const auto requests = make_requests(static_cast<std::size_t>(state.range(0)));
    parallel::Executor executor(static_cast<std::size_t>(state.range(1)));
    measure(state, requests, [&] { return handwritten_summary<Add>(requests, executor); });
}

void map_reduce_sizes(benchmark::internal::Benchmark* test) {
    test->ArgNames({"requests", "workers"})->UseRealTime()->Unit(benchmark::kMicrosecond);
    for (int size : {65536, 1048576}) {
        for (int workers : {1, 4, 8}) {
            test->Args({size, workers});
        }
    }
}

BENCHMARK(pipeline_map_reduce)->Name("map_reduce/Pipeline")->Apply(map_reduce_sizes);
BENCHMARK_TEMPLATE(handwritten_map_reduce, map_and_merge)->Name("map_reduce/HandwrittenSameReducer")
    ->Apply(map_reduce_sizes);
BENCHMARK_TEMPLATE(handwritten_map_reduce, add_directly)->Name("map_reduce/HandwrittenDirect")
    ->Apply(map_reduce_sizes);

}  // namespace
