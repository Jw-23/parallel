#include "parallel.hpp"

#include <gtest/gtest.h>

#ifdef PARALLEL_TEST_BOOST_CONTAINER
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using parallel::Range;
using parallel::Result;
using parallel::into_inter;
using parallel::into_inter_paral;
using parallel::iter;
using parallel::iter_paral;
namespace outcome = boost::outcome_v2;

template<class T, class Expected>
void expect_value(const Result<T>& result, const Expected& expected) {
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result.value(), expected);
}

template<class T, class Error>
void expect_error(const Result<T>& result, const Error& error) {
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), error);
}

template<class T, class = void>
struct SupportsIter : std::false_type {};

template<class T>
struct SupportsIter<T, parallel::detail::void_t<decltype(iter(std::declval<T>()))>> : std::true_type {};

template<class T, class = void>
struct SupportsIntoInter : std::false_type {};

template<class T>
struct SupportsIntoInter<T, parallel::detail::void_t<decltype(into_inter(std::declval<T>()))>> : std::true_type {};

struct MissingEnd {
    int* begin() const;
};

struct AdlRange {
    int values[3] = {4, 5, 6};
};

struct Sentinel {
    const int* end;
};

const int* begin(const AdlRange& range) { return range.values; }
Sentinel end(const AdlRange& range) { return {range.values + 3}; }
bool operator!=(const int* position, Sentinel sentinel) { return position != sentinel.end; }

static_assert(!SupportsIter<int>::value, "scalar values are not iterable");
static_assert(!SupportsIter<MissingEnd>::value, "iterability requires an end");
static_assert(SupportsIter<AdlRange&>::value, "ADL begin and end are supported");
static_assert(!SupportsIter<AdlRange>::value, "the borrowing entry point requires an lvalue");
static_assert(!SupportsIntoInter<std::vector<int>&>::value, "ownership transfer requires an rvalue");
static_assert(!SupportsIntoInter<int*>::value, "raw pointers are borrowed");
static_assert(std::is_same<parallel::ErrorCode, std::error_code>::value, "use the standard error code");

TEST(RangeTest, SupportsDescendingAndEmptyRanges) {
    const Range down(5, -2, -2);
    const std::vector<std::int64_t> expected{5, 3, 1, -1};
    EXPECT_EQ(std::vector<std::int64_t>(down.begin(), down.end()), expected);
    expect_value(down.iter_paral(3).collect(), expected);
    expect_value(Range(1, 4, -1).iter().collect(), std::vector<std::int64_t>{});
}

TEST(RangeTest, HandlesExtremeStridesAndRejectsZero) {
    const auto low = std::numeric_limits<std::int64_t>::min();
    const auto high = std::numeric_limits<std::int64_t>::max();
    expect_value(Range(low, high, high).iter().collect(), std::vector<std::int64_t>{low, -1, high - 1});
    expect_value(Range(high, low, low).iter().collect(), std::vector<std::int64_t>{high, -1});

    const Range invalid(0, 1, 0);
    EXPECT_EQ(invalid.error(), std::errc::invalid_argument);
    expect_error(invalid.iter().collect(), std::errc::invalid_argument);
}

TEST(PipelineTest, SupportsAdlSentinelsAndPackedBooleanOutput) {
    expect_value(into_inter(AdlRange{}).collect(), std::vector<int>{4, 5, 6});
    const std::vector<bool> flags{true, false, true};
    expect_value(iter_paral(flags, 2).map(std::logical_not<>{}).collect(),
                 std::vector<bool>{false, true, false});
}

TEST(PipelineTest, EvaluatesOnTraversalOrCollection) {
    std::vector<int> input{1, 2, 3, 4};
    const auto caller = std::this_thread::get_id();
    int calls = 0;
    const auto mapped = iter(input).map([&](const int& value) {
        EXPECT_EQ(std::this_thread::get_id(), caller);
        ++calls;
        return value * 3;
    });
    const auto plan = mapped.filter([](const int& value) { return value % 2 == 0; })
                            .zip(std::vector<int>{10}, std::plus<>{})
                            .reduce(1, std::plus<>{})
                            .map([](const int& value) { return std::to_string(value); });
    const auto first = plan.begin();
    const auto last = plan.end();
    EXPECT_EQ(calls, 0);
    input[1] = 6;
    ASSERT_NE(first, last);
    EXPECT_EQ(*first, "29");
    EXPECT_EQ(calls, 4);
    EXPECT_EQ(std::next(first), last);
    EXPECT_EQ(calls, 4);
    expect_value(plan.collect_one(), std::string("29"));
    EXPECT_EQ(calls, 8);
    EXPECT_EQ(input, (std::vector<int>{1, 6, 3, 4}));

    expect_value(mapped.zip(Range(2).into_inter()).collect(),
                 std::vector<std::pair<int, std::int64_t>>{{3, 0}, {18, 1}});
}

TEST(PipelineTest, MaterializeCompletesEachStageBeforeTheNext) {
    using Value = Range::value_type;
    constexpr Value size = 1000;
    std::array<std::atomic<Value>, 4> finished;
    for (auto& count : finished) {
        count.store(0, std::memory_order_relaxed);
    }
    const auto stage = [&](std::size_t index) {
        return [&, index](Value value) {
            if (index != 0) {
                EXPECT_EQ(finished[index - 1].load(std::memory_order_relaxed), size);
            }
            finished[index].fetch_add(1, std::memory_order_relaxed);
            return value + 1;
        };
    };

    const auto result = Range(size).iter_paral(4)
        .map(stage(0)).materialize()
        .map(stage(1)).materialize()
        .map(stage(2)).materialize()
        .map(stage(3)).collect();
    const Range expected(4, size + 4);
    expect_value(result, std::vector<Value>(expected.begin(), expected.end()));
}

TEST(PipelineTest, CollectsMoveOnlyOutput) {
    const auto plan = into_inter(std::vector<int>{3}).map([](const int& value) {
        return std::make_unique<int>(value);
    });
    const auto result = plan.collect_one();
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_NE(result.value(), nullptr);
    EXPECT_EQ(*result.value(), 3);
}

TEST(PipelineTest, FusesFourMapsAcrossDifferentTypes) {
    std::vector<int> input{1, 2, 3, 4};
    const auto plan = iter_paral(input, 3)
        .map([](const int& value) { return value * 2; })
        .map([](const int& value) { return std::make_unique<int>(value); })
        .map([](const std::unique_ptr<int>& value) { return std::to_string(*value); })
        .map([](const std::string& value) { return "value=" + value; });
    input[1] = 8;
    const std::vector<std::string> expected{"value=2", "value=16", "value=6", "value=8"};
    expect_value(plan.collect(), expected);
    EXPECT_EQ(std::vector<std::string>(plan.begin(), plan.end()), expected);
    EXPECT_EQ(input, (std::vector<int>{1, 8, 3, 4}));

    const auto twice = [](std::int64_t value) { return value * 2; };
    expect_value(Range(4, 0, -1).iter_paral(3).map(twice).map(twice).collect(),
                 (std::vector<std::int64_t>{16, 12, 8, 4}));
    expect_value(Range(0).into_inter().map(twice).collect(), std::vector<std::int64_t>{});
    expect_error(Range(0, 1, 0).into_inter().map(twice).map(twice).collect(),
                 std::errc::invalid_argument);
}

TEST(PipelineTest, MovesMaterializedValuesIntoFusedCallbacks) {
    const auto plan = Range(128).iter_paral(4)
        .map([](auto value) { return std::make_unique<Range::value_type>(value); }).materialize()
        .map([](std::unique_ptr<Range::value_type> value) { return *value; })
        .map([](auto value) { return value * 3; });

    const Range expected(0, 384, 3);
    expect_value(plan.collect(), (std::vector<Range::value_type>(expected.begin(), expected.end())));
}

TEST(PipelineTest, ExposesWritableReferencesAndUpdatesArrays) {
    std::list<int> values{1, 2, 3, 4};
    for (auto& value : iter(values)) {
        value += 1;
    }
    EXPECT_EQ(values, (std::list<int>{2, 3, 4, 5}));
    static_assert(std::is_same<decltype(*iter(static_cast<const std::list<int>&>(values)).begin()), const int&>::value,
                  "read-only traversal exposes const references");

    int array[]{1, 2, 3};
    for (auto& value : iter(array)) {
        ++value;
    }
    expect_value(iter(array).collect(), std::vector<int>{2, 3, 4});
    expect_value(Range(3).iter().map([](auto value) { return value * 2; }).collect(),
                 std::vector<std::int64_t>{0, 2, 4});
}

TEST(PipelineTest, SupportsSizedAndSingleBorrowedPointers) {
    int array[]{4, 5, 6};
    int* pointer = array;
    expect_value(iter(pointer).collect(), std::vector<int>{4});
    expect_value(iter(array).collect(), std::vector<int>{4, 5, 6});
    expect_value(iter_paral(pointer, 3, 2).map([](int value) { return value * 2; }).collect(),
                 std::vector<int>{8, 10, 12});
    const int* empty = nullptr;
    expect_value(iter(empty, 0).collect(), std::vector<int>{});
    expect_error(iter_paral(empty, 1, 2).collect(), std::errc::invalid_argument);
}

TEST(PipelineTest, ConsumesMoveOnlyRangesAndBuffersOnce) {
    std::vector<std::unique_ptr<int>> source;
    source.emplace_back(std::make_unique<int>(4));
    source.emplace_back(std::make_unique<int>(7));
    expect_value(iter(source).map([](const std::unique_ptr<int>& value) { return *value; }).collect(),
                 std::vector<int>{4, 7});
    expect_value(iter_paral(source, 2).filter([](const auto& value) { return *value > 4; })
                     .map([](const auto& value) { return *value; }).collect(), std::vector<int>{7});
    ASSERT_NE(source.front(), nullptr);
    const auto plan = into_inter_paral(std::move(source), 2).map([](std::unique_ptr<int> value) { return *value; });
    const auto copy = plan;
    expect_value(plan.collect(), std::vector<int>{4, 7});
    expect_error(copy.collect(), std::errc::operation_not_permitted);

    auto buffer = std::make_unique<std::unique_ptr<int>[]>(2);
    buffer[0] = std::make_unique<int>(1);
    buffer[1] = std::make_unique<int>(2);
    auto owned = into_inter(std::move(buffer), 2).collect();
    ASSERT_TRUE(owned) << owned.error().message();
    ASSERT_NE(owned.value()[1], nullptr);
    EXPECT_EQ(*owned.value()[1], 2);
}

TEST(PipelineTest, EnumeratePreservesReferencesAndUsesPipelineOrder) {
    std::vector<int> source{1, 2, 3, 4};
    for (auto item : iter(source).enumerate()) {
        std::get<1>(item) += static_cast<int>(std::get<0>(item));
    }
    EXPECT_EQ(source, (std::vector<int>{1, 3, 5, 7}));

    using Item = std::tuple<std::size_t, int>;
    expect_value(iter_paral(source, 3).filter([](int value) { return value >= 5; }).enumerate().collect(),
                 std::vector<Item>{{0, 5}, {1, 7}});
    expect_value(iter_paral(source, 3).enumerate().filter([](const Item& item) { return std::get<1>(item) >= 5; }).collect(),
                 std::vector<Item>{{2, 5}, {3, 7}});
    expect_value(iter(source).enumerate().map([](std::tuple<std::size_t, int&> item) {
        return std::get<0>(item) + static_cast<std::size_t>(std::get<1>(item));
    }).collect(), std::vector<std::size_t>{1, 4, 7, 10});
}

TEST(PipelineTest, FusedErrorsSkipTheRestOfTheFailedElement) {
    const auto error = std::make_error_code(std::errc::invalid_argument);
    std::vector<Range::value_type> seen;
    const auto collected = Range(4).iter().map([&](Range::value_type value) -> Result<Range::value_type> {
        return value == 1 ? Result<Range::value_type>(outcome::failure(error))
                          : Result<Range::value_type>(outcome::success(value));
    }).enumerate().map([&](const auto& item) {
        seen.push_back(std::get<1>(item));
        return std::get<1>(item);
    }).collect();
    expect_error(collected, error);
    EXPECT_EQ(seen, (std::vector<Range::value_type>{0}));
}

TEST(PipelineTest, InvokesMemberPointersAndPropagatesZippedErrors) {
    struct Sample {
        int value;
        int twice() const { return value * 2; }
    };
    std::vector<Sample> samples{{2}, {3}};
    expect_value(iter(samples).map(&Sample::value).collect(), std::vector<int>{2, 3});
    expect_value(iter(samples).map(&Sample::twice).collect(), std::vector<int>{4, 6});
    auto right = Range(0, 1, 0).iter();
    expect_error(iter(samples).zip(right).collect(), std::errc::invalid_argument);
}

#ifdef PARALLEL_TEST_BOOST_CONTAINER
TEST(PipelineTest, UsesBoostMonotonicScratchWithoutTyingTheOutputToIt) {
    std::list<int> source{1, 2, 3, 4};
    alignas(int) unsigned char buffer[4096];
    boost::container::pmr::monotonic_buffer_resource scratch(buffer, sizeof(buffer));
    const auto available = scratch.remaining_storage();
    const auto result = iter_paral(source, 2).filter([](int value) { return value > 1; }).collect({&scratch});
    EXPECT_LT(scratch.remaining_storage(), available);
    scratch.release();
    expect_value(result, std::vector<int>{2, 3, 4});
}
#endif

TEST(LayerPlanTest, ReadsOneStablePreviousLayerPerRound) {
    std::list<int> values{10, 20, 30, 40};
    const auto caller = std::this_thread::get_id();
    std::size_t advanced = 0;
    const auto plan = iter_paral(values, 3)
        .layer_plan(std::size_t{0}, [](std::size_t i, const auto& previous, std::size_t round) {
            if (round == 0) {
                return previous[i] + previous[(i + previous.size() - 1) % previous.size()];
            }
            return previous[(i + 1) % previous.size()];
        }, [&](std::size_t round, const auto& previous, const auto& completed) {
            EXPECT_EQ(std::this_thread::get_id(), caller);
            EXPECT_EQ(previous[0], round == 0 ? 10 : 50);
            EXPECT_EQ(completed[0], round == 0 ? 50 : 30);
            ++advanced;
            return round + 1;
        }, [](std::size_t round) { return round == 2; });
    EXPECT_EQ(values, (std::list<int>{10, 20, 30, 40}));
    EXPECT_EQ(advanced, 0u);
    expect_value(plan.collect(), std::vector<int>{30, 50, 70, 50});
    EXPECT_EQ(advanced, 2u);
    EXPECT_EQ(values, (std::list<int>{30, 50, 70, 50}));
}

TEST(LayerPlanTest, KeepsLastCompletedRoundOnWorkerError) {
    std::list<int> values{31, 51, 71, 51};
    const auto failure = iter_paral(values, 4).layer_plan(
        std::size_t{0}, [](std::size_t i, const auto& previous, std::size_t round) -> Result<int> {
            if (round == 1 && i == 2) {
                return outcome::failure(std::make_error_code(std::errc::invalid_argument));
            }
            return outcome::success(previous[i] + 1);
        }, [](std::size_t round, const auto&, const auto&) { return round + 1; },
        [](std::size_t round) { return round == 2; });
    expect_error(failure.collect(), std::errc::invalid_argument);
    EXPECT_EQ(values, (std::list<int>{32, 52, 72, 52}));
}

TEST(LayerPlanTest, CanStopBeforeFirstRoundOrOnConvergence) {
    using Value = Range::value_type;
    const auto add_one = [](std::size_t i, const auto& previous, const auto&) -> Value {
        return std::min(previous[i] + 1, Value{3});
    };
    std::size_t advanced = 0;
    const auto track_changes = [&](bool, const auto& previous, const auto& completed) {
        ++advanced;
        for (std::size_t i = 0; i < previous.size(); ++i) {
            if (previous[i] != completed[i]) {
                return true;
            }
        }
        return false;
    };
    const auto converged = [](bool changed) { return !changed; };
    const auto no_rounds = Range(3).iter_paral(2).layer_plan(false, add_one, track_changes, converged);
    expect_value(no_rounds.collect(), std::vector<Value>{0, 1, 2});
    EXPECT_EQ(advanced, 0u);

    const auto until_stable = Range(3).iter_paral(2).layer_plan(true, add_one, track_changes, converged);
    expect_value(until_stable.collect(), std::vector<Value>{3, 3, 3});
    EXPECT_EQ(advanced, 4u);
    expect_value(until_stable.collect(), std::vector<Value>{3, 3, 3});
    EXPECT_EQ(advanced, 8u);
}

TEST(LayerPlanTest, KeepsPendingLayerPrivateOnStateError) {
    const auto error = std::make_error_code(std::errc::permission_denied);
    std::vector<int> input{1, 2, 3};
    const auto failed = iter_paral(input, 2).layer_plan(0,
        [](std::size_t i, const auto& old, int) { return old[i] + 1; },
        [&](int, const auto&, const auto&) -> Result<int> { return outcome::failure(error); },
        [](int) { return false; }).collect();
    expect_error(failed, error);
    EXPECT_EQ(input, (std::vector<int>{1, 2, 3}));
}

TEST(ReduceTest, AppliesSeedOnceAndKeepsOperandOrder) {
    const auto sum = Range(1000).iter_paral(4).reduce(std::int64_t{7}, std::plus<>{});
    expect_value(sum.collect_one(), 499507);
    expect_value(Range(0).iter_paral(4).reduce(std::int64_t{7}, std::plus<>{}).collect_one(), 7);

    const std::vector<std::string> words{"a", "b", "c", "d", "e", "f", "g"};
    expect_value(iter_paral(words, 3).reduce(std::string("!"), std::plus<>{}).collect_one(),
                 std::string("!abcdefg"));
    expect_value(iter_paral(words, 3)
                     .map([](const auto& word) { return word.size(); })
                     .reduce(std::size_t{0}, std::plus<>{}).collect_one(),
                 std::size_t{7});
}

TEST(ReduceTest, ReducesFusedMapOnMultipleWorkers) {
    using Value = Range::value_type;
    std::mutex mutex;
    std::condition_variable ready;
    std::thread::id first_worker;
    bool used_second_worker = false;
    const auto sum = Range(128).iter_paral(2)
        .map([](Value value) { return value + 10; })
        .map([](Value value) { return value + 20; })
        .map([](Value value) { return value + 30; })
        .map([](Value value) { return value + 40; })
        .reduce(Value{7}, [&](Value a, Value b) {
            const auto worker = std::this_thread::get_id();
            std::unique_lock<std::mutex> lock(mutex);
            if (first_worker == std::thread::id{}) {
                first_worker = worker;
                EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return used_second_worker; }));
            } else if (!used_second_worker && worker != first_worker) {
                used_second_worker = true;
                ready.notify_one();
            }
            return a + b;
        });
    expect_value(sum.collect_one(), 20935);
    EXPECT_TRUE(used_second_worker);
}

TEST(PipelineErrorsTest, PropagatesCallbackErrorsAndSkipsDownstreamStages) {
    using Value = Range::value_type;
    const auto error = std::make_error_code(std::errc::permission_denied);
    const auto fail_on_three = [](Value value) -> outcome::result<Value> {
        if (value == 3) {
            return outcome::failure(boost::system::errc::make_error_code(boost::system::errc::permission_denied));
        }
        return outcome::success(value);
    };

    for (auto workers : {std::size_t{1}, std::size_t{4}}) {
        SCOPED_TRACE(workers);
        const auto plan = Range(16).iter_paral(workers).map(fail_on_three);
        const auto downstream = plan.materialize().map([](Value value) {
            ADD_FAILURE() << "a failed stage must skip downstream callbacks";
            return value;
        }).collect();
        expect_error(downstream, std::errc::permission_denied);
        expect_error(plan.map([](Value value) { return value + 1; })
                         .map([](Value value) { return value * 2; })
                         .reduce(Value{0}, std::plus<>{}).collect_one(),
                     std::errc::permission_denied);
        const auto first = plan.begin();
        EXPECT_EQ(first, plan.end());
        EXPECT_EQ(first.error(), std::errc::permission_denied);

        const auto folded = Range(16).iter_paral(workers).reduce(Value{0},
            [&](Value total, Value value) -> Result<Value> {
                if (value == 3) {
                    return outcome::failure(error);
                }
                return outcome::success(total + value);
            }).collect_one();
        expect_error(folded, error);
    }
}

TEST(PipelineErrorsTest, TranslatesExceptionsAndRejectsWrongCollectionSize) {
    using Value = Range::value_type;
    for (auto workers : {std::size_t{1}, std::size_t{4}}) {
        SCOPED_TRACE(workers);
        const std::vector<Value> empty;
        const auto failed = Range(4).iter_paral(workers).map([&](Value) {
            return empty.at(0);
        }).collect();
        expect_error(failed, parallel::make_error_code(parallel::Errc::unexpected_exception));
    }
    expect_error(Range(2).iter().collect_one(), std::errc::result_out_of_range);
}

TEST(SchedulerTest, ReportsStartupFailureBeforeRunningTasks) {
    const auto huge = std::numeric_limits<std::size_t>::max();
    parallel::detail::Executor executor(huge);
    const auto result = executor.run_chunks(huge,
        [](std::size_t, std::size_t) -> Result<void> {
            ADD_FAILURE() << "a failed worker startup must run no tasks";
            return outcome::success();
        });
    expect_error(result, std::errc::value_too_large);
}

std::size_t thread_token() {
    static std::atomic<std::size_t> next{0};
    thread_local const auto token = next.fetch_add(1, std::memory_order_relaxed);
    return token;
}

TEST(SchedulerTest, ReusesWorkersAcrossStagesAndRepeatedCollects) {
    using Value = Range::value_type;
    std::array<std::array<std::size_t, 2>, 4> participants{};
    std::array<std::size_t, 4> arrived{};
    std::mutex mutex;
    std::condition_variable ready;
    const auto record = [&](std::size_t stage, Value value) {
        std::unique_lock<std::mutex> lock(mutex);
        participants[stage][static_cast<std::size_t>(value)] = thread_token();
        if (++arrived[stage] == participants[stage].size()) {
            ready.notify_all();
        } else {
            EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] {
                return arrived[stage] == participants[stage].size();
            }));
        }
        return value;
    };

    const auto plan = Range(2).iter_paral(2)
        .map([&](Value value) { return record(0, value); }).materialize()
        .map([&](Value value) { return record(1, value); })
        .layer_plan(std::size_t{0}, [&](std::size_t i, const auto& previous, std::size_t round) {
            return record(2 + round, previous[i]);
        }, [](std::size_t round, const auto&, const auto&) { return round + 1; },
        [](std::size_t round) { return round == 2; });

    std::array<std::size_t, 2> team{};
    for (int repeat = 0; repeat < 2; ++repeat) {
        arrived.fill(0);
        expect_value(plan.collect(), std::vector<Value>{0, 1});
        for (auto& stage : participants) {
            std::sort(stage.begin(), stage.end());
            EXPECT_NE(stage[0], stage[1]);
        }
        if (repeat == 0) {
            team = participants.front();
        }
        for (const auto& stage : participants) {
            EXPECT_EQ(stage, team);
        }
    }
}

TEST(SchedulerTest, CollectBorrowsTheSameExecutorAcrossDifferentPipelines) {
    using Value = Range::value_type;
    parallel::Executor executor(2);
    const auto caller = thread_token();
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t arrived = 0;
    std::size_t worker = caller;
    const auto record = [&](Value value) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto token = thread_token();
        if (token != caller) {
            worker = token;
        }
        if (++arrived == 2) {
            ready.notify_all();
        } else {
            EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return arrived == 2; }));
        }
        return value;
    };

    expect_value(Range(2).into_inter().map(record).collect(executor), std::vector<Value>{0, 1});
    ASSERT_NE(worker, caller);
    const auto first_worker = worker;
    arrived = 0;
    worker = caller;
    expect_value(Range(2, 4).into_inter().map(record).collect(executor), std::vector<Value>{2, 3});
    EXPECT_EQ(worker, first_worker);
}

TEST(SchedulerTest, DefaultCollectionsReuseWorkersAcrossCallingThreads) {
    using Value = Range::value_type;
    std::array<std::size_t, 2> background{};
    std::array<std::size_t, 2> callers{};
    for (std::size_t batch = 0; batch < background.size(); ++batch) {
        std::thread caller([&, batch] {
            callers[batch] = thread_token();
            std::mutex mutex;
            std::condition_variable ready;
            std::size_t arrived = 0;
            const auto result = Range(2).iter_paral(2).map([&](Value value) {
                std::unique_lock<std::mutex> lock(mutex);
                const auto worker = thread_token();
                if (worker != callers[batch]) {
                    background[batch] = worker;
                }
                if (++arrived == 2) {
                    ready.notify_one();
                } else {
                    EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return arrived == 2; }));
                }
                return value;
            }).collect();
            expect_value(result, std::vector<Value>{0, 1});
        });
        caller.join();
    }
    EXPECT_NE(callers[0], callers[1]);
    EXPECT_NE(background[0], callers[0]);
    EXPECT_NE(background[1], callers[1]);
    EXPECT_EQ(background[0], background[1]);
}

TEST(SchedulerTest, NestedCollectorsCanRunWhileTheOuterTeamIsBusy) {
    using Value = Range::value_type;
    std::mutex mutex;
    std::condition_variable ready;
    int arrived = 0;
    const auto result = Range(2).iter_paral(2).map([&](Value value) -> Result<Value> {
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (++arrived == 2) {
                ready.notify_all();
            } else {
                EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return arrived == 2; }));
            }
        }
        auto inner = Range(value, value + 2).iter_paral(2).map([](Value i) {
            return i + 1;
        }).collect();
        if (!inner) {
            return inner.as_failure();
        }
        return outcome::success(inner.assume_value()[0] + inner.assume_value()[1]);
    }).collect();
    expect_value(result, std::vector<Value>{3, 5});
}

TEST(SchedulerTest, CanGrowAndRunSmallerBatchesAfterAWorkerError) {
    parallel::detail::Executor executor(4);
    const auto failed = executor.run_chunks(2, [](std::size_t, std::size_t) -> Result<void> {
        return outcome::failure(std::make_error_code(std::errc::operation_canceled));
    });
    expect_error(failed, std::errc::operation_canceled);

    for (auto size : {std::size_t{4}, std::size_t{2}}) {
        std::array<int, 4> visits{};
        const auto completed = executor.run_chunks(size, [&](std::size_t first, std::size_t last) -> Result<void> {
            for (auto i = first; i < last; ++i) {
                ++visits[i];
            }
            return outcome::success();
        });
        ASSERT_TRUE(completed) << completed.error().message();
        for (std::size_t i = 0; i < visits.size(); ++i) {
            EXPECT_EQ(visits[i], i < size ? 1 : 0);
        }
    }
}

TEST(SchedulerTest, AllowsTheWholeTeamToWorkOnTheTail) {
    constexpr std::size_t tail_begin = 4096 - 64;
    constexpr std::size_t size = 4100;
    constexpr std::size_t workers = 4;
    parallel::detail::Executor executor(workers);
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t arrived = 0;

    const auto result = executor.run_chunks(size, [&](std::size_t first, std::size_t last) -> Result<void> {
        if (last > tail_begin) {
            EXPECT_GE(first, tail_begin);
            std::unique_lock<std::mutex> lock(mutex);
            if (++arrived >= workers) {
                ready.notify_all();
            } else {
                EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return arrived >= workers; }));
            }
        }
        return outcome::success();
    });

    ASSERT_TRUE(result) << result.error().message();
    EXPECT_GE(arrived, workers);
}

TEST(SchedulerTest, OtherWorkersContinueWhileOneChunkIsBlocked) {
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t completed = 0;
    const auto result = Range(400).iter_paral(4).layer_plan(
        false, [&](std::size_t i, const auto& previous, bool) {
            std::unique_lock<std::mutex> lock(mutex);
            if (i == 0) {
                EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(5), [&] { return completed >= 375; }));
            } else if (++completed == 375) {
                ready.notify_one();
            }
            return previous[i];
        }, [](bool, const auto&, const auto&) { return true; }, [](bool done) { return done; }).collect();
    const Range ordered(400);
    expect_value(result, std::vector<Range::value_type>(ordered.begin(), ordered.end()));
}

}  // namespace

TEST(ChunkTest, RunsChunkBodiesAndPropagatesErrors) {
    std::atomic<std::size_t> visited{0};
    const auto ok = parallel::for_each_chunk(1000, 4, [&](std::size_t first, std::size_t last) {
        visited.fetch_add(last - first, std::memory_order_relaxed);
    });
    EXPECT_TRUE(ok) << ok.error().message();
    EXPECT_EQ(visited.load(), 1000u);

    const auto empty = parallel::for_each_chunk(0, 4, [](std::size_t, std::size_t) {});
    EXPECT_TRUE(empty) << empty.error().message();

    const auto failed = parallel::for_each_chunk(
        100, 2, [](std::size_t, std::size_t) -> Result<void> {
            return outcome::failure(std::make_error_code(std::errc::invalid_argument));
        });
    EXPECT_FALSE(failed);
    EXPECT_EQ(failed.error(), std::errc::invalid_argument);
}

TEST(ChunkTest, HonorsGrainHint) {
    std::mutex mutex;
    std::size_t largest = 0;
    const auto ok = parallel::for_each_chunk(1000, 4, [&](std::size_t first, std::size_t last) {
        std::lock_guard<std::mutex> lock(mutex);
        largest = std::max(largest, last - first);
    }, 100);
    EXPECT_TRUE(ok) << ok.error().message();
    EXPECT_EQ(largest, 100u);
}

TEST(ChunkTest, DefaultGrainGrowsWithoutLosingAnyIndices) {
    for (const auto size : {std::size_t{65539}, std::numeric_limits<std::size_t>::max()}) {
        SCOPED_TRACE(size);
        std::mutex mutex;
        std::vector<std::pair<std::size_t, std::size_t>> chunks;
        const auto result = parallel::for_each_chunk(size, 4, [&](std::size_t& first, std::size_t& last) {
            std::lock_guard<std::mutex> lock(mutex);
            chunks.emplace_back(first, last);
        });
        ASSERT_TRUE(result) << result.error().message();
        std::sort(chunks.begin(), chunks.end());
        std::size_t next = 0;
        std::size_t largest = 0;
        for (const auto& chunk : chunks) {
            EXPECT_EQ(chunk.first, next);
            ASSERT_LT(chunk.first, chunk.second);
            ASSERT_LE(chunk.second, size);
            largest = std::max(largest, chunk.second - chunk.first);
            next = chunk.second;
        }
        EXPECT_EQ(next, size);
        EXPECT_GT(largest, 64u);
    }
}

TEST(ChunkTest, StaticScheduleUsesBalancedContiguousPartitions) {
    constexpr std::size_t size = 17;
    parallel::Executor executor(4, parallel::Schedule::static_partition);
    std::mutex mutex;
    std::vector<std::pair<std::size_t, std::size_t>> partitions;
    const auto result = executor.run_chunks(size, [&](std::size_t first, std::size_t last) -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex);
        partitions.emplace_back(first, last);
        return outcome::success();
    });
    ASSERT_TRUE(result) << result.error().message();
    std::sort(partitions.begin(), partitions.end());
    const decltype(partitions) expected{{0, 5}, {5, 9}, {9, 13}, {13, 17}};
    EXPECT_EQ(partitions, expected);

    const auto stopped = executor.run_chunks(size, [](std::size_t, std::size_t) -> Result<void> {
        return outcome::failure(std::error_code{});
    });
    ASSERT_FALSE(stopped);
    EXPECT_EQ(stopped.error(), std::error_code{});
    const auto plan = Range(size).into_inter().reduce(Range::value_type{0}, std::plus<>{});
    expect_value(plan.collect_one(executor), 136);
}
