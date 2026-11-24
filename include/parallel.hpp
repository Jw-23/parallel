#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include <boost/compat/invoke.hpp>
#include <boost/compat/type_traits.hpp>
#include <boost/optional.hpp>
#include <boost/outcome.hpp>
#include <boost/outcome/policy/terminate.hpp>
#include <boost/outcome/std_result.hpp>
#include <boost/system/system_error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace parallel {

using ErrorCode = std::error_code;

// Check the result before accessing its value. Invalid access terminates.
template<class T>
using Result = boost::outcome_v2::std_result<T, ErrorCode, boost::outcome_v2::policy::terminate>;

enum class Errc { unexpected_exception = 1 };
ErrorCode make_error_code(Errc error) noexcept;

template<class Evaluate, bool InPlace>
class Pipeline;

class Range {
public:
    using value_type = std::int64_t;

    class iterator {
    public:
        using value_type = Range::value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using reference = value_type;
        using pointer = void;

        iterator() = default;
        value_type operator*() const noexcept { return value_; }

        iterator& operator++() noexcept {
            if (--remaining_ != 0) {
                value_ += step_;
            }
            return *this;
        }

        iterator operator++(int) noexcept {
            auto previous = *this;
            ++*this;
            return previous;
        }

        friend bool operator==(const iterator& a, const iterator& b) noexcept {
            return a.remaining_ == b.remaining_ && (a.remaining_ == 0 || a.value_ == b.value_);
        }

        friend bool operator!=(const iterator& a, const iterator& b) noexcept { return !(a == b); }

    private:
        friend class Range;
        iterator(value_type value, value_type step, std::size_t remaining) noexcept
            : value_(value), step_(step), remaining_(remaining) {}

        value_type value_ = 0;
        value_type step_ = 1;
        std::size_t remaining_ = 0;
    };

    explicit Range(value_type stop) noexcept : Range(0, stop) {}

    // The stop value is exclusive, including when step is negative.
    Range(value_type start, value_type stop, value_type step = 1) noexcept : start_(start), step_(step) {
        if (step == 0) {
            error_ = std::make_error_code(std::errc::invalid_argument);
            return;
        }
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            return;
        }
        using U = std::uint64_t;
        const auto distance = step > 0 ? U(stop) - U(start) : U(start) - U(stop);
        const auto stride = step > 0 ? U(step) : U(0) - U(step);
        const auto count = distance / stride + (distance % stride != 0);
        if (count > std::numeric_limits<std::size_t>::max()) {
            error_ = std::make_error_code(std::errc::value_too_large);
            return;
        }
        size_ = static_cast<std::size_t>(count);
    }

    iterator begin() const noexcept { return iterator(start_, step_, size_); }
    iterator end() const noexcept { return {}; }
    std::size_t size() const noexcept { return size_; }
    ErrorCode error() const noexcept { return error_; }

    value_type operator[](std::size_t index) const noexcept {
        using U = std::uint64_t;
        const auto value = U(start_) + U(step_) * U(index);
        if (value <= U(std::numeric_limits<value_type>::max())) {
            return static_cast<value_type>(value);
        }
        return -1 - static_cast<value_type>(std::numeric_limits<U>::max() - value);
    }

    auto iter() const;
    auto into_inter() const;
    auto iter_paral(std::size_t workers = 0) const;

private:
    value_type start_;
    value_type step_;
    std::size_t size_ = 0;
    ErrorCode error_;
};

namespace detail {

namespace outcome = boost::outcome_v2;
using boost::compat::invoke;
using boost::compat::void_t;

inline ErrorCode standard_error(const boost::system::error_code& error) noexcept {
    try {
        return static_cast<ErrorCode>(error);
    } catch (const std::bad_alloc&) {
        return std::make_error_code(std::errc::not_enough_memory);
    } catch (...) {
        return make_error_code(Errc::unexpected_exception);
    }
}

template<class Function>
auto run_safe(const Function& function) noexcept -> decltype(function()) {
    try {
        return function();
    } catch (const std::system_error& error) {
        return outcome::failure(error.code());
    } catch (const boost::system::system_error& error) {
        return outcome::failure(standard_error(error.code()));
    } catch (const std::bad_alloc&) {
        return outcome::failure(std::make_error_code(std::errc::not_enough_memory));
    } catch (const std::length_error&) {
        return outcome::failure(std::make_error_code(std::errc::value_too_large));
    } catch (...) {
        return outcome::failure(make_error_code(Errc::unexpected_exception));
    }
}

template<class T>
struct ResultTraits : std::false_type { using value_type = T; };

template<class T, class E, class Policy>
struct ResultTraits<outcome::basic_result<T, E, Policy>> : std::true_type { using value_type = T; };

template<class Function, class... Args>
using Invocation = std::result_of_t<Function(Args...)>;

template<class Function, class... Args>
using CallbackValue = typename ResultTraits<std::decay_t<Invocation<Function, Args...>>>::value_type;

template<class R>
auto unpack_success(R&& result, std::false_type) {
    return outcome::success(std::move(result).assume_value());
}

template<class R>
auto unpack_success(R&&, std::true_type) { return outcome::success(); }

template<class T>
Result<T> into_result(Result<T> result) { return result; }

template<class T, class E, class Policy>
Result<T> into_result(outcome::basic_result<T, E, Policy> result) {
    if (!result) {
        return outcome::failure(ErrorCode(result.assume_error()));
    }
    return unpack_success(std::move(result), std::is_void<T>{});
}

template<class T, std::enable_if_t<!ResultTraits<std::decay_t<T>>::value, int> = 0>
Result<std::decay_t<T>> into_result(T&& value) {
    return outcome::success(std::forward<T>(value));
}

template<class Function, class... Args,
         std::enable_if_t<std::is_void<Invocation<const Function&, Args&&...>>::value, int> = 0>
Result<void> call(const Function& function, Args&&... args) {
    detail::invoke(function, std::forward<Args>(args)...);
    return outcome::success();
}

template<class Function, class... Args,
         std::enable_if_t<!std::is_void<Invocation<const Function&, Args&&...>>::value, int> = 0>
auto call(const Function& function, Args&&... args) {
    return into_result(detail::invoke(function, std::forward<Args>(args)...));
}

// An executor runs synchronous batches and retains its team between them.
class Executor {
public:
    using ChunkTask = std::function<Result<void>(std::size_t, std::size_t)>;
    enum class Reuse { evaluation, calling_thread };

    explicit Executor(std::size_t workers, Reuse reuse = Reuse::evaluation) noexcept;
    ~Executor();

    std::size_t worker_limit() const noexcept { return worker_limit_; }
    Result<void> run_chunks(std::size_t size, const ChunkTask& run);

private:
    struct Team;
    struct TeamCache;
    void acquire_team();

    std::size_t worker_limit_;
    Reuse reuse_;
    TeamCache* cache_ = nullptr;
    std::unique_ptr<Team> team_;
};

template<class Function>
Result<void> for_each_index(std::size_t size, Executor& executor, const Function& apply) {
    return executor.run_chunks(size, [&](std::size_t first, std::size_t last) -> Result<void> {
        for (auto i = first; i < last; ++i) {
            const auto applied = call(apply, i);
            if (!applied) {
                return applied.as_failure();
            }
        }
        return outcome::success();
    });
}

using std::begin;
using std::end;

template<class R>
using Iterator = decltype(begin(std::declval<R&>()));

template<class R>
using Sentinel = decltype(end(std::declval<R&>()));

template<class I>
using Dereference = decltype(*std::declval<I&>());

template<class I, class = void>
struct IteratorValue {
    using type = std::decay_t<Dereference<I>>;
    using category = std::input_iterator_tag;
};

template<class I>
struct IteratorValue<I, void_t<typename std::iterator_traits<I>::value_type,
                                    typename std::iterator_traits<I>::iterator_category>> {
    using type = typename std::iterator_traits<I>::value_type;
    using category = typename std::iterator_traits<I>::iterator_category;
};

template<class R, class = void>
struct IsIterable : std::false_type {};

template<class R>
struct IsIterable<R, void_t<Iterator<R>, Sentinel<R>, Dereference<Iterator<R>>,
                                decltype(++std::declval<Iterator<R>&>()),
                                decltype(bool(std::declval<Iterator<R>&>() !=
                                              std::declval<Sentinel<R>&>()))>> : std::true_type {};

template<class R>
struct IsInitializerList : std::false_type {};

template<class T>
struct IsInitializerList<std::initializer_list<T>> : std::true_type {};

template<class R>
using EnableIterable = std::enable_if_t<IsIterable<R>::value &&
    (std::is_lvalue_reference<R>::value ||
     (!std::is_array<std::remove_reference_t<R>>::value && !IsInitializerList<std::decay_t<R>>::value)), int>;

template<class T, bool Writable = true>
class Values {
public:
    using value_type = T;
    static constexpr bool writable = Writable;
    // vector<bool> shares words between elements. Give each bool its own slot.
    static constexpr bool direct = std::is_arithmetic<T>::value && !std::is_same<T, bool>::value;

    explicit Values(std::size_t size = 0) : slots(size) {}

    std::size_t size() const noexcept { return slots.size(); }
    const T& read(std::size_t index) const { return value(slots[index]); }
    T& write(std::size_t index) { return value(slots[index]); }
    decltype(auto) take(std::size_t index) { return std::move_if_noexcept(write(index)); }

    void resize(std::size_t size) { slots.resize(size); }
    void reserve(std::size_t size) { slots.reserve(size); }
    void swap(Values& other) noexcept { slots.swap(other.slots); }

    template<class U>
    void put(std::size_t index, U&& input) { store(slots[index], std::forward<U>(input)); }

    template<class U>
    void append(U&& input) { append(std::forward<U>(input), std::integral_constant<bool, direct>{}); }

    std::vector<T> release_vector() { return std::move(slots); }

private:
    static T& value(T& slot) { return slot; }
    static const T& value(const T& slot) { return slot; }
    static T& value(boost::optional<T>& slot) { return *slot; }
    static const T& value(const boost::optional<T>& slot) { return *slot; }

    template<class U>
    static void store(T& slot, U&& input) { slot = T(std::forward<U>(input)); }

    template<class U>
    static void store(boost::optional<T>& slot, U&& input) { slot.emplace(std::forward<U>(input)); }

    template<class U>
    void append(U&& input, std::true_type) { slots.emplace_back(std::forward<U>(input)); }

    template<class U>
    void append(U&& input, std::false_type) { slots.emplace_back(boost::in_place_init, std::forward<U>(input)); }

    using Slot = std::conditional_t<direct, T, boost::optional<T>>;
    std::vector<Slot> slots;
};

template<class Storage>
auto collect_values(Storage& values) -> std::vector<typename Storage::value_type> {
    std::vector<typename Storage::value_type> result;
    result.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result.emplace_back(values.take(i));
    }
    return result;
}

template<class T, bool Writable, std::enable_if_t<Values<T, Writable>::direct, int> = 0>
std::vector<T> collect_values(Values<T, Writable>& values) {
    return values.release_vector();
}

template<class Output, class Value, class Function>
void map_value(Output& output, std::size_t i, const Value& value, const Function& function, std::false_type) {
    output.put(i, detail::invoke(function, value));
}

template<class Output, class Value, class Function>
Result<void> map_value(Output& output, std::size_t i, const Value& value, const Function& function, std::true_type) {
    auto mapped = call(function, value);
    if (!mapped) {
        return mapped.as_failure();
    }
    output.put(i, std::move(mapped).assume_value());
    return outcome::success();
}

template<class Access>
class References {
    using Reference = Invocation<const Access&, std::size_t>;

public:
    using value_type = std::remove_cv_t<std::remove_reference_t<Reference>>;
    static constexpr bool writable = !std::is_const<std::remove_reference_t<Reference>>::value;

    References(std::size_t size, Access access) : size_(size), access_(std::move(access)) {}

    std::size_t size() const noexcept { return size_; }
    const value_type& read(std::size_t index) const { return access_(index); }
    Reference write(std::size_t index) { return access_(index); }
    const value_type& take(std::size_t index) const { return read(index); }

private:
    std::size_t size_;
    Access access_;
};

struct RangeValues {
    using value_type = Range::value_type;
    static constexpr bool writable = false;

    std::size_t size() const noexcept { return range.size(); }
    value_type read(std::size_t index) const noexcept { return range[index]; }
    value_type take(std::size_t index) const noexcept { return read(index); }

    Range range;
};

template<class S>
constexpr bool can_update = S::writable || std::is_same<S, RangeValues>::value;

template<class S>
Result<S> writable_values(S input, Executor&) {
    return outcome::success(std::move(input));
}

inline Result<Values<Range::value_type>> writable_values(RangeValues input, Executor& executor) {
    Values<Range::value_type> result(input.size());
    const auto filled = for_each_index(input.size(), executor, [&](std::size_t i) {
        result.put(i, input.read(i));
    });
    if (!filled) {
        return filled.as_failure();
    }
    return outcome::success(std::move(result));
}

template<class T>
Result<void> publish_layer(Values<T>& current, Values<T>& pending, Executor&) noexcept {
    current.swap(pending);
    return outcome::success();
}

template<class S, class T>
Result<void> publish_layer(S& current, Values<T>& pending, Executor& executor) {
    return for_each_index(current.size(), executor, [&](std::size_t i) {
        current.write(i) = std::move(pending.write(i));
    });
}

template<class S>
class LayerView {
public:
    explicit LayerView(const S& input) : input_(input) {}
    std::size_t size() const noexcept { return input_.size(); }
    const typename S::value_type& operator[](std::size_t index) const { return input_.read(index); }

private:
    const S& input_;
};

template<class Access>
auto make_references(std::size_t size, Access access) {
    return References<Access>(size, std::move(access));
}

template<class I>
auto load_iterators(I first, I last, std::random_access_iterator_tag) {
    // Empty ranges can use null pointers for both endpoints.
    const auto size = first != last ? static_cast<std::size_t>(last - first) : 0;
    return make_references(size, [first](std::size_t i) -> Dereference<I> {
        return first[static_cast<typename std::iterator_traits<I>::difference_type>(i)];
    });
}

template<class I, class S>
auto load_iterators(I first, S last, std::forward_iterator_tag) {
    using Element = std::remove_reference_t<Dereference<I>>;
    std::vector<Element*> positions;
    for (; first != last; ++first) {
        positions.push_back(std::addressof(*first));
    }
    const auto size = positions.size();
    return make_references(size, [positions = std::move(positions)](std::size_t i) -> Dereference<I> {
        return *positions[i];
    });
}

template<class I, class S>
auto load_iterators(I first, S last, std::input_iterator_tag) {
    Values<typename IteratorValue<I>::type, false> result;
    for (; first != last; ++first) {
        result.append(*first);
    }
    return result;
}

inline Result<RangeValues> load(const Range& range) {
    if (range.error()) {
        return outcome::failure(range.error());
    }
    return outcome::success(RangeValues{range});
}

template<class R, std::enable_if_t<!std::is_same<std::decay_t<R>, Range>::value, int> = 0>
auto load(R& range) {
    using I = Iterator<R>;
    using Traits = IteratorValue<I>;
    using Category = typename Traits::category;
    using Reference = Dereference<I>;
    using Element = std::remove_reference_t<Reference>;
    constexpr bool stable_reference = std::is_base_of<std::forward_iterator_tag, Category>::value &&
        std::is_lvalue_reference<Reference>::value &&
        std::is_same<std::remove_const_t<Element>, typename Traits::type>::value;
    constexpr bool indexed = stable_reference && std::is_same<I, Sentinel<R>>::value &&
        std::is_base_of<std::random_access_iterator_tag, Category>::value;
    using Access = std::conditional_t<indexed, std::random_access_iterator_tag,
        std::conditional_t<stable_reference, std::forward_iterator_tag, std::input_iterator_tag>>;
    auto first = begin(range);
    auto last = end(range);
    auto input = load_iterators(std::move(first), std::move(last), Access{});
    return Result<decltype(input)>(outcome::success(std::move(input)));
}

template<class R>
struct OwnedRange {
    mutable R range;
    R& get() const { return range; }
};

template<class R>
auto hold_range(R&& range, std::true_type) { return std::ref(range); }

template<class R>
auto hold_range(R&& range, std::false_type) {
    return OwnedRange<std::decay_t<R>>{std::forward<R>(range)};
}

template<class T>
struct IsPipeline : std::false_type {};

template<class Evaluate, bool InPlace>
struct IsPipeline<Pipeline<Evaluate, InPlace>> : std::true_type {};

template<class R>
auto as_pipeline(R&& range, std::size_t workers, std::false_type);

template<class R>
auto as_pipeline(R&& range, std::size_t, std::true_type) { return std::forward<R>(range); }

template<class R>
auto as_pipeline(R&& range, std::size_t workers) {
    return as_pipeline(std::forward<R>(range), workers, IsPipeline<std::decay_t<R>>{});
}

struct MakePair {
    template<class A, class B>
    auto operator()(const A& a, const B& b) const { return std::pair<A, B>(a, b); }
};

template<class Evaluate>
using PlanStorage = typename Invocation<const Evaluate&, Executor&>::value_type;

template<class Parent, class Function>
class MapPlan;

template<class Evaluate>
auto prepare_values(const Evaluate& evaluate, Executor& executor) {
    return evaluate(executor);
}

template<class Parent, class Function>
auto prepare_values(const MapPlan<Parent, Function>& evaluate, Executor& executor) {
    return evaluate.prepare(executor);
}

template<class Input>
auto chunk_reader(const Input& input, int) -> decltype(input.for_chunk()) {
    return input.for_chunk();
}

inline RangeValues chunk_reader(const RangeValues& input, int) noexcept {
    return input;
}

template<class Input>
const Input& chunk_reader(const Input& input, long) noexcept {
    return input;
}

template<class Parent, class Function>
class MapPlan {
    using InputValue = typename PlanStorage<Parent>::value_type;
    using Value = std::decay_t<Invocation<const Function&, const InputValue&>>;

    template<class Input>
    struct Reader {
        using value_type = Value;

        std::size_t size() const noexcept { return input.size(); }

        Value read(std::size_t index) const {
            const InputValue& value = input.read(index);
            return detail::invoke(transform, value);
        }

        auto for_chunk() const {
            using ChunkInput = decltype(detail::chunk_reader(input, 0));
            return Reader<ChunkInput>{detail::chunk_reader(input, 0), transform};
        }

        Input input;
        const Function& transform;
    };

public:
    MapPlan(std::shared_ptr<const Parent> parent, Function transform)
        : parent_(std::move(parent)), transform_(std::move(transform)) {}

    // Prepared readers borrow the callbacks from the plan for this evaluation.
    auto prepare(Executor& executor) const {
        auto input = prepare_values(*parent_, executor);
        using Prepared = Reader<typename decltype(input)::value_type>;
        if (!input) {
            return Result<Prepared>(input.as_failure());
        }
        return Result<Prepared>(outcome::success(Prepared{std::move(input).assume_value(), transform_}));
    }

    Result<Values<Value>> operator()(Executor& executor) const {
        auto prepared = prepare(executor);
        if (!prepared) {
            return prepared.as_failure();
        }
        const auto& input = prepared.assume_value();
        Values<Value> output(input.size());
        const auto mapped = executor.run_chunks(input.size(), [&](std::size_t first, std::size_t last) -> Result<void> {
            // Snapshot Range's metadata in the loop's local reader. Materialized input stays borrowed.
            const auto reader = input.for_chunk();
            for (auto i = first; i < last; ++i) {
                output.put(i, reader.read(i));
            }
            return outcome::success();
        });
        if (!mapped) {
            return mapped.as_failure();
        }
        return outcome::success(std::move(output));
    }

private:
    std::shared_ptr<const Parent> parent_;
    Function transform_;
};

template<class S, class A, class F>
Result<A> fold(const S& input, A total, const F& combine, std::size_t first, std::size_t last) {
    for (auto i = first; i < last; ++i) {
        const typename S::value_type& value = input.read(i);
        auto combined_result = call(combine, std::move(total), value);
        if (!combined_result) {
            return combined_result.as_failure();
        }
        total = std::move(combined_result).assume_value();
    }
    return outcome::success(std::move(total));
}

template<class S, class A, class F>
Result<A> reduce(const S& input, const A& initial, const F& combine, Executor& executor) {
    const auto count = input.size();
    const auto workers = executor.worker_limit();
    if (workers <= 1 || count <= 1) {
        return fold(input, initial, combine, 0, count);
    }

    const auto block_size = std::max(std::size_t{2}, count / std::min(workers, count) / 8);
    const auto blocks = count / block_size + (count % block_size != 0);
    Values<A> partials(blocks);
    const auto reduced = for_each_index(blocks, executor, [&](std::size_t i) -> Result<void> {
        const auto first = i * block_size;
        const auto last = first + std::min(block_size, count - first);
        auto partial_result = fold(input, A(input.read(first)), combine, first + 1, last);
        if (!partial_result) {
            return partial_result.as_failure();
        }
        partials.put(i, std::move(partial_result).assume_value());
        return outcome::success();
    });
    if (!reduced) {
        return reduced.as_failure();
    }

    A total = initial;
    for (std::size_t i = 0; i < blocks; ++i) {
        auto combined_result = call(combine, std::move(total), partials.read(i));
        if (!combined_result) {
            return combined_result.as_failure();
        }
        total = std::move(combined_result).assume_value();
    }
    return outcome::success(std::move(total));
}

}  // namespace detail

using Executor = detail::Executor;

// Operations build reusable plans. Each collection evaluates the plan again.
// A calling thread can retain one idle team between collections.
// Borrowed sources must remain alive and structurally stable during evaluation.
// Parallel callbacks may run concurrently and must synchronize any shared state.
// Callbacks can return plain values (or void) or an Outcome result with an error code.
template<class Evaluate, bool InPlace>
class Pipeline {
    using Storage = detail::PlanStorage<Evaluate>;

public:
    using value_type = typename Storage::value_type;

    Pipeline(Evaluate evaluate, std::size_t workers)
        : evaluate_(std::make_shared<const Evaluate>(std::move(evaluate))), workers_(workers) {}

    auto iter() const { return Pipeline<Evaluate, true>(evaluate_, 1); }
    auto into_inter() const { return Pipeline<Evaluate, false>(evaluate_, workers_); }
    Pipeline iter_paral(std::size_t workers = 0) const { return Pipeline(evaluate_, workers); }

    template<class Function, bool Mutable = InPlace, std::enable_if_t<Mutable, int> = 0>
    auto map(Function transform) const { return update(std::move(transform)); }

    template<class Function, bool Mutable = InPlace, std::enable_if_t<!Mutable, int> = 0>
    auto map(Function transform) const {
        using Output = detail::Values<detail::CallbackValue<const Function&, const value_type&>>;
        return next([parent = evaluate_, transform = std::move(transform)](detail::Executor& executor)
                        -> Result<Output> {
            auto input_result = (*parent)(executor);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            Output output(input.size());
            using Callback = detail::Invocation<const Function&, const value_type&>;
            using ReturnsResult = typename detail::ResultTraits<std::decay_t<Callback>>::type;
            const auto map_index = [&](std::size_t i) -> decltype(auto) {
                const value_type& value = input.read(i);
                return detail::map_value(output, i, value, transform, ReturnsResult{});
            };
            const auto mapped = detail::for_each_index(input.size(), executor, map_index);
            if (!mapped) {
                return mapped.as_failure();
            }
            return detail::outcome::success(std::move(output));
        });
    }

    // A pure transform may run together with another stage on each element. It
    // must produce a value without side effects or errors, using only its input
    // and immutable state. Use map for callbacks with observable stage order.
    template<class Function, bool Mutable = InPlace,
             class Value = std::decay_t<detail::Invocation<const Function&, const value_type&>>,
             std::enable_if_t<!Mutable && !std::is_void<Value>::value && !detail::ResultTraits<Value>::value, int> = 0>
    auto map_pure(Function transform) const {
        return next(detail::MapPlan<Evaluate, Function>(evaluate_, std::move(transform)));
    }

    template<class Function>
    auto filter(Function predicate) const {
        using Output = detail::Values<value_type>;
        return next<false>([parent = evaluate_, predicate = std::move(predicate)](detail::Executor& executor)
                               -> Result<Output> {
            auto input_result = (*parent)(executor);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            std::vector<unsigned char> accepted(input.size());
            const auto test_index = [&](std::size_t i) -> Result<void> {
                const value_type& value = input.read(i);
                auto keep_result = detail::call(predicate, value);
                if (!keep_result) {
                    return keep_result.as_failure();
                }
                accepted[i] = bool(keep_result.assume_value());
                return detail::outcome::success();
            };
            const auto filtered = detail::for_each_index(input.size(), executor, test_index);
            if (!filtered) {
                return filtered.as_failure();
            }
            Output output;
            const auto count = std::count(accepted.begin(), accepted.end(), 1);
            output.reserve(static_cast<std::size_t>(count));
            for (std::size_t i = 0; i < input.size(); ++i) {
                if (accepted[i]) {
                    output.append(input.take(i));
                }
            }
            return detail::outcome::success(std::move(output));
        });
    }

    // Zip evaluates its left and right inputs in that order, under its own execution mode.
    template<class R, class Function = detail::MakePair, detail::EnableIterable<R> = 0>
    auto zip(R&& other, Function combine = {}) const {
        auto right = detail::as_pipeline(std::forward<R>(other), workers_);
        using RightValue = typename decltype(right)::value_type;
        using Value = detail::CallbackValue<const Function&, const value_type&, const RightValue&>;
        using Output = detail::Values<Value>;
        return next<false>([left = evaluate_, right = right.evaluate_, combine = std::move(combine)]
                           (detail::Executor& executor) -> Result<Output> {
            auto left_input_result = (*left)(executor);
            if (!left_input_result) {
                return left_input_result.as_failure();
            }
            auto& left_input = left_input_result.assume_value();
            auto right_input_result = (*right)(executor);
            if (!right_input_result) {
                return right_input_result.as_failure();
            }
            auto& right_input = right_input_result.assume_value();
            const auto count = std::min(left_input.size(), right_input.size());
            Output output(count);
            const auto zip_index = [&](std::size_t i) -> Result<void> {
                const value_type& a = left_input.read(i);
                const RightValue& b = right_input.read(i);
                auto combined_result = detail::call(combine, a, b);
                if (!combined_result) {
                    return combined_result.as_failure();
                }
                output.put(i, std::move(combined_result).assume_value());
                return detail::outcome::success();
            };
            const auto zipped = detail::for_each_index(count, executor, zip_index);
            if (!zipped) {
                return zipped.as_failure();
            }
            return detail::outcome::success(std::move(output));
        });
    }

    // The reducer must be associative. Parallel grouping preserves operand order
    // and applies the seed once. Use map or map_pure to change the element type.
    template<class Accumulator, class Function,
             std::enable_if_t<std::is_same<Accumulator, value_type>::value, int> = 0>
    auto reduce(Accumulator initial, Function combine) const {
        using Output = detail::Values<Accumulator>;
        return next<false>([parent = evaluate_, initial = std::move(initial), combine = std::move(combine)]
                           (detail::Executor& executor) -> Result<Output> {
            auto input_result = detail::prepare_values(*parent, executor);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            auto total_result = detail::reduce(input, initial, combine, executor);
            if (!total_result) {
                return total_result.as_failure();
            }
            Output output(1);
            output.put(0, std::move(total_result).assume_value());
            return detail::outcome::success(std::move(output));
        });
    }

    // Each callback modifies only its own element. A borrowed mutable range is updated in place.
    // Changes already made to elements persist if another callback reports an error.
    template<class Function, class S = Storage, std::enable_if_t<InPlace && detail::can_update<S>, int> = 0>
    auto update(Function change) const {
        static_assert(std::is_void<detail::CallbackValue<const Function&, value_type&>>::value,
                      "update expects a callback that returns void or Result<void>");
        return next([parent = evaluate_, change = std::move(change)](detail::Executor& executor)
                        -> Result<WritableStorage> {
            auto source_result = (*parent)(executor);
            if (!source_result) {
                return source_result.as_failure();
            }
            auto input_result = detail::writable_values(std::move(source_result).assume_value(), executor);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            const auto updated = detail::for_each_index(input.size(), executor, [&](std::size_t i) {
                return detail::call(change, input.write(i));
            });
            if (!updated) {
                return updated.as_failure();
            }
            return detail::outcome::success(std::move(input));
        });
    }

    // stop(state) runs before each round, including the first. Each compute(i, iter, state)
    // reads the previous layer through iter[i] and iter.size(). Both inputs are read-only.
    // advance(state, previous, next) returns the new state after all values have been computed.
    // A compute or advance failure leaves the last published layer in the source.
    // Layer views and their references are valid only during the callbacks that receive them.
    template<class State, class Compute, class Advance, class Stop, class S = Storage,
             std::enable_if_t<InPlace && detail::can_update<S>, int> = 0>
    auto layer_plan(State initial_state, Compute compute, Advance advance, Stop stop) const {
        return next([parent = evaluate_, initial_state = std::move(initial_state),
                     compute = std::move(compute), advance = std::move(advance), stop = std::move(stop)]
                    (detail::Executor& executor) -> Result<WritableStorage> {
            auto source_result = (*parent)(executor);
            if (!source_result) {
                return source_result.as_failure();
            }
            auto current_result = detail::writable_values(std::move(source_result).assume_value(), executor);
            if (!current_result) {
                return current_result.as_failure();
            }
            auto& current = current_result.assume_value();
            const auto size = current.size();
            auto state = initial_state;
            const State& state_view = state;
            detail::Values<value_type> pending;

            for (;;) {
                auto done_result = detail::call(stop, state_view);
                if (!done_result) {
                    return done_result.as_failure();
                }
                if (done_result.assume_value()) {
                    return detail::outcome::success(std::move(current));
                }
                pending.resize(size);
                const detail::LayerView<WritableStorage> previous(current);
                const auto compute_index = [&](std::size_t i) -> Result<void> {
                    auto computed_result = detail::call(compute, i, previous, state_view);
                    if (!computed_result) {
                        return computed_result.as_failure();
                    }
                    pending.put(i, std::move(computed_result).assume_value());
                    return detail::outcome::success();
                };
                const auto computed = detail::for_each_index(size, executor, compute_index);
                if (!computed) {
                    return computed.as_failure();
                }
                const detail::LayerView<detail::Values<value_type>> completed(pending);
                auto next_state_result = detail::call(advance, state_view, previous, completed);
                if (!next_state_result) {
                    return next_state_result.as_failure();
                }
                state = std::move(next_state_result).assume_value();
                const auto published = detail::publish_layer(current, pending, executor);
                if (!published) {
                    return published.as_failure();
                }
            }
        });
    }

    Result<std::vector<value_type>> collect() const noexcept {
        Executor executor(workers_, Executor::Reuse::calling_thread);
        return collect(executor);
    }

    // Borrows the executor for this call. Its worker limit controls execution.
    // Use one collection at a time per executor, including nested collections.
    Result<std::vector<value_type>> collect(Executor& executor) const noexcept {
        return detail::run_safe([&]() -> Result<std::vector<value_type>> {
            auto values_result = (*evaluate_)(executor);
            if (!values_result) {
                return values_result.as_failure();
            }
            return detail::outcome::success(detail::collect_values(values_result.assume_value()));
        });
    }

    Result<value_type> collect_one() const noexcept {
        return detail::run_safe([&]() -> Result<value_type> {
            detail::Executor executor(workers_, detail::Executor::Reuse::calling_thread);
            auto values_result = (*evaluate_)(executor);
            if (!values_result) {
                return values_result.as_failure();
            }
            auto& values = values_result.assume_value();
            if (values.size() != 1) {
                const auto error = std::make_error_code(std::errc::result_out_of_range);
                return detail::outcome::failure(error);
            }
            return detail::outcome::success(value_type(values.take(0)));
        });
    }

private:
    using WritableStorage = std::conditional_t<std::is_same<Storage, detail::RangeValues>::value,
                                               detail::Values<value_type>, Storage>;

    struct Traversal {
        Result<WritableStorage>& values() {
            if (!cache) {
                cache.emplace(detail::run_safe([&]() -> Result<WritableStorage> {
                    detail::Executor executor(workers, detail::Executor::Reuse::calling_thread);
                    auto source_result = (*evaluate)(executor);
                    if (!source_result) {
                        return source_result.as_failure();
                    }
                    return detail::writable_values(std::move(source_result).assume_value(), executor);
                }));
            }
            return *cache;
        }

        decltype(auto) at(std::size_t index, std::true_type) { return values().value().write(index); }
        decltype(auto) at(std::size_t index, std::false_type) { return values().value().read(index); }

        decltype(auto) at(std::size_t index) {
            return at(index, std::integral_constant<bool, InPlace && WritableStorage::writable>{});
        }

        std::shared_ptr<const Evaluate> evaluate;
        std::size_t workers;
        boost::optional<Result<WritableStorage>> cache;
    };

public:
    class iterator {
    public:
        using value_type = Pipeline::value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using reference = decltype(std::declval<Traversal&>().at(std::size_t{}));
        using pointer = std::add_pointer_t<std::remove_reference_t<reference>>;

        iterator() = default;
        reference operator*() const { return traversal_->at(index_); }
        pointer operator->() const { return std::addressof(operator*()); }

        // An execution failure ends traversal and is available through this iterator.
        ErrorCode error() const {
            if (traversal_) {
                const auto& result = traversal_->values();
                if (!result) {
                    return result.assume_error();
                }
            }
            return {};
        }

        iterator& operator++() { ++index_; return *this; }

        iterator operator++(int) {
            auto previous = *this;
            ++*this;
            return previous;
        }

        friend bool operator==(const iterator& a, const iterator& b) {
            if (a.traversal_ && b.traversal_) {
                return a.traversal_ == b.traversal_ && a.index_ == b.index_;
            }
            return a.at_end() && b.at_end();
        }

        friend bool operator!=(const iterator& a, const iterator& b) { return !(a == b); }

    private:
        friend class Pipeline;
        explicit iterator(std::shared_ptr<Traversal> traversal) : traversal_(std::move(traversal)) {}

        bool at_end() const {
            if (!traversal_) {
                return true;
            }
            const auto& result = traversal_->values();
            return !result || index_ == result.assume_value().size();
        }

        std::shared_ptr<Traversal> traversal_;
        std::size_t index_ = 0;
    };

    // The first traversal check or dereference collects once for that traversal.
    iterator begin() const { return iterator(std::make_shared<Traversal>(Traversal{evaluate_, workers_, {}})); }
    iterator end() const noexcept { return {}; }

private:
    template<class, bool>
    friend class Pipeline;

    Pipeline(std::shared_ptr<const Evaluate> evaluate, std::size_t workers)
        : evaluate_(std::move(evaluate)), workers_(workers) {}

    template<bool Mutable = InPlace, class Next>
    auto next(Next evaluate) const { return Pipeline<Next, Mutable>(std::move(evaluate), workers_); }

    std::shared_ptr<const Evaluate> evaluate_;
    std::size_t workers_;
};

namespace detail {

template<class R>
auto as_pipeline(R&& range, std::size_t workers, std::false_type) {
    auto evaluate = [source = hold_range(std::forward<R>(range), std::is_lvalue_reference<R>{})](Executor&) {
        return load(source.get());
    };
    return Pipeline<decltype(evaluate), true>(std::move(evaluate), workers);
}

}  // namespace detail

template<class R, detail::EnableIterable<R> = 0>
auto iter(R&& range) { return detail::as_pipeline(std::forward<R>(range), 1).iter(); }

template<class R, detail::EnableIterable<R> = 0>
auto into_inter(R&& range) { return detail::as_pipeline(std::forward<R>(range), 1).into_inter(); }

template<class R, detail::EnableIterable<R> = 0>
auto iter_paral(R&& range, std::size_t workers = 0) {
    return detail::as_pipeline(std::forward<R>(range), workers).iter_paral(workers);
}

inline auto Range::iter() const { return parallel::iter(Range(*this)); }
inline auto Range::into_inter() const { return parallel::into_inter(Range(*this)); }
inline auto Range::iter_paral(std::size_t workers) const { return parallel::iter_paral(Range(*this), workers); }

}  // namespace parallel

#endif
