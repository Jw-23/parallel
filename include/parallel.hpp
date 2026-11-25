#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/optional.hpp>
#include <boost/outcome.hpp>
#include <boost/outcome/policy/terminate.hpp>
#include <boost/outcome/std_result.hpp>
#include <boost/system/system_error.hpp>

#include <algorithm>
#include <atomic>
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
#include <tuple>
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

// Dynamic workers share chunks. Static workers each own a balanced contiguous partition.
// Work stealing uses local queues on sufficiently large batches and the dynamic queue otherwise.
enum class Schedule { dynamic, static_partition, work_stealing };

struct EvaluationOptions {
    // Borrowed for the call. Library scratch allocations run on the calling thread.
    boost::container::pmr::memory_resource* scratch = nullptr;
};

template<class Evaluate>
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
    auto into_inter_paral(std::size_t workers = 0) const;

private:
    value_type start_;
    value_type step_;
    std::size_t size_ = 0;
    ErrorCode error_;
};

namespace detail {

namespace outcome = boost::outcome_v2;

template<class...>
using void_t = void;

template<class F, class... Args, std::enable_if_t<!std::is_member_pointer<std::decay_t<F>>::value, int> = 0>
auto invoke(F&& function, Args&&... args) -> decltype(std::forward<F>(function)(std::forward<Args>(args)...)) {
    return std::forward<F>(function)(std::forward<Args>(args)...);
}

template<class F, class... Args, std::enable_if_t<std::is_member_pointer<std::decay_t<F>>::value, int> = 0>
auto invoke(F&& function, Args&&... args) -> decltype(std::mem_fn(function)(std::forward<Args>(args)...)) {
    return std::mem_fn(function)(std::forward<Args>(args)...);
}

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
using Invocation = decltype(detail::invoke(std::declval<Function>(), std::declval<Args>()...));

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
    enum class Reuse { evaluation, shared, calling_thread = shared };

    explicit Executor(std::size_t workers, Reuse reuse = Reuse::evaluation) noexcept;
    Executor(std::size_t workers, Schedule schedule, Reuse reuse = Reuse::evaluation) noexcept;
    ~Executor();

    std::size_t worker_limit() const noexcept { return worker_limit_; }
    Result<void> run_chunks(std::size_t size, const ChunkTask& run, std::size_t grain_hint = 0);

    template<class Function>
    Result<void> run_borrowed_chunks(std::size_t size, const Function& run, std::size_t grain_hint = 0) {
        // Borrow the callable for this synchronous batch, including plain function references.
        const auto callback = [&](std::size_t first, std::size_t last) -> decltype(auto) {
            return detail::invoke(run, std::move(first), std::move(last));
        };
        using Callback = decltype(callback);
        return submit(size, {&callback, [](const void* object, std::size_t first,
                                            std::size_t last, ErrorCode& error) {
            const auto& apply = *static_cast<const Callback*>(object);
            return invoke_chunk(apply, first, last, error, std::is_void<decltype(apply(first, last))>{});
        }}, grain_hint);
    }

private:
    struct ChunkCallback {
        const void* object;
        bool (*invoke)(const void*, std::size_t, std::size_t, ErrorCode&);

        bool operator()(std::size_t first, std::size_t last, ErrorCode& error) const {
            return invoke(object, first, last, error);
        }
    };

    struct Team;
    struct TeamCache;
    struct Batch;
    void acquire_team(std::size_t background_workers);
    Result<void> submit(std::size_t size, ChunkCallback run, std::size_t grain_hint);

    template<class Callback>
    static bool invoke_chunk(const Callback& apply, std::size_t first, std::size_t last,
                             ErrorCode&, std::true_type) {
        apply(first, last);
        return true;
    }

    template<class Callback>
    static bool invoke_chunk(const Callback& apply, std::size_t first, std::size_t last,
                             ErrorCode& error, std::false_type) {
        const auto result = chunk_result(apply(first, last));
        if (result) {
            return true;
        }
        error = result.assume_error();
        return false;
    }

    static Result<void> chunk_result(Result<void> result) { return result; }

    template<class E, class Policy>
    static Result<void> chunk_result(outcome::basic_result<void, E, Policy> result) {
        return into_result(std::move(result));
    }

    std::size_t worker_limit_;
    Reuse reuse_;
    Schedule schedule_;
    std::shared_ptr<TeamCache> cache_;
    std::unique_ptr<Team> team_;
};

template<class Function>
Result<void> for_each_index(std::size_t size, Executor& executor, const Function& apply, std::true_type) {
    return executor.run_borrowed_chunks(size, [&](std::size_t first, std::size_t last) {
        for (auto i = first; i < last; ++i) {
            detail::invoke(apply, i);
        }
    });
}

template<class Function>
Result<void> for_each_index(std::size_t size, Executor& executor, const Function& apply, std::false_type) {
    return executor.run_borrowed_chunks(size, [&](std::size_t first, std::size_t last) -> Result<void> {
        for (auto i = first; i < last; ++i) {
            const auto applied = call(apply, i);
            if (!applied) {
                return applied.as_failure();
            }
        }
        return outcome::success();
    });
}

template<class Function>
Result<void> for_each_index(std::size_t size, Executor& executor, const Function& apply) {
    return for_each_index(size, executor, apply, std::is_void<Invocation<const Function&, std::size_t&>>{});
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
using EnableOwnedRange = std::enable_if_t<IsIterable<R>::value && !std::is_lvalue_reference<R>::value &&
    !std::is_const<std::remove_reference_t<R>>::value && !std::is_array<std::remove_reference_t<R>>::value &&
    !IsInitializerList<std::decay_t<R>>::value, int>;

template<class R>
using EnableZipRange = std::enable_if_t<IsIterable<R>::value &&
    (std::is_lvalue_reference<R>::value ||
     (!std::is_array<std::remove_reference_t<R>>::value && !IsInitializerList<std::decay_t<R>>::value)), int>;

struct Evaluation {
    Executor& executor;
    EvaluationOptions options;
};

template<class T>
struct ScratchAllocator {
    using value_type = T;
    boost::container::pmr::memory_resource* resource = nullptr;

    ScratchAllocator() = default;
    explicit ScratchAllocator(const Evaluation& evaluation) noexcept : resource(evaluation.options.scratch) {}

    template<class U>
    ScratchAllocator(const ScratchAllocator<U>& other) noexcept : resource(other.resource) {}

    T* allocate(std::size_t size) {
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        return resource ? static_cast<T*>(resource->allocate(size * sizeof(T), alignof(T)))
                        : std::allocator<T>{}.allocate(size);
    }

    void deallocate(T* value, std::size_t size) noexcept {
        if (resource) {
            resource->deallocate(value, size * sizeof(T), alignof(T));
        } else {
            std::allocator<T>{}.deallocate(value, size);
        }
    }

    template<class U>
    bool operator==(const ScratchAllocator<U>& other) const noexcept { return resource == other.resource; }

    template<class U>
    bool operator!=(const ScratchAllocator<U>& other) const noexcept { return !(*this == other); }
};

template<class T, bool Writable = true, class Allocator = std::allocator<T>>
class Values {
public:
    using value_type = T;
    static constexpr bool writable = Writable;
    static constexpr bool fallible = false;
    // vector<bool> shares words between elements. Give each bool its own slot.
    static constexpr bool direct = std::is_arithmetic<T>::value && !std::is_same<T, bool>::value;

    explicit Values(std::size_t size = 0, const Allocator& allocator = {}) : slots(size, allocator) {}

    std::size_t size() const noexcept { return slots.size(); }
    const T& read(std::size_t index) const { return value(slots[index]); }
    T& write(std::size_t index) { return value(slots[index]); }
    decltype(auto) take(std::size_t index) { return std::move_if_noexcept(write(index)); }
    T&& argument(std::size_t index) { return std::move(write(index)); }

    void resize(std::size_t size) { slots.resize(size); }
    void reserve(std::size_t size) { slots.reserve(size); }
    void swap(Values& other) noexcept { slots.swap(other.slots); }

    template<class U>
    void put(std::size_t index, U&& input) { store(slots[index], std::forward<U>(input)); }

    template<class U>
    void append(U&& input) { append(std::forward<U>(input), std::integral_constant<bool, direct>{}); }

    auto release_vector() { return std::move(slots); }

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
    using SlotAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Slot>;
    std::vector<Slot, SlotAllocator> slots;
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

template<class Access, bool Consume>
class References {
    using Reference = Invocation<const Access&, std::size_t>;
    using Argument = std::conditional_t<Consume, std::remove_reference_t<Reference>&&, Reference>;

public:
    using value_type = std::remove_cv_t<std::remove_reference_t<Reference>>;
    static constexpr bool writable = !std::is_const<std::remove_reference_t<Reference>>::value;
    static constexpr bool fallible = false;

    References(std::size_t size, Access access) : size_(size), access_(std::move(access)) {}

    std::size_t size() const noexcept { return size_; }
    const value_type& read(std::size_t index) const { return access_(index); }
    Reference write(std::size_t index) { return access_(index); }
    using Taken = std::conditional_t<Consume, Argument, const value_type&>;
    Taken take(std::size_t index) const { return static_cast<Taken>(access_(index)); }
    Argument argument(std::size_t index) const { return static_cast<Argument>(access_(index)); }

private:
    std::size_t size_;
    Access access_;
};

struct RangeValues {
    using value_type = Range::value_type;
    static constexpr bool writable = false;
    static constexpr bool fallible = false;

    std::size_t size() const noexcept { return range.size(); }
    value_type read(std::size_t index) const noexcept { return range[index]; }
    value_type take(std::size_t index) const noexcept { return read(index); }
    value_type argument(std::size_t index) const noexcept { return read(index); }

    Range range;
};

template<class Input>
class Selection {
public:
    using value_type = typename Input::value_type;
    using Indices = std::vector<std::size_t, ScratchAllocator<std::size_t>>;
    static constexpr bool writable = false;
    static constexpr bool fallible = false;

    Selection(Input input, Indices indices) : input_(std::move(input)), indices_(std::move(indices)) {}

    std::size_t size() const noexcept { return indices_.size(); }
    decltype(auto) read(std::size_t index) const { return input_.read(indices_[index]); }
    decltype(auto) take(std::size_t index) { return input_.take(indices_[index]); }
    decltype(auto) argument(std::size_t index) { return input_.argument(indices_[index]); }

private:
    Input input_;
    Indices indices_;
};

template<class S>
constexpr bool can_publish_layer = S::writable || std::is_same<S, RangeValues>::value;

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

template<bool Consume, class Access>
auto make_references(std::size_t size, Access access) {
    return References<Access, Consume>(size, std::move(access));
}

template<bool Consume, class I>
auto load_iterators(I first, I last, std::random_access_iterator_tag, Evaluation&) {
    const auto size = first != last ? static_cast<std::size_t>(last - first) : 0;
    return make_references<Consume>(size, [first](std::size_t i) -> Dereference<I> {
        return first[static_cast<typename std::iterator_traits<I>::difference_type>(i)];
    });
}

template<bool Consume, class I, class S>
auto load_iterators(I first, S last, std::forward_iterator_tag, Evaluation& evaluation) {
    using Element = std::remove_reference_t<Dereference<I>>;
    std::vector<Element*, ScratchAllocator<Element*>> positions{ScratchAllocator<Element*>(evaluation)};
    for (; first != last; ++first) {
        positions.push_back(std::addressof(*first));
    }
    const auto size = positions.size();
    return make_references<Consume>(size, [positions = std::move(positions)](std::size_t i) -> Dereference<I> {
        return *positions[i];
    });
}

template<class I>
decltype(auto) load_item(I& first, std::false_type) { return *first; }

template<class I>
decltype(auto) load_item(I& first, std::true_type) {
    using Reference = Dereference<I>;
    // A proxy returned by value must leave this function by value.
    using Item = std::conditional_t<std::is_reference<Reference>::value, std::remove_reference_t<Reference>&&, Reference>;
    return static_cast<Item>(*first);
}

template<bool Consume, class I, class S>
auto load_iterators(I first, S last, std::input_iterator_tag, Evaluation&) {
    Values<typename IteratorValue<I>::type, false> result;
    for (; first != last; ++first) {
        result.append(load_item(first, std::integral_constant<bool, Consume>{}));
    }
    return result;
}

template<bool Consume>
Result<RangeValues> load(const Range& range, Evaluation&) {
    if (range.error()) {
        return outcome::failure(range.error());
    }
    return outcome::success(RangeValues{range});
}

template<bool Consume, class T>
auto load_buffer(T* data, std::size_t size) {
    auto input = make_references<Consume>(size, [data](std::size_t i) -> T& { return data[i]; });
    using Storage = decltype(input);
    if (data == nullptr && size != 0) {
        return Result<Storage>(outcome::failure(std::make_error_code(std::errc::invalid_argument)));
    }
    return Result<Storage>(outcome::success(std::move(input)));
}

template<bool Consume, class R, std::enable_if_t<!std::is_same<std::decay_t<R>, Range>::value, int> = 0>
auto load(R& range, Evaluation& evaluation) {
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
    auto input = load_iterators<Consume>(std::move(first), std::move(last), Access{}, evaluation);
    return Result<decltype(input)>(outcome::success(std::move(input)));
}

template<class Loader>
class ConsumingSource {
public:
    explicit ConsumingSource(Loader loader) : loader_(std::move(loader)) {}

    ConsumingSource(ConsumingSource&& other)
        : loader_(std::move(other.loader_)), consumed_(other.consumed_.load(std::memory_order_relaxed)) {}

    auto operator()(Evaluation& evaluation) const {
        using Output = Invocation<Loader&, Evaluation&>;
        if (consumed_.exchange(true, std::memory_order_relaxed)) {
            return Output(outcome::failure(std::make_error_code(std::errc::operation_not_permitted)));
        }
        return loader_(evaluation);
    }

private:
    mutable Loader loader_;
    mutable std::atomic<bool> consumed_{false};
};

template<class Evaluate>
auto as_pipeline(Pipeline<Evaluate> plan, std::size_t workers);

template<class R>
auto as_pipeline(R&& range, std::size_t workers);

struct MakePair {
    template<class A, class B>
    auto operator()(const A& a, const B& b) const { return std::pair<A, B>(a, b); }
};

template<class Evaluate>
using PlanStorage = typename Invocation<const Evaluate&, Evaluation&>::value_type;

template<class Input>
using Item = std::conditional_t<Input::fallible, typename Input::value_type&&,
                                decltype(std::declval<Input&>().argument(std::size_t{}))>;

template<class Function>
struct MapOperation {
    Function function;

    template<class Input>
    using Value = CallbackValue<const Function&, Item<Input>>;

    template<class Input>
    auto apply(std::size_t, Item<Input> item) const -> std::decay_t<Invocation<const Function&, Item<Input>>> {
        return detail::invoke(function, std::forward<Item<Input>>(item));
    }
};

struct EnumerateOperation {
    template<class Input>
    using Value = std::tuple<std::size_t, typename Input::value_type>;

    template<class Input>
    auto apply(std::size_t index, Item<Input> item) const -> std::tuple<std::size_t, Item<Input>> {
        return {index, std::forward<Item<Input>>(item)};
    }
};

template<class Input>
struct ChunkView {
    using value_type = typename Input::value_type;
    static constexpr bool fallible = Input::fallible;

    std::size_t size() const noexcept { return input.size(); }
    decltype(auto) argument(std::size_t index) { return input.argument(index); }

    Input& input;
};

template<class Input>
auto chunk_view(Input& input, int) -> decltype(input.for_chunk()) { return input.for_chunk(); }

inline RangeValues chunk_view(RangeValues& input, int) noexcept { return input; }

template<class Input>
ChunkView<Input> chunk_view(Input& input, long) noexcept { return {input}; }

template<class Input, class Operation>
class ElementReader {
    using Raw = decltype(std::declval<const Operation&>().template apply<Input>(
        std::size_t{}, std::declval<Item<Input>>()));
    using ReturnsResult = typename ResultTraits<Raw>::type;

public:
    using value_type = typename Operation::template Value<Input>;
    static constexpr bool fallible = Input::fallible || ReturnsResult::value;

    ElementReader(Input input, const Operation& operation)
        : input_(std::move(input)), operation_(operation) {}

    std::size_t size() const noexcept { return input_.size(); }

    decltype(auto) argument(std::size_t index) {
        return argument(index, std::integral_constant<bool, Input::fallible>{}, ReturnsResult{});
    }

    auto for_chunk() {
        using View = decltype(chunk_view(input_, 0));
        return ElementReader<View, Operation>(chunk_view(input_, 0), operation_);
    }

private:
    Raw argument(std::size_t index, std::false_type, std::false_type) {
        return operation_.template apply<Input>(index, input_.argument(index));
    }

    Result<value_type> argument(std::size_t index, std::false_type, std::true_type) {
        return into_result(operation_.template apply<Input>(index, input_.argument(index)));
    }

    template<class Wrap>
    Result<value_type> argument(std::size_t index, std::true_type, Wrap wrap) {
        auto input = input_.argument(index);
        if (!input) {
            return input.as_failure();
        }
        return normalize(operation_.template apply<Input>(index, std::move(input).assume_value()), wrap);
    }

    static Result<value_type> normalize(Raw raw, std::true_type) { return into_result(std::move(raw)); }
    static Result<value_type> normalize(Raw raw, std::false_type) {
        return outcome::success(value_type(std::move(raw)));
    }

    Input input_;
    const Operation& operation_;
};

template<class Parent, class Operation>
class ElementPlan;

template<class Evaluate>
auto prepare_values(const Evaluate& evaluate, Evaluation& evaluation) {
    return evaluate(evaluation);
}

template<class Parent, class Operation>
auto prepare_values(const ElementPlan<Parent, Operation>& evaluate, Evaluation& evaluation) {
    return evaluate.prepare(evaluation);
}

template<class Parent>
using PreparedStorage = typename decltype(prepare_values(std::declval<const Parent&>(),
                                                         std::declval<Evaluation&>()))::value_type;

template<class Input, class Apply>
decltype(auto) visit_item(Input& input, std::size_t index, const Apply& apply, std::false_type) {
    return apply(input.argument(index));
}

template<class Input, class Apply>
auto visit_item(Input& input, std::size_t index, const Apply& apply, std::true_type)
    -> decltype(apply(std::declval<Item<Input>>())) {
    auto value = input.argument(index);
    if (!value) {
        return value.as_failure();
    }
    return apply(std::move(value).assume_value());
}

template<class Input, class Output>
void store_chunk(Input& input, Output& output, std::size_t first, std::size_t last, std::false_type) {
    for (auto i = first; i < last; ++i) {
        output.put(i, input.argument(i));
    }
}

template<class Input, class Output>
Result<void> store_chunk(Input& input, Output& output, std::size_t first, std::size_t last, std::true_type) {
    for (auto i = first; i < last; ++i) {
        auto item = input.argument(i);
        if (!item) {
            return item.as_failure();
        }
        output.put(i, std::move(item).assume_value());
    }
    return outcome::success();
}

template<class Input>
auto materialize_values(Input& input, Executor& executor) -> Result<Values<typename Input::value_type>> {
    Values<typename Input::value_type> output(input.size());
    const auto finished = executor.run_borrowed_chunks(input.size(), [&](std::size_t first, std::size_t last)
                                                       -> decltype(auto) {
        // Copy range metadata into the chunk so it cannot alias the output.
        auto reader = chunk_view(input, 0);
        return store_chunk(reader, output, first, last, std::integral_constant<bool, Input::fallible>{});
    });
    if (!finished) {
        return finished.as_failure();
    }
    return outcome::success(std::move(output));
}

template<class Parent, class Operation>
class ElementPlan {
    using Reader = ElementReader<PreparedStorage<Parent>, Operation>;

public:
    ElementPlan(std::shared_ptr<const Parent> parent, Operation operation)
        : parent_(std::move(parent)), operation_(std::move(operation)) {}

    Result<Reader> prepare(Evaluation& evaluation) const {
        auto input = prepare_values(*parent_, evaluation);
        if (!input) {
            return input.as_failure();
        }
        return outcome::success(Reader(std::move(input).assume_value(), operation_));
    }

    Result<Values<typename Reader::value_type>> operator()(Evaluation& evaluation) const {
        auto prepared = prepare(evaluation);
        if (!prepared) {
            return prepared.as_failure();
        }
        return materialize_values(prepared.assume_value(), evaluation.executor);
    }

    const Parent& parent() const noexcept { return *parent_; }

private:
    std::shared_ptr<const Parent> parent_;
    Operation operation_;
};

template<class Storage, std::enable_if_t<Storage::writable, int> = 0>
decltype(auto) traverse_item(Storage& values, std::size_t i) { return values.write(i); }

template<class Storage, std::enable_if_t<!Storage::writable, int> = 0>
decltype(auto) traverse_item(Storage& values, std::size_t i) { return values.read(i); }

template<class Storage>
struct EnumeratedTraversal {
    static constexpr bool writable = false;

    std::size_t size() const noexcept { return values.size(); }
    auto read(std::size_t i) -> std::tuple<std::size_t, decltype(traverse_item(std::declval<Storage&>(), i))> {
        return {i, traverse_item(values, i)};
    }

    Storage values;
};

template<class Plan>
auto traverse(const Plan& plan, Evaluation& evaluation) {
    return plan(evaluation);
}

template<class Parent>
auto traverse(const ElementPlan<Parent, EnumerateOperation>& plan, Evaluation& evaluation) {
    auto input = traverse(plan.parent(), evaluation);
    using Storage = EnumeratedTraversal<typename decltype(input)::value_type>;
    if (!input) {
        return Result<Storage>(input.as_failure());
    }
    return Result<Storage>(outcome::success(Storage{std::move(input).assume_value()}));
}

template<class S, class A, class F>
Result<A> fold(S& input, A total, const F& combine, std::size_t first, std::size_t last) {
    for (auto i = first; i < last; ++i) {
        auto combined = visit_item(input, i, [&](auto&& value) {
            return call(combine, std::move(total), std::forward<decltype(value)>(value));
        }, std::integral_constant<bool, S::fallible>{});
        if (!combined) {
            return combined.as_failure();
        }
        total = std::move(combined).assume_value();
    }
    return outcome::success(std::move(total));
}

template<class S, class A, class F>
Result<A> reduce(S& input, const A& initial, const F& combine, Evaluation& evaluation) {
    const auto count = input.size();
    const auto workers = evaluation.executor.worker_limit();
    if (workers <= 1 || count <= 1) {
        return fold(input, initial, combine, 0, count);
    }

    const auto block_size = std::max(std::size_t{2}, count / std::min(workers, count) / 8);
    const auto blocks = count / block_size + (count % block_size != 0);
    Values<A, true, ScratchAllocator<A>> partials(blocks, ScratchAllocator<A>(evaluation));
    const auto reduced = for_each_index(blocks, evaluation.executor, [&](std::size_t i) -> Result<void> {
        const auto first = i * block_size;
        const auto last = first + std::min(block_size, count - first);
        auto partial_result = visit_item(input, first, [&](auto&& value) {
            return fold(input, A(std::forward<decltype(value)>(value)), combine, first + 1, last);
        }, std::integral_constant<bool, S::fallible>{});
        if (!partial_result) {
            return partial_result.as_failure();
        }
        partials.put(i, std::move(partial_result).assume_value());
        return outcome::success();
    });
    if (!reduced) {
        return reduced.as_failure();
    }

    return fold(partials, initial, combine, 0, blocks);
}

}  // namespace detail

using Executor = detail::Executor;

// Run `apply(first, last)` over arbitrary half-open chunks of [0, size). This
// is the chunk-level primitive behind the collection pipeline; it is useful
// when the body is a cache-friendly loop over several arrays rather than a
// per-element transform. The calling thread participates, and idle teams can
// be reused across callers. A callback may return void or Result<void>. A
// nonzero grain hint limits chunk width when the batch runs in parallel.
template<class Function>
Result<void> for_each_chunk(std::size_t size, std::size_t workers, const Function& apply,
                            std::size_t grain_hint = 0) {
    Executor executor(workers, Executor::Reuse::shared);
    return executor.run_borrowed_chunks(size, [&](std::size_t first, std::size_t last) -> decltype(auto) {
        return detail::invoke(apply, first, last);
    }, grain_hint);
}

// Borrowed sources must stay alive and structurally stable during evaluation.
// Owning sources may be evaluated once. Copies share the same source.
// Parallel callbacks may run concurrently and must synchronize shared state.
template<class Evaluate>
class Pipeline {
    using Storage = detail::PlanStorage<Evaluate>;

public:
    using value_type = typename Storage::value_type;

    Pipeline(Evaluate evaluate, std::size_t workers)
        : evaluate_(std::make_shared<const Evaluate>(std::move(evaluate))), workers_(workers) {}

    // Consecutive element operations run together for each element. Use
    // materialize() when a stage must finish before the next can start.
    template<class Function, class Parent = Evaluate,
             class Value = detail::CallbackValue<const Function&, detail::Item<detail::PreparedStorage<Parent>>>,
             std::enable_if_t<!std::is_void<Value>::value, int> = 0>
    auto map(Function transform) const {
        using Operation = detail::MapOperation<Function>;
        return next(detail::ElementPlan<Evaluate, Operation>(evaluate_, Operation{std::move(transform)}));
    }

    auto enumerate() const {
        using Operation = detail::EnumerateOperation;
        return next(detail::ElementPlan<Evaluate, Operation>(evaluate_, Operation{}));
    }

    auto materialize() const {
        return next([parent = evaluate_](detail::Evaluation& evaluation) {
            return (*parent)(evaluation);
        });
    }

    template<class Function>
    auto filter(Function predicate) const {
        using Output = detail::Selection<Storage>;
        return next([parent = evaluate_, predicate = std::move(predicate)](detail::Evaluation& evaluation)
                               -> Result<Output> {
            auto input_result = (*parent)(evaluation);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            using Allocator = detail::ScratchAllocator<unsigned char>;
            std::vector<unsigned char, Allocator> accepted(input.size(), Allocator(evaluation));
            const auto test_index = [&](std::size_t i) -> Result<void> {
                const value_type& value = input.read(i);
                auto keep_result = detail::call(predicate, value);
                if (!keep_result) {
                    return keep_result.as_failure();
                }
                accepted[i] = bool(keep_result.assume_value());
                return detail::outcome::success();
            };
            const auto filtered = detail::for_each_index(input.size(), evaluation.executor, test_index);
            if (!filtered) {
                return filtered.as_failure();
            }
            const auto count = std::count(accepted.begin(), accepted.end(), 1);
            typename Output::Indices indices{detail::ScratchAllocator<std::size_t>(evaluation)};
            indices.reserve(static_cast<std::size_t>(count));
            for (std::size_t i = 0; i < input.size(); ++i) {
                if (accepted[i]) {
                    indices.push_back(i);
                }
            }
            return detail::outcome::success(Output(std::move(input), std::move(indices)));
        });
    }

    // Zip evaluates its left and right inputs in that order, under its own execution mode.
    template<class R, class Function = detail::MakePair, detail::EnableZipRange<R> = 0>
    auto zip(R&& other, Function combine = {}) const {
        auto right = detail::as_pipeline(std::forward<R>(other), workers_);
        using RightValue = typename decltype(right)::value_type;
        using Value = detail::CallbackValue<const Function&, const value_type&, const RightValue&>;
        using Output = detail::Values<Value>;
        return next([left = evaluate_, right = right.evaluate_, combine = std::move(combine)]
                           (detail::Evaluation& evaluation) -> Result<Output> {
            auto left_input_result = (*left)(evaluation);
            if (!left_input_result) {
                return left_input_result.as_failure();
            }
            auto& left_input = left_input_result.assume_value();
            auto right_input_result = (*right)(evaluation);
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
            const auto zipped = detail::for_each_index(count, evaluation.executor, zip_index);
            if (!zipped) {
                return zipped.as_failure();
            }
            return detail::outcome::success(std::move(output));
        });
    }

    // The reducer must be associative. Parallel grouping preserves operand order
    // and applies the seed once. Use map to change the element type.
    template<class Accumulator, class Function,
             std::enable_if_t<std::is_same<Accumulator, value_type>::value, int> = 0>
    auto reduce(Accumulator initial, Function combine) const {
        using Output = detail::Values<Accumulator>;
        return next([parent = evaluate_, initial = std::move(initial), combine = std::move(combine)]
                           (detail::Evaluation& evaluation) -> Result<Output> {
            auto input_result = detail::prepare_values(*parent, evaluation);
            if (!input_result) {
                return input_result.as_failure();
            }
            auto& input = input_result.assume_value();
            auto total_result = detail::reduce(input, initial, combine, evaluation);
            if (!total_result) {
                return total_result.as_failure();
            }
            Output output(1);
            output.put(0, std::move(total_result).assume_value());
            return detail::outcome::success(std::move(output));
        });
    }

    // stop(state) runs before each round, including the first. Each compute(i, iter, state)
    // reads the previous layer through iter[i] and iter.size(). Both inputs are read-only.
    // advance(state, previous, next) returns the new state after all values have been computed.
    // A compute or advance failure leaves the last published layer in the source.
    // Layer views and their references are valid only during the callbacks that receive them.
    template<class State, class Compute, class Advance, class Stop, class S = Storage,
             std::enable_if_t<detail::can_publish_layer<S>, int> = 0>
    auto layer_plan(State initial_state, Compute compute, Advance advance, Stop stop) const {
        return next([parent = evaluate_, initial_state = std::move(initial_state),
                     compute = std::move(compute), advance = std::move(advance), stop = std::move(stop)]
                    (detail::Evaluation& evaluation) -> Result<WritableStorage> {
            auto source_result = (*parent)(evaluation);
            if (!source_result) {
                return source_result.as_failure();
            }
            auto current_result = detail::writable_values(std::move(source_result).assume_value(), evaluation.executor);
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
                const auto computed = detail::for_each_index(size, evaluation.executor, compute_index);
                if (!computed) {
                    return computed.as_failure();
                }
                const detail::LayerView<detail::Values<value_type>> completed(pending);
                auto next_state_result = detail::call(advance, state_view, previous, completed);
                if (!next_state_result) {
                    return next_state_result.as_failure();
                }
                state = std::move(next_state_result).assume_value();
                const auto published = detail::publish_layer(current, pending, evaluation.executor);
                if (!published) {
                    return published.as_failure();
                }
            }
        });
    }

    Result<std::vector<value_type>> collect(EvaluationOptions options = {}) const noexcept {
        Executor executor(workers_, Executor::Reuse::shared);
        return collect(executor, options);
    }

    // An explicit executor controls worker count and scheduling for this call.
    Result<std::vector<value_type>> collect(Executor& executor, EvaluationOptions options = {}) const noexcept {
        return detail::run_safe([&]() -> Result<std::vector<value_type>> {
            detail::Evaluation evaluation{executor, options};
            auto values_result = (*evaluate_)(evaluation);
            if (!values_result) {
                return values_result.as_failure();
            }
            return detail::outcome::success(detail::collect_values(values_result.assume_value()));
        });
    }

    Result<value_type> collect_one(EvaluationOptions options = {}) const noexcept {
        Executor executor(workers_, Executor::Reuse::shared);
        return collect_one(executor, options);
    }

    Result<value_type> collect_one(Executor& executor, EvaluationOptions options = {}) const noexcept {
        return detail::run_safe([&]() -> Result<value_type> {
            detail::Evaluation evaluation{executor, options};
            auto values_result = (*evaluate_)(evaluation);
            if (!values_result) {
                return values_result.as_failure();
            }
            auto& values = values_result.assume_value();
            if (values.size() != 1) {
                return detail::outcome::failure(std::make_error_code(std::errc::result_out_of_range));
            }
            return detail::outcome::success(value_type(values.take(0)));
        });
    }

private:
    using WritableStorage = std::conditional_t<std::is_same<Storage, detail::RangeValues>::value,
                                               detail::Values<value_type>, Storage>;
    using TraversalResult = decltype(detail::traverse(std::declval<const Evaluate&>(),
                                                     std::declval<detail::Evaluation&>()));

    struct Traversal {
        TraversalResult& values() {
            if (!cache) {
                cache.emplace(detail::run_safe([&]() -> TraversalResult {
                    Executor executor(workers, Executor::Reuse::shared);
                    detail::Evaluation evaluation{executor, {}};
                    return detail::traverse(*evaluate, evaluation);
                }));
            }
            return *cache;
        }

        decltype(auto) at(std::size_t index) { return detail::traverse_item(values().value(), index); }

        std::shared_ptr<const Evaluate> evaluate;
        std::size_t workers;
        boost::optional<TraversalResult> cache;
    };

    template<class T>
    struct ArrowProxy {
        T value;
        const T* operator->() const noexcept { return std::addressof(value); }
    };

public:
    class iterator {
    public:
        using value_type = Pipeline::value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using reference = decltype(std::declval<Traversal&>().at(std::size_t{}));
        using IsReference = std::is_lvalue_reference<reference>;
        using pointer = std::conditional_t<IsReference::value, std::add_pointer_t<std::remove_reference_t<reference>>,
                                           ArrowProxy<reference>>;

        iterator() = default;
        reference operator*() const { return traversal_->at(index_); }
        pointer operator->() const { return arrow(IsReference{}); }

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

        pointer arrow(std::true_type) const { return std::addressof(operator*()); }
        pointer arrow(std::false_type) const { return {operator*()}; }

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
    template<class>
    friend class Pipeline;

    template<class Next>
    auto next(Next evaluate) const { return Pipeline<Next>(std::move(evaluate), workers_); }

    std::shared_ptr<const Evaluate> evaluate_;
    std::size_t workers_;
};

namespace detail {

template<class Load>
auto borrowed_pipeline(Load load, std::size_t workers) {
    return Pipeline<Load>(std::move(load), workers);
}

template<class Load>
auto consuming_pipeline(Load load, std::size_t workers) {
    return Pipeline<ConsumingSource<Load>>(ConsumingSource<Load>(std::move(load)), workers);
}

template<class R>
auto borrow_range(R& range, std::size_t workers) {
    return borrowed_pipeline([source = std::ref(range)](Evaluation& evaluation) {
        return load<false>(source.get(), evaluation);
    }, workers);
}

inline auto generated_range(Range range, std::size_t workers) {
    return borrowed_pipeline([range](Evaluation& evaluation) {
        return load<false>(range, evaluation);
    }, workers);
}

template<class T>
auto borrow_buffer(T* data, std::size_t size, std::size_t workers) {
    return borrowed_pipeline([=](Evaluation&) { return load_buffer<false>(data, size); }, workers);
}

template<class R>
auto consume_range(R&& range, std::size_t workers) {
    return consuming_pipeline([source = std::forward<R>(range)](Evaluation& evaluation) mutable {
        return load<true>(source, evaluation);
    }, workers);
}

template<class T, class Deleter>
auto consume_buffer(std::unique_ptr<T[], Deleter> data, std::size_t size, std::size_t workers) {
    return consuming_pipeline([source = std::move(data), size](Evaluation&) {
        return load_buffer<true>(source.get(), size);
    }, workers);
}

template<class R>
auto zip_source(R&& range, std::size_t workers, std::true_type) {
    return borrow_range(range, workers);
}

template<class R>
struct RetainedRange {
    mutable R range;

    auto operator()(Evaluation& evaluation) const { return load<false>(range, evaluation); }
};

template<class R>
auto zip_source(R&& range, std::size_t workers, std::false_type) {
    // A temporary right-hand range is retained for reusable zipped plans.
    return borrowed_pipeline(RetainedRange<std::decay_t<R>>{std::forward<R>(range)}, workers);
}

template<class Evaluate>
auto as_pipeline(Pipeline<Evaluate> plan, std::size_t) {
    return plan;
}

template<class R>
auto as_pipeline(R&& range, std::size_t workers) {
    return zip_source(std::forward<R>(range), workers, std::is_lvalue_reference<R>{});
}

}  // namespace detail

// Borrowing entries preserve references when the source provides stable ones.
template<class R, std::enable_if_t<detail::IsIterable<R>::value, int> = 0>
auto iter(R& range) { return detail::borrow_range(range, 1); }

template<class R, std::enable_if_t<detail::IsIterable<R>::value, int> = 0>
auto iter_paral(R& range, std::size_t workers = 0) { return detail::borrow_range(range, workers); }

template<class T, std::enable_if_t<std::is_object<T>::value, int> = 0>
auto iter(T* data, std::size_t size) { return detail::borrow_buffer(data, size, 1); }

// A single pointer denotes one element. A typed null pointer can be used with size zero.
template<class Pointer, class T = std::remove_pointer_t<std::decay_t<Pointer>>,
         std::enable_if_t<std::is_pointer<std::remove_reference_t<Pointer>>::value &&
                          std::is_object<T>::value, int> = 0>
auto iter(Pointer&& data) { return iter(data, 1); }

template<class T, std::enable_if_t<std::is_object<T>::value, int> = 0>
auto iter_paral(T* data, std::size_t size, std::size_t workers = 0) {
    return detail::borrow_buffer(data, size, workers);
}

template<class R, detail::EnableOwnedRange<R> = 0>
auto into_inter(R&& range) { return detail::consume_range(std::forward<R>(range), 1); }

template<class R, detail::EnableOwnedRange<R> = 0>
auto into_inter_paral(R&& range, std::size_t workers = 0) {
    return detail::consume_range(std::forward<R>(range), workers);
}

template<class T, class Deleter>
auto into_inter(std::unique_ptr<T[], Deleter>&& data, std::size_t size) {
    return detail::consume_buffer(std::move(data), size, 1);
}

template<class T, class Deleter>
auto into_inter_paral(std::unique_ptr<T[], Deleter>&& data, std::size_t size, std::size_t workers = 0) {
    return detail::consume_buffer(std::move(data), size, workers);
}

inline auto Range::iter() const { return detail::generated_range(*this, 1); }
inline auto Range::iter_paral(std::size_t workers) const { return detail::generated_range(*this, workers); }
inline auto Range::into_inter() const { return parallel::into_inter(Range(*this)); }
inline auto Range::into_inter_paral(std::size_t workers) const { return parallel::into_inter_paral(Range(*this), workers); }

}  // namespace parallel

#endif
