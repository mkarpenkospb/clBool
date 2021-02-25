#pragma once

#include <type_traits>
#include <matrix_dcsr.hpp>
#include <cpu_matrices.hpp>
#include "cl_includes.hpp"
#include "controls.hpp"
#include "cstdlib"

namespace utils {

    template<typename, typename = std::void_t<>>
    struct proper_container : std::false_type { };

    template<typename T>
    struct proper_container<T, std::void_t<typename T::iterator,
            typename T::const_iterator,
            typename T::value_type>> : std::true_type { };

    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu);

    void fill_random_buffer(cpu_buffer &buf, uint32_t seed = -1);

    void fill_random_buffer(cpu_buffer_f &buf, uint32_t seed = -1);

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    Controls create_controls();

    std::string error_name(cl_int error);

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);

    template <typename buf>
    void print_cpu_buffer(const buf &buffer, uint32_t size = -1) {
        uint32_t end = size;
        if (size == -1) end = buffer.size();
        for (uint32_t i = 0; i < end; ++i) {
            std::cout << buffer[i] << ", ";
        }
        std::cout << std::endl;
    }

    // https://gist.github.com/donny-dont/1471329
    template <typename T, std::size_t Alignment>
    class aligned_allocator
    {
    public:

        // The following will be the same for virtually all allocators.
        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T value_type;
        typedef std::size_t size_type;
        typedef ptrdiff_t difference_type;

        T * address(T& r) const
        {
            return &r;
        }

        const T * address(const T& s) const
        {
            return &s;
        }

        std::size_t max_size() const
        {
            // The following has been carefully written to be independent of
            // the definition of size_t and to avoid signed/unsigned warnings.
            return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
        }


        // The following must be the same for all allocators.
        template <typename U>
        struct rebind
        {
            typedef aligned_allocator<U, Alignment> other;
        } ;

        bool operator!=(const aligned_allocator& other) const
        {
            return !(*this == other);
        }

        void construct(T * const p, const T& t) const
        {
            void * const pv = static_cast<void *>(p);

            new (pv) T(t);
        }

        void destroy(T * const p) const
        {
            p->~T();
        }

        // Returns true if and only if storage allocated from *this
        // can be deallocated from other, and vice versa.
        // Always returns true for stateless allocators.
        bool operator==(const aligned_allocator& other) const
        {
            return true;
        }


        // Default constructor, copy constructor, rebinding constructor, and destructor.
        // Empty for stateless allocators.
        aligned_allocator() { }

        aligned_allocator(const aligned_allocator&) { }

        template <typename U> aligned_allocator(const aligned_allocator<U, Alignment>&) { }

        ~aligned_allocator() { }


        // The following will be different for each allocator.
        T * allocate(const std::size_t n) const
        {
            // The return value of allocate(0) is unspecified.
            // Mallocator returns NULL in order to avoid depending
            // on malloc(0)'s implementation-defined behavior
            // (the implementation can define malloc(0) to return NULL,
            // in which case the bad_alloc check below would fire).
            // All allocators can return NULL in this case.
            if (n == 0) {
                return NULL;
            }

            // All allocators should contain an integer overflow check.
            // The Standardization Committee recommends that std::length_error
            // be thrown in the case of integer overflow.
            if (n > max_size())
            {
                throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
            }

            // Mallocator wraps malloc().
            void * const pv = _mm_malloc(n * sizeof(T), Alignment);

            // Allocators should throw std::bad_alloc in the case of memory allocation failure.
            if (pv == NULL)
            {
                throw std::bad_alloc();
            }

            return static_cast<T *>(pv);
        }

        void deallocate(T * const p, const std::size_t n) const
        {
            _mm_free(p);
        }


        // The following will be the same for all allocators that ignore hints.
        template <typename U>
        T * allocate(const std::size_t n, const U * /* const hint */) const
        {
            return allocate(n);
        }


        // Allocators are not required to be assignable, so
        // all allocators should have a private unimplemented
        // assignment operator. Note that this will trigger the
        // off-by-default (enabled under /Wall) warning C4626
        // "assignment operator could not be generated because a
        // base class assignment operator is inaccessible" within
        // the STL headers, but that warning is useless.
    private:
        aligned_allocator& operator=(const aligned_allocator&);
    };

    template<typename T>
    void compare_buffers(Controls &controls,
                            const cl::Buffer &buffer_g, const std::vector<T> &buffer_c, uint32_t size,
                            std::string name = "") {
//        void *allocator = _aligned_malloc(sizeof(T), 64);
        using buf = std::vector<T, aligned_allocator<T, 64>>;
        static float epsilon = 0.00001;
        buf cpu_copy(size);
        timer t;
        t.restart();
        cl::Event ev;

        controls.queue.enqueueReadBuffer(buffer_g, CL_TRUE, 0, sizeof(typename buf::value_type) * cpu_copy.size(),
                                         cpu_copy.data(), nullptr, &ev);

        double time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "read buffer in " << time << " \n";

        for (uint32_t i = 0; i < size; ++i) {
            typename buf::value_type d = cpu_copy[i] - buffer_c[i];
            if (d >  epsilon || d < -epsilon) {
                uint32_t start = std::max(0, (int) i - 10);
                uint32_t stop = std::min(size, i + 10);
                for (uint32_t j = start; j < stop; ++j) {
                    std::cout << j << ": (" << cpu_copy[j] << ", " << buffer_c[j] << "), ";
                }
                std::cout << std::endl;
                throw std::runtime_error("buffers for " + name + " are different");
            }
        }
        std::cout << "buffers are equal" << std::endl;
    }

    void program_handler(const cl::Error &e, const cl::Program &program,
                         const cl::Device &device, const std::string &name);

    void show_devices();
//    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const coo_utils::matrix_dcsr_cpu &m, uint32_t size);
}