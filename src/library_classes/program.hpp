#pragma once

#include "utils.hpp"

using program_id = std::string; // program name
using kernel_id = std::pair<program_id, std::string>; // program name, kernel name


struct KernelCache {
    // https://stackoverflow.com/a/32685618
    struct pair_hash {
        template<class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

    static inline std::unordered_map<program_id, cl::Program> programs{};
    static inline std::unordered_map<kernel_id, cl::Kernel, pair_hash> kernels{};

    static const cl::Program &get_program(const Controls &controls, const std::string &_program_name,
                                          const std::string options = "") {
        cl::Program cl_program;
        try {
            if (KernelCache::programs.find(_program_name) != KernelCache::programs.end()) {
                return programs[_program_name];
            }
            #ifndef FPGA
            cl_program = controls.create_program_from_source(_kernel, _kernel_length);
            std::stringstream options;
            options <<  options_str << " -D RUN " << " -D GROUP_SIZE=" << _block_size;
            cl_program.build(options.str().c_str());
            #else
            timer t;
            cl_program = controls.create_program_from_binaries(_program_name);
            cl_program.build(options.c_str());
            KernelCache::programs[_program_name] = cl_program;
            double time = t.elapsed();
            if (DEBUG_ENABLE) *logger << "program " << _program_name << " created in " << time << " \n";
            return KernelCache::programs[_program_name];
            #endif
        } catch (const cl::Error &e) {
            utils::program_handler(e, cl_program, controls.device, _program_name);
        }
    }

    static const cl::Kernel &get_kernel(const Controls &controls, const kernel_id &id, const std::string& options = "") {
        cl::Program cl_program = get_program(controls, id.first, options);
        if (kernels.find(id) != kernels.end()) {
            return kernels[id];
        }
        timer t;
        kernels[id] = cl::Kernel(cl_program, id.second.c_str());
        double time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "kernel " << id.second << " created in " << time << " \n";
        return kernels[id];
    }

};


template<typename ... Args>
class program {
public:
    using kernel_type = cl::KernelFunctor<Args...>;
private:
//#ifndef FPGA
    const char *_kernel = "";
    uint32_t _kernel_length = 0;
//#else
    std::string _program_name;
//#endif
    std::string _kernel_name;
    uint32_t _block_size = 0;
    uint32_t _needed_work_size = 0;
    cl::Program cl_program;
    bool _async = false;
    bool _is_spec_queue = false;
    cl::CommandQueue* _queue;

    #ifdef WIN
    std::string options_str = "-D GPU=1";
    #endif

    void check_completeness(const Controls &controls) {
        if (_block_size == 0) _block_size = controls.block_size;
        #ifndef FPGA
        if (_kernel_length == 0) throw std::runtime_error("zero kernel length");
        #endif
        if (_kernel_name == "") throw std::runtime_error("no kernel name");
        if (_needed_work_size == 0) throw std::runtime_error("zero global_work_size");
    }

public:
    program() = default;

    explicit program(const char *kernel, uint32_t kernel_length)
            : _kernel(kernel), _kernel_length(kernel_length) {}

    program &set_sources(const char *kernel, uint32_t kernel_length) {
        _kernel = kernel;
        _kernel_length = kernel_length;
        return *this;
    }

    program &add_option(std::string name, std::string value = "") {
        #ifndef FPGA
        options_str += (" -D " + name + "=" + value);
        #endif
        return *this;
    }

    template<typename OptionType>
    program &add_option(std::string name, const OptionType &value) {
        #ifndef FPGA
        options_str += (" -D " + name + "=" + std::to_string(value));
        _built = false;
        #endif
        return *this;
    }

    explicit program(std::string program_name) : _program_name(program_name) {}

    program &set_kernel_name(std::string kernel_name) {
        _kernel_name = std::move(kernel_name);
        return *this;
    }

    program &set_block_size(uint32_t block_size) {
        _block_size = block_size;
        return *this;
    }

    program &set_needed_work_size(uint32_t needed_work_size) {
        _needed_work_size = needed_work_size;
        return *this;
    }

    program &set_async(bool async) {
        _async = async;
        return *this;
    }

    program &set_queue(cl::CommandQueue& queue) {
        _is_spec_queue = true;
        _queue = &queue;
    }

//    void build(Controls &controls) {
//        build_cl_program(controls);
//        _built = true;
//    }

    cl::Event run(Controls &controls, Args ... args) {
        check_completeness(controls);
        try {
            cl::Kernel kernel = KernelCache::get_kernel(controls, {_program_name, _kernel_name}, options_str);
            kernel_type functor(kernel);
#ifndef FPGA
            cl::EnqueueArgs eargs(_async ? controls.async_queue : controls.queue,
                                  cl::NDRange(utils::calculate_global_size(_block_size, _needed_work_size)),
                                  cl::NDRange(_block_size));
#else
            cl::EnqueueArgs eargs(_is_spec_queue ? *_queue : controls.queue,
                                  cl::NDRange(utils::calculate_global_size(_block_size, _needed_work_size)),
                                  cl::NDRange(_block_size));
#endif

            return functor(eargs, args...);
        } catch (const cl::Error &e) {
            utils::program_handler(e, cl_program, controls.device, _kernel_name);
        }
    }


};
