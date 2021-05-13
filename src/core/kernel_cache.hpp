#pragma once
#include "headers_map.hpp"

namespace clbool::details {
    using program_id = std::string; // program name|options
    using kernel_id = std::pair<program_id, std::string>; // program name,  kernel name

    struct KernelCache {
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
                                              std::string options = "") {
            cl::Program cl_program;
            try {
                options += " -D RUN ";
                std::string program_key = _program_name + "|" + options;
                if (KernelCache::programs.find(program_key) != KernelCache::programs.end()) {
                    return programs[_program_name];
                }

                auto source_ptr = HeadersMap.find(_program_name);
                if (source_ptr == HeadersMap.end()) {
                    throw std::runtime_error("Cannot fine program " + _program_name + "! 326784368165");
                }

                SET_TIMER
                {
                    START_TIMING
                    KernelSource source = source_ptr->second;
                    cl_program = controls.create_program_from_source(source.kernel, source.length);
                    cl_program.build(options.c_str());
                    KernelCache::programs[_program_name] = cl_program;
                    END_TIMING(" program " + _program_name + " build in: ");
                }

                return KernelCache::programs[_program_name];
            } catch (const cl::Error &e) {
                utils::program_handler(e, cl_program, controls.device, _program_name);
            }
        }

        static const cl::Kernel &get_kernel(const Controls &controls, const kernel_id &id, const std::string& options) {
            cl::Program cl_program = get_program(controls, id.first, options);
            if (kernels.find(id) != kernels.end()) {
                return kernels[id];
            }
            kernels[id] = cl::Kernel(cl_program, id.second.c_str());
            return kernels[id];
        }

    };
}