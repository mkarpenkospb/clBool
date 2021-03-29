#pragma once

#include "cl_includes.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
//#include <io.h>
#include <unistd.h>


struct Controls {
#ifdef WIN
    std::string WORKING_DIR = R"(C:\Users\mkarp\GitReps\clean_matrix\sparse_boolean_matrix_operations\)";
#else
    std::string WORKING_DIR = "/root/Desktop/GitReps/sparse_boolean_matrix_operations";
#endif
    std::string FPGA_BINARIES = "src/cl/fpga/multiplication_hash_kernels/";
    const cl::Device device;
    const cl::Context context;
    cl::CommandQueue queue;
    cl::CommandQueue async_queue;
    const uint32_t block_size = uint32_t(256);

    Controls(cl::Device& device) :
            device(device)
    , context(cl::Context(device))
    , queue(cl::CommandQueue(context))
#ifdef WIN
    , async_queue(cl::CommandQueue(context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
#else
    , async_queue(queue)
#endif
    {
        chdir(WORKING_DIR.c_str());
    }

    cl::Program create_program_from_source(const char * kernel, uint32_t length) const {
        return cl::Program(context, {{kernel, length}});
    }

    cl::Program create_program_from_binaries(std::string program_name) const {
#ifdef WIN
        program_name += ".cl";
        std::ifstream input(FPGA_BINARIES +  program_name, std::ios::binary);
        std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
        return cl::Program(context, {{buffer.data(), buffer.size()}});

#else
        timer localt;
        program_name += ".aocx";
        std::cout << "prepare_positions.aocx" << std::endl;
        std::string file = FPGA_BINARIES +  "prepare_positions.aocx";


        localt.start();

        //https://stackoverflow.com/a/6755132
        std::ifstream is(file);
        is.seekg(0, std::ios_base::end);
        std::size_t size=is.tellg();
        is.seekg(0, std::ios_base::beg);
        std::vector<unsigned char> buffer(size/sizeof(unsigned char));
        is.read((char*) &buffer[0], size);
        // Close the file
        is.close();


        double time = localt.elapsed();
        if (DEBUG_ENABLE) *logger << "create buffer with binaries in " << time << "\n";

        return cl::Program(context, {device}, {buffer});
#endif

    }

};


