#include "clBool_tests.hpp"

std::shared_ptr<clbool::Controls> Wrapper::controls = nullptr;

void Wrapper::initControls() {
    if (controls == nullptr) {
        controls = std::make_shared<clbool::Controls>(clbool::create_controls(0, 0));
    }
}

