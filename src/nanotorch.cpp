#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

static int add_ints(int a, int b){
  return a + b;
}

static std::string hello() {
  return "nanoTorch: import OK";
}

PYBIND11_MODULE(nanotorch, m){
  m.doc() = "nanoTorch - minimal PyTorch";

  m.def("add_ints", &add_ints, "Add two integers",py::arg("a"),py::arg("b"));
  m.def("hello", &hello, "Hello, World!");
  m.attr("__version__") = "0.0.1";
}
