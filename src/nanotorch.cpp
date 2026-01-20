#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static int add_ints(int a, int b){ return a + b; }
static std::string hello() { return "nanoTorch: import OK"; }

static int64_t numel_from_shape(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape){
    if (d <= 0) throw std::runtime_error("Tensor shape dims must be > 0");
    n *= d;
  }
  return n;
}

class Tensor {
public:
  Tensor() = default;

  Tensor(std::vector<int64_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
      const int64_t expected = numel_from_shape(shape_);
      if (static_cast<int64_t>(data_.size()) != expected){
        throw std::runtime_error("Tensor data size does not match shape numel");
      }
    }

  const std::vector<int64_t>& shape() const { return shape_;}
  int64_t numel() const { return numel_from_shape(shape_);}

  // converting Tensor -> ndarray (numpy) and returns it to Python
  py::array_t<float> numpy() const {
    std::vector<ssize_t> np_shape;
    np_shape.reserve(shape_.size());
    for (ssize_t d : shape_) np_shape.push_back(static_cast<ssize_t>(d));

    py::array_t<float> arr(np_shape);
    auto buf = arr.mutable_unchecked(); // fast write without bound checking
    for (ssize_t i = 0; i < static_cast<ssize_t>(data_.size()); i++) {
      buf.data(i) = data_[static_cast<size_t>(i)];
    }
    return arr;
  }

  std::string repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); i++){
      oss << shape_[i];
      if (i + 1 < shape_.size()) oss << ", ";
    }
    oss << "], dtype=float32";
    return oss.str();
  }

  static Tensor from_numpy(const py::array& arr_any) {
    py::array_t<float, py::array::c_style | py::array::forcecast> arr(arr_any);

    py::buffer_info info = arr.request();
    if (info.ndim < 1) throw std::runtime_error("Tensor must have at least 1 dimension");

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(info.ndim));
    for (int i = 0; i < info.ndim; i++){
      shape.push_back(static_cast<int64_t>(info.shape[static_cast<size_t>(i)]));
    }

    cosnt int64_t n = numel_from_shape(shape);
    const float* ptr = static_cast<const float*>(info.ptr);

    std::vector<float> data;
    data.assign(ptr, ptr+n);

    return Tensor(std::move(shape), std::move(data));
  }
private:
  std::vector<int64_t> shape_;
  std::vector<float> data_;
};


PYBIND11_MODULE(nanotorch, m){
  m.doc() = "nanoTorch - minimal PyTorch";

  m.def("add_ints", &add_ints, "Add two integers",py::arg("a"),py::arg("b"));
  m.def("hello", &hello, "Hello, World!");
  m.attr("__version__") = "0.0.1";

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<>())
    // used to construct from numpy array -> Tensor(np_array)
    .def(py::init([](py::array arr) { return Tensor::from_numpy(arr); }),
        py::arg("array"))

    // static constructor for clarity
    .def_static("from_numpy", &Tensor::from_numpy, py::arg("array)"))
    .def_property_readonly("shape", &Tensor::shape)
    .def("numel", &Tensor::numel)
    .def("numpy", &Tensor::numpy)
    .def("__repr__", &Tensor::repr);
}
