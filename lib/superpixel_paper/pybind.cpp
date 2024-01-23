#include <torch/extension.h>

// -- search --
void init_eff_normz(py::module &);
void init_nsa_agg(py::module &);
void init_nsa_attn(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_eff_normz(m);
  init_nsa_agg(m);
  init_nsa_attn(m);
}
