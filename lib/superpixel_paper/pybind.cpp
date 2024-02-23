#include <torch/extension.h>

// -- search --
void init_eff_normz(py::module &);
void init_sna_agg(py::module &);
void init_sna_attn(py::module &m);
void init_ssna_attn(py::module &m);
void init_ssna_agg(py::module &m);
void init_ssna_reweight(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_eff_normz(m);
  init_sna_agg(m);
  init_sna_attn(m);
  init_ssna_attn(m);
  init_ssna_agg(m);
  init_ssna_reweight(m);
}
