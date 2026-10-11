// Safe 3D alpha persistence using GUDHI/CGAL; no radius or interval truncation.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <CGAL/Epeck_d.h>
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace py = pybind11;
using Kernel = CGAL::Epeck_d<CGAL::Dimension_tag<3>>;
using Alpha = Gudhi::alpha_complex::Alpha_complex<Kernel>;
struct Options : Gudhi::Simplex_tree_options_fast_persistence { using Filtration_value = double; };
using Tree = Gudhi::Simplex_tree<Options>;
using Persistence = Gudhi::persistent_cohomology::Persistent_cohomology<Tree, Gudhi::persistent_cohomology::Field_Zp>;
using Diagrams = std::array<std::vector<std::array<double, 2>>, 3>;

Diagrams compute(const double* coordinates, int n) {
    std::vector<Kernel::Point_d> points;
    points.reserve(n);
    for (int i = 0; i < n; ++i) {
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(coordinates[3*i+axis])) throw std::invalid_argument("Nonfinite alpha input");
        for (int j = 0; j < i; ++j)
            if (coordinates[3*i] == coordinates[3*j] && coordinates[3*i+1] == coordinates[3*j+1]
                && coordinates[3*i+2] == coordinates[3*j+2]) throw std::invalid_argument("Duplicate alpha input");
        points.emplace_back(coordinates+3*i, coordinates+3*i+3);
    }
    Alpha alpha(points);
    Tree tree;
    if (!alpha.create_complex(tree)) throw std::runtime_error("3D alpha-complex construction failed");
    if (tree.dimension() != 3) throw std::runtime_error("3D alpha backend requires a full-dimensional patch");
    Persistence persistence(tree, false);
    persistence.init_coefficients(2);
    persistence.compute_persistent_cohomology(0.);
    Diagrams diagrams;
    for (const auto& pair : persistence.get_persistent_pairs()) {
        const auto birth = std::get<0>(pair), death = std::get<1>(pair);
        if (death == tree.null_simplex()) continue;
        const int dimension = tree.dimension(birth);
        if (dimension <= 2)
            diagrams[dimension].push_back({tree.filtration(birth), tree.filtration(death)});
    }
    return diagrams;
}

void summarize(const std::vector<std::array<double, 2>>& pairs, int dim, double norm,
               const unsigned char* selected, double* out) {
    const int spectrum = dim == 0 ? 12 : 36;
    std::fill(out, out + 7 + spectrum + 8, 0.);
    std::vector<int> bins, curves;
    std::array<bool,6> use_birth{}, use_life{};
    for (int j=0; j<spectrum; ++j) if (selected[7+j]) {
        bins.push_back(j);
        if (dim>0) { use_birth[j/6]=true; use_life[j%6]=true; }
    }
    for (int j=0; j<8; ++j) if (selected[7+spectrum+j]) curves.push_back(j);
    const bool log_life=selected[1] || selected[2] || selected[3];
    std::vector<std::array<double, 3>> intervals;
    intervals.reserve(pairs.size());
    double total = 0.;
    for (const auto& pair : pairs) {
        const double birth = std::sqrt(std::max(0., pair[0]));
        const double death = std::sqrt(std::max(0., pair[1]));
        intervals.push_back({birth, death, death-birth});
        if (selected[4]) total += death-birth;
    }
    total = std::max(total, 1e-14);
    if (selected[0]) out[0] = pairs.size()/norm;
    for (const auto& interval : intervals) {
        const double birth = interval[0], death = interval[1], life = interval[2];
        const double loglife = log_life ? std::log1p(life) : 0.;
        if (selected[1]) out[1] += loglife/norm;
        if (selected[2]) out[2] += loglife*loglife/norm;
        if (selected[3]) out[3] = std::max(out[3], loglife);
        if (selected[4]) {
            const double p=life/total;
            if (p > 0.) out[4] -= p*std::log(p);
        }
        if (selected[5]) out[5] += std::log1p(birth)/norm;
        if (selected[6]) out[6] += std::log1p(death)/norm;
        if (dim == 0) {
            for (int j : bins) {
                const double z = (death - (.6 + 2.4*j/11))/.15;
                out[7+j] += std::exp(-.5*z*z)/norm;
            }
        } else if (!bins.empty()) {
            double b[6], l[6];
            for (int j = 0; j < 6; ++j) {
                const double zb = (birth - (.6 + 3.4*j/5))/.25;
                const double zl = (life - 1.5*j/5)/.15;
                if (use_birth[j]) b[j] = std::exp(-.5*zb*zb);
                if (use_life[j]) l[j] = std::exp(-.5*zl*zl);
            }
            const double weight = std::tanh(life)/norm;
            for (int j : bins) out[7+j] += b[j/6]*l[j%6]*weight;
        }
        for (int j : curves) {
            const double alpha = .75 + .5*j;
            out[7+spectrum+j] += .25*(1+std::tanh((alpha-birth)/.1))*(1+std::tanh((death-alpha)/.1))/norm;
        }
    }
}

void packet(const double* coordinates, int count, const unsigned char* selected, double* out) {
    for (int level = 0; level < 2; ++level) {
        if (std::none_of(selected+129*level,selected+129*(level+1),[](auto x){return x!=0;})) continue;
        const int n = std::min(count, level == 0 ? 32 : 80);
        const auto result = compute(coordinates, n);
        int offset = 129*level;
        for (int dim = 0; dim < 3; ++dim) {
            const int length=dim == 0 ? 27 : 51;
            if (std::any_of(selected+offset,selected+offset+length,[](auto x){return x!=0;}))
                summarize(result[dim], dim, n-1, selected+offset, out+offset);
            offset += length;
        }
    }
}

py::array_t<double> packets(py::array_t<double, py::array::c_style> coordinates,
                           py::array_t<int, py::array::c_style> counts, int threads, py::object selection) {
    if (coordinates.ndim() != 3 || coordinates.shape(1) != 80 || coordinates.shape(2) != 3
        || counts.ndim() != 1 || counts.shape(0) != coordinates.shape(0) || threads < 1)
        throw std::invalid_argument("Expected float64 (B,80,3), int32 (B,) counts, and positive threads");
    const int size = counts.shape(0);
    for (int i = 0; i < size; ++i)
        if (counts.data()[i] < 14 || counts.data()[i] > 80) throw std::invalid_argument("Require 14..80 consumed points");
    std::vector<int> indices;
    if (selection.is_none()) for (int j=0; j<258; ++j) indices.push_back(j);
    else indices=selection.cast<std::vector<int>>();
    if (indices.empty()) throw std::invalid_argument("Require at least one TDA output");
    std::array<unsigned char,258> selected{};
    int previous=-1;
    for (int j : indices) {
        if (j<=previous || j>=258) throw std::invalid_argument("TDA indices must be sorted, unique and in [0,258)");
        selected[j]=1; previous=j;
    }
    py::array_t<double> output({py::ssize_t(size), py::ssize_t(indices.size())});
    const auto* x = coordinates.data(); const auto* n = counts.data(); auto* dest = output.mutable_data();
    std::vector<std::string> errors(size);
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(dynamic, 8) if(threads > 1)
        for (int i = 0; i < size; ++i) {
            try {
                std::array<double,258> values{};
                packet(x+240*i, n[i], selected.data(), values.data());
                for (size_t j=0; j<indices.size(); ++j) dest[indices.size()*i+j]=values[indices[j]];
            }
            catch (const std::exception& error) { errors[i] = error.what(); }
        }
    }
    for (int i = 0; i < size; ++i)
        if (!errors[i].empty()) throw std::runtime_error("Alpha batch row " + std::to_string(i) + ": " + errors[i]);
    return output;
}

py::list diagrams(py::array_t<double, py::array::c_style> coordinates) {
    if (coordinates.ndim() != 2 || coordinates.shape(1) != 3 || coordinates.shape(0) < 4)
        throw std::invalid_argument("Expected a contiguous float64 (N,3) patch, N>=4");
    Diagrams result;
    {
        py::gil_scoped_release release;
        result = compute(coordinates.data(), coordinates.shape(0));
    }
    py::list output;
    for (const auto& pairs : result) {
        py::array_t<double> values({static_cast<py::ssize_t>(pairs.size()), py::ssize_t(2)});
        auto* dest = values.mutable_data();
        for (const auto& pair : pairs) { *dest++ = pair[0]; *dest++ = pair[1]; }
        output.append(values);
    }
    return output;
}

PYBIND11_MODULE(pcm_alpha3d, module) {
    module.def("diagrams", &diagrams);
    module.def("packets", &packets, py::arg("coordinates"), py::arg("counts"), py::arg("threads")=1,
               py::arg("selection")=py::none());
}
