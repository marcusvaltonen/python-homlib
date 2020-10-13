#include "get_fitzgibbon_cvpr_2001.hpp"
#include "get_kukelova_cvpr_2015.hpp"
#include "get_valtonenornhag_arxiv_2020a.hpp"
#include "get_valtonenornhag_arxiv_2020b.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::dict get_fitzgibbon_cvpr_2001_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                          const Eigen::Ref<const Eigen::MatrixXd> &x2)
{

    if (x1.rows() != 2 || x1.cols() != 5) {
        throw std::invalid_argument("First argument should be of size 2x5");
    }
    if (x2.rows() != 2 || x2.cols() != 5) {
        throw std::invalid_argument("Second argument should be of size 2x5");
    }

    HomLib::PoseData posedata = HomLib::FitzgibbonCVPR2001::get(x1, x2);

    py::dict d;
    d["H"] = posedata.homography;
    d["lam"] = posedata.distortion_parameter;

    return d;
}

py::list get_kukelova_cvpr_2015_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                        const Eigen::Ref<const Eigen::MatrixXd> &x2)
{

    if (x1.rows() != 2 || x1.cols() != 5) {
        throw std::invalid_argument("First argument should be of size 2x5");
    }
    if (x2.rows() != 2 || x2.cols() != 5) {
        throw std::invalid_argument("Second argument should be of size 2x5");
    }

    std::vector<HomLib::PoseData> posedata = HomLib::KukelovaCVPR2015::get(x1, x2);

    py::list lst;
    for (int i = 0; i < posedata.size(); i++) {
        py::dict d;
        d["H"] = posedata[i].homography;
        d["lam1"] = posedata[i].distortion_parameter;
        d["lam2"] = posedata[i].distortion_parameter2;
        lst.append(d);
    }

    return lst;
}

py::dict get_valtonenornhag_arxiv_2020a_fHf_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &x2,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &R1,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &R2)
{

    if (x1.rows() != 2 || x1.cols() != 3) {
        throw std::invalid_argument("First argument should be of size 2x3");
    }
    if (x2.rows() != 2 || x2.cols() != 3) {
        throw std::invalid_argument("Second argument should be of size 2x3");
    }
    if (R1.rows() != 3 || R1.cols() != 3) {
        throw std::invalid_argument("Third argument should be of size 3x3");
    }
    if (R2.rows() != 3 || R2.cols() != 3) {
        throw std::invalid_argument("Fourth argument should be of size 3x3");
    }

    HomLib::PoseData posedata = HomLib::ValtonenOrnhagArxiv2020A::get_fHf(x1, x2, R1, R2);

    py::dict d;
    d["H"] = posedata.homography;
    d["f"] = posedata.focal_length;

    return d;
}

py::list get_valtonenornhag_arxiv_2020b_fHf_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &x2,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &R1,
                                                    const Eigen::Ref<const Eigen::MatrixXd> &R2)
{

    if (x1.rows() != 2 || x1.cols() != 2) {
        throw std::invalid_argument("First argument should be of size 2x2");
    }
    if (x2.rows() != 2 || x2.cols() != 2) {
        throw std::invalid_argument("Second argument should be of size 2x2");
    }
    if (R1.rows() != 3 || R1.cols() != 3) {
        throw std::invalid_argument("Third argument should be of size 3x3");
    }
    if (R2.rows() != 3 || R2.cols() != 3) {
        throw std::invalid_argument("Fourth argument should be of size 3x3");
    }

    std::vector<HomLib::PoseData> posedata = HomLib::ValtonenOrnhagArxiv2020B::get_fHf(x1, x2, R1, R2);

    py::list lst;
    for (int i = 0; i < posedata.size(); i++) {
        py::dict d;
        d["H"] = posedata[i].homography;
        d["f"] = posedata[i].focal_length;
        lst.append(d);
    }

    return lst;
}

py::dict get_valtonenornhag_arxiv_2020b_frHfr_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                                      const Eigen::Ref<const Eigen::MatrixXd> &x2,
                                                      const Eigen::Ref<const Eigen::MatrixXd> &R1,
                                                      const Eigen::Ref<const Eigen::MatrixXd> &R2)
{

    if (x1.rows() != 2 || x1.cols() != 3) {
        throw std::invalid_argument("First argument should be of size 2x3");
    }
    if (x2.rows() != 2 || x2.cols() != 3) {
        throw std::invalid_argument("Second argument should be of size 2x3");
    }
    if (R1.rows() != 3 || R1.cols() != 3) {
        throw std::invalid_argument("Third argument should be of size 3x3");
    }
    if (R2.rows() != 3 || R2.cols() != 3) {
        throw std::invalid_argument("Fourth argument should be of size 3x3");
    }

    HomLib::PoseData posedata = HomLib::ValtonenOrnhagArxiv2020B::get_frHfr(x1, x2, R1, R2);

    py::dict d;
    d["H"] = posedata.homography;
    d["f"] = posedata.focal_length;
    d["lam"] = posedata.distortion_parameter;

    return d;
}

PYBIND11_MODULE(homlib, m) {
  m.doc() = R"doc(
        Python module
        -----------------------
        .. currentmodule:: homlib
        .. autosummary::
           :toctree: _generate

           get_fitzgibbon_cvpr_2001
           get_kukelova_cvpr_2015
           get_valtonenornhag_arxiv_2020a_fHf
           get_valtonenornhag_arxiv_2020b_fHf
           get_valtonenornhag_arxiv_2020b_frHfr
    )doc";

  m.def("get_fitzgibbon_cvpr_2001", &get_fitzgibbon_cvpr_2001_wrapper, R"doc(
        Fitzgibbon (CVPR, 2001) 5-point radial distortion homography.

        More info to be added...
    )doc");

  m.def("get_kukelova_cvpr_2015", &get_kukelova_cvpr_2015_wrapper, R"doc(
        Kukelova et al. (CVPR, 2015) 5-point radial distortion homography.

        More info to be added...
    )doc");

  m.def("get_valtonenornhag_arxiv_2020a_fHf", &get_valtonenornhag_arxiv_2020a_fHf_wrapper, R"doc(
        Valtonen Ornhag et al. (ArXiV, 2020) 2.5-point homography using IMU data.

        More info to be added...
    )doc");

  m.def("get_valtonenornhag_arxiv_2020b_fHf", &get_valtonenornhag_arxiv_2020b_fHf_wrapper, R"doc(
        Valtonen Ornhag et al. (ArXiV, 2020) 2-point homography using IMU data.

        More info to be added...
    )doc");

  m.def("get_valtonenornhag_arxiv_2020b_frHfr", &get_valtonenornhag_arxiv_2020b_frHfr_wrapper, R"doc(
        Valtonen Ornhag et al. (ArXiV, 2020) 2.5-point radial distortion homography using IMU data.

        More info to be added...
    )doc");
}
