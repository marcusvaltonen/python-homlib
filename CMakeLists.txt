cmake_minimum_required(VERSION 3.4)
project(python-homlib)

find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

SET(HOMLIB_DIR "lib/HomLib")

SET(SOURCE_DIR "src")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
include_directories(${HOMLIB_DIR})
include_directories(${HOMLIB_DIR}/includes/HomLib)
include_directories(${HOMLIB_DIR}/src/helpers)
include_directories(${HOMLIB_DIR}/src/solvers/fitzgibbon_cvpr_2001)
include_directories(${HOMLIB_DIR}/src/solvers/kukelova_cvpr_2015)
include_directories(${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020a)
include_directories(${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b)
SET(SRCS_HOMLIB
    ${HOMLIB_DIR}/src/helpers/gj.cpp
    ${HOMLIB_DIR}/src/helpers/normalize2dpts.cpp
    ${HOMLIB_DIR}/src/helpers/radial.cpp
    ${HOMLIB_DIR}/src/helpers/roots.cpp
    ${HOMLIB_DIR}/src/solvers/fitzgibbon_cvpr_2001/get_fitzgibbon_cvpr_2001.cpp
    ${HOMLIB_DIR}/src/solvers/kukelova_cvpr_2015/get_kukelova_cvpr_2015.cpp
    ${HOMLIB_DIR}/src/solvers/kukelova_cvpr_2015/solver_kukelova_cvpr_2015.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020a/get_valtonenornhag_arxiv_2020a_fHf.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020a/solver_valtonenornhag_arxiv_2020a_fHf.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/get_valtonenornhag_arxiv_2020b_fHf.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/get_valtonenornhag_arxiv_2020b_frHfr.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/solver_valtonenornhag_arxiv_2020b_fHf.cpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/solver_valtonenornhag_arxiv_2020b_frHfr.cpp
)
SET(HDRS_HOMLIB
    ${HOMLIB_DIR}/includes/HomLib/get_fitzgibbon_cvpr_2001.hpp
    ${HOMLIB_DIR}/includes/HomLib/get_kukelova_cvpr_2015.hpp
    ${HOMLIB_DIR}/includes/HomLib/get_valtonenornhag_arxiv_2020a.hpp
    ${HOMLIB_DIR}/includes/HomLib/get_valtonenornhag_arxiv_2020b.hpp
    ${HOMLIB_DIR}/src/helpers/gj.hpp
    ${HOMLIB_DIR}/src/helpers/normalize2dpts.hpp
    ${HOMLIB_DIR}/src/helpers/radial.hpp
    ${HOMLIB_DIR}/src/helpers/roots.hpp
    ${HOMLIB_DIR}/src/solvers/kukelova_cvpr_2015/solver_kukelova_cvpr_2015.hpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020a/solver_valtonenornhag_arxiv_2020a_fHf.hpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/solver_valtonenornhag_arxiv_2020b_fHf.hpp
    ${HOMLIB_DIR}/src/solvers/valtonenornhag_arxiv_2020b/solver_valtonenornhag_arxiv_2020b_frHfr.hpp
)

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(homlib "${SOURCE_DIR}/bindings.cpp" ${HDRS_HOMLIB} ${SRCS_HOMLIB})
target_link_libraries(homlib PRIVATE Eigen3::Eigen)

add_library(${PROJECT_NAME} STATIC ${HDRS_HOMLIB} ${SRCS_HOMLIB})

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
