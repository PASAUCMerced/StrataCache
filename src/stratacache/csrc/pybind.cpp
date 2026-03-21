#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pcm.h"
#include <memory>

namespace py = pybind11;

class PCMCollectorHandle
{
private:
    std::unique_ptr<PCMCollector> collector;

public:
    PCMCollectorHandle()
    {
        collector = std::make_unique<PCMCollector>();
    }

    void tick()
    {
        if (collector)
            collector->tick();
    }

    py::array_t<double> getMemBandwidthBuffer()
    {
        auto sockets = collector->getNumSockets();
        auto channels = collector->getMaxIMCChannels();
        const auto &buf = collector->getMemBandwidthBuffer();

        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(sockets),
            static_cast<py::ssize_t>(channels),
            2};

        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(channels * 2 * sizeof(double)),
            static_cast<py::ssize_t>(2 * sizeof(double)),
            sizeof(double)};

        return py::array_t<double>(
            shape,
            strides,
            buf.data(),
            py::cast(this));
    }

    py::array_t<double> getPcieBandwidthBuffer()
    {
        auto sockets = collector->getNumSockets();
        const auto &buf = collector->getPcieBandwidthBuffer();

        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(sockets),
            2};

        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(2 * sizeof(double)),
            sizeof(double)};

        return py::array_t<double>(
            shape,
            strides,
            buf.data(),
            py::cast(this));
    }
};

PYBIND11_MODULE(pcm, m)
{
    py::class_<PCMCollectorHandle>(m, "PCMCollectorHandle")
        .def(py::init<>())
        .def("tick", &PCMCollectorHandle::tick, py::call_guard<py::gil_scoped_release>())
        .def("get_mem_bandwidth_buffer", &PCMCollectorHandle::getMemBandwidthBuffer)
        .def("get_pcie_bandwidth_buffer", &PCMCollectorHandle::getPcieBandwidthBuffer);
}