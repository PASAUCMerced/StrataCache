#include "pcm.h"
#include "pcm-pcie.h"
#include <stdexcept>

using namespace pcm;

PCMCollector::PCMCollector() : _isValid(true)
{
    m = pcm::PCM::getInstance();
    if (!m->good())
    {
        throw std::runtime_error("PCM implementation not available (Check root privileges / MSR access).");
    }
    if (!m->memoryTrafficMetricsAvailable())
    {
        throw std::runtime_error("Memory traffic metrics not available on this platform.");
    }

    if (m->programServerUncoreMemoryMetrics(PartialWrites) != pcm::PCM::Success)
    {
        throw std::runtime_error("Failed to program Server Uncore Memory Metrics.");
    }

    _numSockets = m->getNumSockets();
    _maxIMCChannels = ServerUncoreCounterState::maxChannels;
    _lastStates.reserve(_numSockets);

    _memBandwidthBuffer.resize(_numSockets * _maxIMCChannels * 2, 0.0);
    _pcieBandwidthBuffer.resize(_numSockets  * 2, 0.0);

    try {
        static constexpr uint32_t kPmonMultiplier = 1000;
        _pciePlatform.reset(IPlatform::getPlatform(m, true, true, true, kPmonMultiplier));
    } catch (const std::exception& e) {
        std::cerr << "PCIeCollector: " << e.what() << " (PCIe metrics disabled)\n";
    }

    for (uint32_t i = 0; i < _numSockets; ++i)
    {
        _lastStates.push_back(m->getServerUncoreCounterState(i));
    }
    _lastTick = m->getTickCount();
}

void PCMCollector::tick()
{
    if (!m || !_isValid) return;

    auto current_tick = m->getTickCount();
    auto elapsed_ticks = current_tick - _lastTick;
    auto elapsed_sec = elapsed_ticks / 1000.0;

    std::vector<ServerUncoreCounterState> current_states(_numSockets);
    
    _pciePlatform->cleanup();
    _pciePlatform->getEvents();
    for (uint32_t i = 0; i < _numSockets; ++i)
    {
        current_states[i] = m->getServerUncoreCounterState(i);

        // Calculate memory bandwidth per channel
        for (uint32_t channel = 0; channel < _maxIMCChannels; ++channel)
        {
            uint64_t reads = getMCCounter(channel, ServerUncorePMUs::EventPosition::READ, _lastStates[i], current_states[i]);
            uint64_t writes = getMCCounter(channel, ServerUncorePMUs::EventPosition::WRITE, _lastStates[i], current_states[i]);

            double read_bw = (reads * 64.0) / elapsed_sec; // B/s (divide by 1000000 for MB/s)
            double write_bw = (writes * 64.0) / elapsed_sec; // B/s (divide by 1000000 for MB/s)

            size_t base_idx = (i * _maxIMCChannels * 2) + (channel * 2);
            _memBandwidthBuffer[base_idx] = read_bw;
            _memBandwidthBuffer[base_idx + 1] = write_bw;
        }

        {
            uint64_t read_bytes = _pciePlatform->getReadBw(i, IPlatform::TOTAL);
            uint64_t write_bytes = _pciePlatform->getWriteBw(i, IPlatform::TOTAL);

            size_t base_idx = (i * 2);
            _pcieBandwidthBuffer[base_idx] = read_bytes / elapsed_sec; // B/s
            _pcieBandwidthBuffer[base_idx + 1] = write_bytes / elapsed_sec; // B/s
        }
    }

    std::swap(_lastStates, current_states);
    _lastTick = current_tick;
}

PCMCollector::~PCMCollector()
{
    _isValid = false;
}