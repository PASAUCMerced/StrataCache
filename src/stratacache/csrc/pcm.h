#pragma once

#include "cpucounters.h"
class IPlatform;
#include <vector>
#include <chrono>
#include <memory>

class PCMCollector {
private:
    pcm::PCM *m;
    std::vector<pcm::ServerUncoreCounterState> _lastStates;
    std::unique_ptr<IPlatform> _pciePlatform;
    uint64_t _lastTick;
    uint32_t _numSockets;
    uint32_t _maxIMCChannels;
    bool _isValid;

    std::vector<double> _memBandwidthBuffer;
    std::vector<double> _pcieBandwidthBuffer;

public:
    explicit PCMCollector();
    ~PCMCollector();

    void tick();

    const std::vector<double>& getMemBandwidthBuffer() const { return _memBandwidthBuffer; }
    const std::vector<double>& getPcieBandwidthBuffer() const { return _pcieBandwidthBuffer; }
    uint32_t getNumSockets() const { return _numSockets; }
    uint32_t getMaxIMCChannels() const { return _maxIMCChannels; }
};