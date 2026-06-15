#pragma once

#include <string>
#include <cxxopts.hpp>

namespace cuda_filter
{

    enum class InputSource
    {
        WEBCAM,
        IMAGE,
        VIDEO,
        SYNTHETIC
    };

    enum class SyntheticPattern
    {
        CHECKERBOARD,
        GRADIENT,
        NOISE
    };

    struct FilterOptions
    {
        InputSource inputSource;
        std::string inputPath;
        SyntheticPattern syntheticPattern;
        int deviceId;
        
        // Pipeline configuration
        std::string pipelineString; // Stores the comma-separated pipeline string (e.g., "blur,hdr,wipe")
        
        // Filter parameters
        std::string filterType; // Kept for legacy compatibility
        int kernelSize;
        float sigma;
        float intensity;
        bool preview;
        
        // HDR parameters
        float exposure;
        float gamma;
        float saturation;
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);

        FilterOptions parseArgs();

    private:
        int m_argc;
        char **m_argv;

        void setupOptions(cxxopts::Options &options);
        InputSource stringToInputSource(const std::string &str);
        SyntheticPattern stringToSyntheticPattern(const std::string &str);
    };

} // namespace cuda_filter