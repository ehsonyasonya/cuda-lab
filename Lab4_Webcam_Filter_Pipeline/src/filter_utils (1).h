#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    enum class FilterType
    {
        BLUR,
        SHARPEN,
        EDGE_DETECTION,
        EMBOSS,
        IDENTITY,
        HDR_TONEMAPPING,
        WIPE_TRANSITION
    };

    struct PipelineNode {
    FilterType type;
    float intensity = 1.0f;
    float progress = 0.0f;
    };

    class FilterUtils
    {
    public:
        static FilterType stringToFilterType(const std::string &filterName);
        static cv::Mat createFilterKernel(FilterType type, int kernelSize, float intensity = 1.0f);

        static void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    };

    struct HDRParams {
      float exposure;
      float gamma;
      float saturation;
    };

} // namespace cuda_filter