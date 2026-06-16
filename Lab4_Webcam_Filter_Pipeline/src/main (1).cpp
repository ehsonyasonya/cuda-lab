#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <vector>
#include <sstream>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/filter_utils.h"
#include "kernels/kernels.h"

int main(int argc, char **argv)
{
    plog::init(plog::info, new plog::ConsoleAppender<plog::TxtFormatter>());
    
    //text flags into object FilterOptions
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened()) {
        PLOG_ERROR << "Failed to open input source.";
        return -1;
    }

    cuda_filter::PipelineManager pm;

    //dynamic pipeline
    std::vector<cuda_filter::PipelineNode> pipeline;
    std::stringstream ss(options.pipelineString);
    std::string segment;

    //making text into the to do list, using PipelineNode
    while (std::getline(ss, segment, ',')) {
        if (segment == "blur")
            pipeline.push_back({cuda_filter::FilterType::BLUR, 1.0f, 0.0f});
        else if (segment == "hdr")
            pipeline.push_back({cuda_filter::FilterType::HDR_TONEMAPPING, 0.0f, 0.0f});
        else if (segment == "wipe")
            pipeline.push_back({cuda_filter::FilterType::WIPE_TRANSITION, 0.0f, 0.5f});
        else
            PLOG_WARNING << "Unknown filter type in pipeline: " << segment;
    }

    cv::Mat frame, previousFrame, currentFrame, outputFrame;
    
    while (true)
    {
        if (!inputHandler.readFrame(frame)) break;
        if (previousFrame.empty()) previousFrame = frame.clone();
        
        currentFrame = frame.clone();
        outputFrame = currentFrame.clone();

        //work with frames
        for (const auto& node : pipeline)
        {
            if (node.type == cuda_filter::FilterType::WIPE_TRANSITION)
            {
                cuda_filter::applyWipeTransitionGPU(pm, previousFrame, currentFrame, outputFrame, node.progress);
            }
            else if (node.type == cuda_filter::FilterType::HDR_TONEMAPPING)
            {
                cuda_filter::applyHDRToneMappingGPU(pm, currentFrame, outputFrame, options.exposure, options.gamma, options.saturation);
            }
            else
            {
                cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(node.type, options.kernelSize, node.intensity);
                cuda_filter::applyFilterGPU(pm, currentFrame, outputFrame, kernel);
            }
            outputFrame.copyTo(currentFrame);
        }

        cudaStreamSynchronize(pm.stream);

        frame.copyTo(previousFrame);
        cv::imwrite("/content/result.jpg", outputFrame);
        
        PLOG_INFO << "Frame processed and saved.";
        break; 
    }

    return 0;
}
