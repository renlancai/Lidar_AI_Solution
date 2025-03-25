#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>

class GPUVectorizer {
public:
    GPUVectorizer(int h, int w, int num_classes, 
                 float threshold=0.5f, int min_area=50)
        : height_(h), width_(w), num_classes_(num_classes),
          threshold_(threshold), min_area_(min_area) {
        // 预分配GPU内存
        d_binary_.create(height_, width_, CV_8UC1);
        d_temp_.create(height_, width_, CV_8UC1);
        
        // 创建流处理队列
        streams_.resize(num_classes_);
        for(auto& s : streams_) {
            s = cv::cuda::Stream();
        }
    }

    std::vector<MapElement> process(const cv::Mat& bev_tensor) {
        std::vector<MapElement> results;
        cv::Mat cpu_binary;

        // 多通道并行处理
        for(int c = 0; c < num_classes_; ++c) {
            // 提取当前类别通道
            cv::Mat channel(bev_tensor.size(), CV_32F, 
                           (void*)(bev_tensor.data + c*height_*width_*4));

            // 上传数据到GPU
            cv::cuda::GpuMat d_channel(height_, width_, CV_32F, 
                                      const_cast<float*>(channel.ptr<float>()));
            
            // 异步阈值处理
            cv::cuda::threshold(d_channel, d_binary_, threshold_, 1.0, 
                               cv::THRESH_BINARY, streams_[c]);
            
            // 形态学后处理（膨胀去噪）
            cv::cuda::erode(d_binary_, d_temp_, cv::Mat(), 
                           cv::Point(-1,-1), 1, streams_[c]);
            cv::cuda::dilate(d_temp_, d_binary_, cv::Mat(), 
                            cv::Point(-1,-1), 1, streams_[c]);

            // 下载二值图到CPU（异步）
            d_binary_.download(cpu_binary, streams_[c]);
            
            // 同步流并处理轮廓
            streams_[c].waitForCompletion();
            process_class_contours(cpu_binary, c, results);
        }
        return results;
    }

private:
    void process_class_contours(cv::Mat& binary, int class_id, 
                               std::vector<MapElement>& results) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, 
                        cv::CHAIN_APPROX_SIMPLE);

        for(auto& contour : contours) {
            if(cv::contourArea(contour) < min_area_) continue;

            MapElement elem;
            elem.class_id = class_id;
            cv::approxPolyDP(contour, elem.polygon, epsilon_, true);
            elem.bounding_box = cv::boundingRect(elem.polygon);
            
            results.emplace_back(std::move(elem));
        }
    }

    int height_, width_, num_classes_;
    float threshold_, epsilon_=1.5;
    int min_area_;
    std::vector<cv::cuda::Stream> streams_;
    cv::cuda::GpuMat d_binary_, d_temp_;
};

