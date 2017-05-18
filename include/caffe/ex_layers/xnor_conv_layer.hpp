#ifndef CAFFE_XNOR_CONVOLUTION_LAYER_HPP_
#define CAFFE_XNOR_CONVOLUTION_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "operator/xnet_common.h"
#include "operator/xnet_function.h"
using namespace xnet;
namespace caffe {
/*
 *@brief the test layer for XNOR Convolution
 */
template<typename Dtype>
class XNORConvolutionLayer: public Layer<Dtype>{
  public:
    explicit XNORConvolutionLayer(const LayerParameter& param):
      Layer<Dtype>(param){}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline int MinBottomBlobs() const {return 1;}
    virtual inline int MinTopBlobs() const {return 1;}
    virtual inline bool EqualNumberBottomTopBlobs() const {return 1;}

    virtual inline const char* type() const {return "XNORConvolution";}

  protected:
    BinBlob<Dtype> binary_weights_;
    BinBlob<Dtype> binary_inputs_;
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;};
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {Forward_cpu(bottom, top);};
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;};

    /// @brief The spatial dimensions of the input.
    inline int input_shape(int i) {
      return (*bottom_shape_)[channel_axis_ + i];
    }
    
    virtual void compute_output_shape();
    void forward_cpu_bias(Dtype* output, const Dtype* bias); 
    // reverse_dimensions should return true iff we are implementing deconv, so
    // that conv helpers know which dimensions are which.
    // Compute height_out_ and width_out_ from other parameters.

    /// @brief The spatial dimensions of a filter kernel.
    Blob<int> kernel_shape_;
    /// @brief The spatial dimensions of the stride.
    Blob<int> stride_;
    /// @brief The spatial dimensions of the padding.
    Blob<int> pad_;
    /// @brief The spatial dimensions of the dilation.
    Blob<int> dilation_;
    /// @brief The spatial dimensions of the convolution input.
    Blob<int> conv_input_shape_;
    /// @brief The spatial dimensions of the col_buffer.
    //vector<int> col_buffer_shape_;
    /// @brief The spatial dimensions of the output.
    vector<int> output_shape_;
    const vector<int>* bottom_shape_;

    int num_spatial_axes_;
    int bottom_dim_;
    int top_dim_;

    int channel_axis_;
    int num_;
    int channels_;
    int group_;
    int out_spatial_dim_;
    int weight_offset_;
    int num_output_;
    bool bias_term_;
    bool is_1x1_;
    bool force_nd_im2col_;

    int conv_out_channels_;
    int conv_in_channels_;
    int conv_out_spatial_dim_;
    int kernel_dim_;
    Blob<Dtype> bias_multiplier_;
};

}
#endif 
