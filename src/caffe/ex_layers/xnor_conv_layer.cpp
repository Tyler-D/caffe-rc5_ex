#include "caffe/ex_layers/xnor_conv_layer.hpp"
#include "caffe/filler.hpp"
#include <iostream>
#include <fstream>
using std::ofstream;

namespace caffe{

template<typename Dtype>
void XNORConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    //specify the bottom shape:
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    force_nd_im2col_ = conv_param.force_nd_im2col();
    channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom[0]->num_axes();
    num_spatial_axes_ = num_axes - first_spatial_axis;
    CHECK_GE(num_spatial_axes_, 0);
    vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
    // Setup filter kernel dimensions (kernel_shape_).
    kernel_shape_.Reshape(spatial_dim_blob_shape);
    int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "kernel_h & kernel_w can only be used for 2D convolution.";
      CHECK_EQ(0, conv_param.kernel_size_size())
          << "Either kernel_size or kernel_h/w should be specified; not both.";
      kernel_shape_data[0] = conv_param.kernel_h();
      kernel_shape_data[1] = conv_param.kernel_w();
    } else {
      const int num_kernel_dims = conv_param.kernel_size_size();
      CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
          << "kernel_size must be specified once, or once per spatial dimension "
          << "(kernel_size specified " << num_kernel_dims << " times; "
          << num_spatial_axes_ << " spatial dims).";
        for (int i = 0; i < num_spatial_axes_; ++i) {
          kernel_shape_data[i] =
              conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
    }
    for (int i = 0; i < num_spatial_axes_; ++i) {
      CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
    }
    // Setup stride dimensions (stride_).
    stride_.Reshape(spatial_dim_blob_shape);
    int* stride_data = stride_.mutable_cpu_data();
    if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "stride_h & stride_w can only be used for 2D convolution.";
      CHECK_EQ(0, conv_param.stride_size())
          << "Either stride or stride_h/w should be specified; not both.";
      stride_data[0] = conv_param.stride_h();
      stride_data[1] = conv_param.stride_w();
    } else {
      const int num_stride_dims = conv_param.stride_size();
      CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
            num_stride_dims == num_spatial_axes_)
          << "stride must be specified once, or once per spatial dimension "
          << "(stride specified " << num_stride_dims << " times; "
          << num_spatial_axes_ << " spatial dims).";
      const int kDefaultStride = 1;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
            conv_param.stride((num_stride_dims == 1) ? 0 : i);
        CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
      }
    }
    // Setup pad dimensions (pad_).
    pad_.Reshape(spatial_dim_blob_shape);
    int* pad_data = pad_.mutable_cpu_data();
    if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "pad_h & pad_w can only be used for 2D convolution.";
      CHECK_EQ(0, conv_param.pad_size())
          << "Either pad or pad_h/w should be specified; not both.";
      pad_data[0] = conv_param.pad_h();
      pad_data[1] = conv_param.pad_w();
    } else {
      const int num_pad_dims = conv_param.pad_size();
      CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
            num_pad_dims == num_spatial_axes_)
          << "pad must be specified once, or once per spatial dimension "
          << "(pad specified " << num_pad_dims << " times; "
          << num_spatial_axes_ << " spatial dims).";
      const int kDefaultPad = 0;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
            conv_param.pad((num_pad_dims == 1) ? 0 : i);
      }
    }
    // Setup dilation dimensions (dilation_).
    dilation_.Reshape(spatial_dim_blob_shape);
    int* dilation_data = dilation_.mutable_cpu_data();
    const int num_dilation_dims = conv_param.dilation_size();
    CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
          num_dilation_dims == num_spatial_axes_)
        << "dilation must be specified once, or once per spatial dimension "
        << "(dilation specified " << num_dilation_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultDilation = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                        conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
    }
    // Special case: im2col is the identity for 1x1 convolution with stride 1
    // and no padding, so flag for skipping the buffer and transformation.
    is_1x1_ = true;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      is_1x1_ &=
          kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
      if (!is_1x1_) { break; }
    }
    // Configure output channels and groups.
    channels_ = bottom[0]->shape(channel_axis_);
    num_output_ = this->layer_param_.convolution_param().num_output();
    CHECK_GT(num_output_, 0);
    group_ = this->layer_param_.convolution_param().group();
    CHECK_EQ(channels_ % group_, 0);
    CHECK_EQ(num_output_ % group_, 0)
        << "Number of output should be multiples of group.";

    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
    // Handle the parameters: weights and biases.
    // - blobs_[0] holds the filter weights
    // - blobs_[1] holds the biases (optional)
    vector<int> weight_shape(2);
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      weight_shape.push_back(kernel_shape_data[i]);
    }
    bias_term_ = this->layer_param_.convolution_param().bias_term();
    vector<int> bias_shape(bias_term_, num_output_);
    if (this->blobs_.size() > 0) {
      CHECK_EQ(1 + bias_term_, this->blobs_.size())
          << "Incorrect number of weight blobs.";
      if (weight_shape != this->blobs_[0]->shape()) {
        Blob<Dtype> weight_shaped_blob(weight_shape);
        LOG(FATAL) << "Incorrect weight shape: expected shape "
            << weight_shaped_blob.shape_string() << "; instead, shape was "
            << this->blobs_[0]->shape_string();
      }
      if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
        Blob<Dtype> bias_shaped_blob(bias_shape);
        LOG(FATAL) << "Incorrect bias shape: expected shape "
            << bias_shaped_blob.shape_string() << "; instead, shape was "
            << this->blobs_[1]->shape_string();
      }
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      if (bias_term_) {
        this->blobs_.resize(2);
      } else {
        this->blobs_.resize(1);
      }
      // Initialize and fill the weights:
      // output channels x input channels per-group x kernel height x kernel width
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
      // If necessary, initialize and fill the biases.
      if (bias_term_) {
        this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
      }
    }
    kernel_dim_ = this->blobs_[0]->count(1);
    //weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    //Weights binarization
    binarized_ = false;
    binary_weights_.Reshape(conv_out_channels_ , conv_in_channels_ , 
        kernel_shape_data[0], kernel_shape_data[1]);
}

template<typename Dtype>
void XNORConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);

  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  //for XNOR binarizeing weight and only support 2-D convolution
  //binary_inputs_.Reshape((*bottom_shape_)[0], (*bottom_shape_)[1], 
                         //(*bottom_shape_)[2], (*bottom_shape_)[3]);  
  //binary_inputs_.copyRealValueFrom(bottom[0]->cpu_data());
}

template<typename Dtype>
void XNORConvolutionLayer<Dtype>::compute_output_shape(){
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template<typename Dtype>
void XNORConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  //xnor convolution
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();  
  caffe_memset(top[0]->count()*sizeof(Dtype), 0, top_data);

  if (!binarized_){
    alphas_.resize(num_output_);
#ifdef XNOR_OMP
    binarizeWeights_omp(this->blobs_[0]->cpu_data(), binary_weights_, alphas_); 
#else
    binarizeWeights(this->blobs_[0]->cpu_data(), binary_weights_, alphas_);
#endif
    binarized_ = true;
  }

  for(int n = 0; n < this->num_; n++){
    const Dtype* input_data = bottom[0]->cpu_data()+
                              n * this->bottom_dim_;

  //binarized inputs
#ifdef XNOR_OMP
    binarizeIm2Col(input_data, binary_inputs_, conv_in_channels_, 
        (*bottom_shape_)[2], (*bottom_shape_)[3], kernel_shape_data[0],
        kernel_shape_data[1], pad_data[0], pad_data[1], stride_data[0], 
        stride_data[1], dilation_data[0], dilation_data[1]);
#else
    binarizeIm2Col(input_data, binary_inputs_, conv_in_channels_, 
        (*bottom_shape_)[2], (*bottom_shape_)[3], kernel_shape_data[0],
        kernel_shape_data[1], pad_data[0], pad_data[1], stride_data[0], 
        stride_data[1], dilation_data[0], dilation_data[1]);
#endif

  //popcnt_gemm
#ifdef XNOR_OMP
    //LOG(INFO)<<"Use OpenMP GEMM";
    xnorGEMM_omp_unrolled(conv_out_channels_, ceil((float)kernel_dim_/BIN_SIZE), out_spatial_dim_, 
                      binary_weights_.b_data(), ceil((float)kernel_dim_/BIN_SIZE),
                      binary_inputs_.b_data(), out_spatial_dim_,
                      top_data+n*this->top_dim_, out_spatial_dim_,
                      kernel_dim_, alphas_);
#else
    //LOG(INFO)<<"Use No OpenMP";
    xnorGEMM_baseline(conv_out_channels_, ceil((float)kernel_dim_/BIN_SIZE), out_spatial_dim_, 
                      binary_weights_.b_data(), ceil((float)kernel_dim_/BIN_SIZE),
                      binary_inputs_.b_data(), out_spatial_dim_,
                      top_data+n*this->top_dim_, out_spatial_dim_,
                      kernel_dim_, alphas_);
#endif 
  
  //bias:
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void XNORConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

INSTANTIATE_CLASS(XNORConvolutionLayer);
REGISTER_LAYER_CLASS(XNORConvolution);

}
