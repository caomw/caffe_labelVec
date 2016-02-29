#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LabelVecLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LabelVecLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LabelVecParameter label_vec_param = this->layer_param_.label_vec_param();
  n_ = label_vec_param.n();
  num_axes_ = bottom[0]->num_axes();
  int num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), n_, 1, 1);

}

template <typename Dtype>
void LabelVecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  int outer_num_ = bottom[0]->count(0);
  int inner_num_ = bottom[0]->count(1);
  for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < n_; j++) {
      if (bottom_data[i] == j)
        top_data[i * n_ + j] = sqrt((float)(n_) / (float)(j + 1)) - sqrt((float)(j + 1) / (float)(n_));
      else
        top_data[i * n_ + j] = - sqrt((float)(j + 1) / (float)(n_));
    }
  }
}

template <typename Dtype>
void LabelVecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(LabelVecLayer);
REGISTER_LAYER_CLASS(LabelVec);

}  // namespace caffe
