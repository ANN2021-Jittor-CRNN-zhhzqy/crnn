#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using std::cout;
using std::endl;

int main(int argc, const char *argv[]) {
  std::cout << "Hello!" << std::endl;
  auto image = cv::imread(argv[2]);

  //缩放至指定大小

  cv::resize(image, image, cv::Size(100, 32));

  std::vector<cv::Mat> channels;
  split(image, channels);

  //转成张量
  auto input_tensor =
      torch::from_blob(channels[0].data, {image.rows, image.cols, 1},
                       torch::kByte)
          .permute({2, 0, 1})
          .unsqueeze(0)
          .to(torch::kFloat32) /
      225.0;
  auto model = torch::jit::load(argv[1]);

  model.eval();

  //前向传播
  auto output = model.forward({input_tensor}).toTensor();
  // cout<<output<<endl;

  auto max_result = output.max(2);
  auto max_index = std::get<1>(max_result).transpose(0, 1);

  std::string a = "";
  void *ptr = max_index.data_ptr();
  for (int i = 0; i < max_index.size(1); ++i) {
    int tmp = *((int *)(ptr + i));
    if (tmp != 0) {
      if (tmp < 10) {
        a += (char)('0' + tmp - 1);
      } else {
        a += (char)('a' + tmp - 11);
      }
    }
  }

  cout <<"result:"<< a << endl;

  return 0;
}