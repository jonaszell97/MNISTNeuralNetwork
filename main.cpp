
#include "neural/Network.h"

using namespace neural;

static std::pair<Mat, Vec> loadMNISTDataset(std::ifstream &data, std::ifstream &labels)
{
   std::string labelsStr((std::istreambuf_iterator<char>(labels)),
                        std::istreambuf_iterator<char>());

   /*
   [offset] [type]          [value]          [description]
   0000     32 bit integer  0x00000801(2049) magic number (MSB first)
   0004     32 bit integer  60000            number of items
   0008     unsigned byte   ??               label
   0009     unsigned byte   ??               label
   ........
   xxxx     unsigned byte   ??               label
   The labels values are 0 to 9.
    */
   Vec labelData(labelsStr.size() - 8);
   for (int i = 0; i < labelData.size(); ++i) {
      labelData[i] = (float)(unsigned char)labelsStr[i + 8];
   }

   std::string dataStr((std::istreambuf_iterator<char>(data)),
                         std::istreambuf_iterator<char>());

   /*
   [offset] [type]          [value]          [description]
   0000     32 bit integer  0x00000803(2051) magic number
   0004     32 bit integer  60000            number of images
   0008     32 bit integer  28               number of rows
   0012     32 bit integer  28               number of columns
   0016     unsigned byte   ??               pixel
   0017     unsigned byte   ??               pixel
   ........
   xxxx     unsigned byte   ??               pixel
   Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    */
   int pixelsPerImage = 28 * 28;
   int numImages = (dataStr.size() - 16) / pixelsPerImage;

   Mat imageData(numImages, pixelsPerImage);
   for (int n = 0; n < numImages; ++n) {
      for (int i = 0; i < 28; ++i) {
         for (int j = 0; j < 28; ++j) {
            imageData[n][i * 28 + j] = ((float)(unsigned char)dataStr[16 + n * pixelsPerImage + i * 28 + j]) / 255.f;
         }
      }
   }

   return { imageData, labelData };
}

static std::pair<std::pair<Mat, Vec>, std::pair<Mat, Vec>> loadMNIST(const std::string &path)
{
   std::ifstream trainingData(path + "/train-images.idx3-ubyte");
   std::ifstream trainingLabels(path + "/train-labels.idx1-ubyte");
   std::ifstream testData(path + "/t10k-images.idx3-ubyte");
   std::ifstream testLabels(path + "/t10k-labels.idx1-ubyte");

   return { loadMNISTDataset(trainingData, trainingLabels), loadMNISTDataset(testData, testLabels) };
}

int main(int argc, char **argv)
{
   if (argc < 2)
   {
      printf("MNIST dataset path missing!\n");
      return 1;
   }

   auto data = loadMNIST(argv[1]);
   Network network = Network::create(Network::CostFunction::Quadratic)
      .addLayer(28 * 28, Layer::ActivationFunction::Sigmoid)
      .addLayer(30, Layer::ActivationFunction::Sigmoid)
      .addLayer(10, Layer::ActivationFunction::Sigmoid)
      .build(Network::InitializationStrategy::Gaussian);

   network.saveParameters("bias_init.txt", "weight_init.txt");
   network.SGD(data.first.first, data.first.second,
               30,10, 3.f,
               data.second.first, data.second.second);

   network.saveParameters("bias.txt", "weight.txt");
}
