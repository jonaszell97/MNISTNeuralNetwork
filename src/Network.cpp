
#include "neural/Network.h"

#include <algorithm>
#include <random>

using namespace neural;

float neural::identity(float t)
{
   return t;
}

Vec neural::identity(const Vec &v)
{
   return v;
}

float neural::identityDerivative(float)
{
   return 1.f;
}

Vec neural::identityDerivative(const Vec &v)
{
   int size = v.size();

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = 1.f;

   return result;
}

float neural::sigmoid(float t)
{
   return 1.f / (1.f + exp(-t));
}

Vec neural::sigmoid(const Vec &v)
{
   int size = v.size();

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = sigmoid(v[i]);

   return result;
}

float neural::sigmoidDerivative(float t)
{
   float sig = sigmoid(t);
   return sig * (1.f - sig);
}

Vec neural::sigmoidDerivative(const Vec &v)
{
   int size = v.size();

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = sigmoidDerivative(v[i]);

   return result;
}

float neural::relu(float t)
{
   return std::max(0.f, t);
}

Vec neural::relu(const Vec &v)
{
   int size = v.size();

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = relu(v[i]);

   return result;
}

float neural::reluDerivative(float t)
{
   if (t <= 0.f)
      return 0.f;

   return 1.f;
}

Vec neural::reluDerivative(const Vec &v)
{
   int size = v.size();

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = reluDerivative(v[i]);

   return result;
}

Vec neural::quadraticCost(const Vec &a, const Vec &y)
{
   assert(a.size() == y.size());

   int size = a.size();
   Vec result(size);

   for (int i = 0; i < size; ++i) {
      float y_v = y[i];
      float a_v = a[i];

      result[i] = (y_v - a_v) * (y_v - a_v);
   }

   return result;
}

Vec neural::quadraticCostDerivative(const Vec &a, const Vec &y)
{
   assert(a.size() == y.size());
   return a - y;
}

float neural::combinedQuadraticCost(const Vec &loss)
{
   float sum = 0.f;
   for (int i = 0; i < loss.size(); ++i)
      sum += loss[i];

   return (1.f / (2.f * loss.size())) * sum;
}

Vec neural::crossentropy(const Vec &a, const Vec &y)
{
   assert(a.size() == y.size());

   static constexpr float eps = 1e-8f;
   int size = a.size();
   Vec result(size);

   for (int i = 0; i < size; ++i) {
      float y_v = y[i];
      float a_v = std::max(eps, a[i]);

      result[i] = y_v * std::log(a_v) + (1.f - y_v) * std::log(1.f - a_v);
   }

   return result;
}

Vec neural::crossentropyDerivative(const Vec &a, const Vec &y)
{
   assert(a.size() == y.size());

   static constexpr float eps = 1e-8f;
   int size = a.size();
   Vec result(size);

   for (int i = 0; i < size; ++i) {
      float y_v = y[i];
      float a_v = std::max(eps, a[i]);

      result[i] = -((y_v / a_v) + ((1.f - y_v) / (1.f - a_v)));
   }

   return result;
}

float neural::combinedCrossEntropy(const Vec &loss)
{
   float sum = 0.f;
   for (int i = 0; i < loss.size(); ++i)
      sum += loss[i];

   return -(1.f / loss.size()) * sum;
}

Vec neural::softmax(const Vec &v)
{
   float sum = 0.f;

   int size = v.size();
   assert(size < 1024 && "layer is too large!");

   float exps[1024];
   for (int i = 0; i < size; ++i) {
      exps[i] = exp(v[i]);
      sum += exps[i];
   }

   Vec result(size);
   for (int i = 0; i < size; ++i)
      result[i] = exps[i] / sum;

   return result;
}

Mat neural::softmax(const Mat &m)
{
   Mat result(m.rows, m.cols);
   float exps[1024];

   for (int i = 0; i < m.rows; ++i) {
      float sum = 0.f;
      assert(m.cols < 1024 && "layer is too large!");

      for (int j = 0; j < m.cols; ++j) {
         exps[j] = exp(m[i][j]);
         sum += exps[j];
      }

      for (int j = 0; j < m.cols; ++j)
         result[i][j] = exps[j] / sum;
   }

   return result;
}

void neural::dump(const Mat &m)
{
   std::cerr << m;
}

void neural::dump(const Vec &v)
{
   std::cerr << v;
}

float Vec::max() const
{
   float max = -std::numeric_limits<float>::infinity();
   for (int i = 0; i < rows; ++i) {
      float val = (*this)(i);
      if (val > max)
         max = val;
   }

   return max;
}

int Vec::argmax() const
{
   float max = -std::numeric_limits<float>::infinity();
   int maxIdx = 0;

   for (int i = 0; i < rows; ++i) {
      float val = (*this)(i);
      if (val > max) {
         max = val;
         maxIdx = i;
      }
   }

   return maxIdx;
}

float Vec::sqrMagnitude() const
{
   float sum = 0.f;
   for (int i = 0; i < rows; ++i)
      sum += (*this)[i] * (*this)[i];

   return sum;
}

float Vec::magnitude() const
{
   return sqrt(sqrMagnitude());
}

void Layer::updateBiases(const Vec &delta, float learningRate)
{
   assert(delta.size() == biases.size());

   int size = delta.size();
   for (int i = 0; i < size; ++i) {
      biases[i] -= delta[i] * learningRate;
   }
}

void Layer::updateWeights(const Mat &delta, float learningRate)
{
   assert(delta.size == weights.size);

   for (int i = 0; i < delta.rows; ++i) {
      for (int j = 0; j < delta.cols; ++j) {
         weights[i][j] -= delta[i][j] * learningRate;
      }
   }
}

cv::Mat_<float> createGaussianKernel1D(int kSize, float sigma)
{
   bool isEven = (kSize & 0x1) == 0;
   if (isEven)
      ++kSize;

   cv::Mat_<float> kernel(1, kSize);

   float sigmaSquared = pow(sigma, 2);
   float sigmaSquaredTwice = 2 * sigmaSquared;
   float factor = 1.f / (sigma * sqrt(2 * (float)M_PI));

   float sum = 0;
   int halfSize = kSize / 2;

   for (int x = 0; x < kSize; ++x) {
      float xSquared = pow(x - halfSize, 2.f);

      kernel[0][x] = factor * exp(-(xSquared / sigmaSquaredTwice));
      sum += kernel[0][x];
   }

   float missing = 1.f - sum;
   for (int x = 0; x < kSize; ++x) {
      kernel[0][x] += missing * (kernel[0][x] / sum);
   }

   if (isEven) {
      cv::Mat_<float> result(1, kSize - 1);
      int offset = 0;

      missing = 0.f;
      for (int i = 0; i < kSize; ++i) {
         if (i == halfSize) {
            offset = 1;
            missing = kernel[0][i];
            continue;
         }

         result[0][i - offset] = kernel[0][i];
      }

      result[0][halfSize - 1] += missing * .5f;
      result[0][halfSize] += missing * .5f;

      return result;
   }

   return kernel;
}

static std::vector<float> split(const std::string &fileName)
{
   std::ifstream file(fileName);
   std::string str((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());

   std::vector<std::string> result;
   std::istringstream is(str);

   std::copy(std::istream_iterator<std::string>(is),
             std::istream_iterator<std::string>(),
             std::back_inserter(result));

   std::vector<float> resultVec(result.size());
   std::transform(result.begin(), result.end(), resultVec.begin(), [](const std::string &s) {
      return std::stof(s);
   });

   return resultVec;
}

void Network::loadParameters(const std::string &biasFile,
                             const std::string &weightFile) const {
   auto biasValues = split(biasFile);
   auto weightValues = split(weightFile);

   assert((int)biasValues[0] == layers.size() - 1);

   int n = 1;
   for (int i = 1; i < layers.size(); ++i) {
      auto &layer = *layers[i];
      assert((int)biasValues[n] == layer.size());
      ++n;

      Vec b(layer.size());
      for (int j = 0; j < layer.size(); ++j) {
         b[j] = biasValues[n++];
      }

      layer.biases = b;
   }

   assert((int)weightValues[0] == layers.size() - 1);

   n = 1;
   for (int i = 1; i < layers.size(); ++i) {
      auto &layer = *layers[i];
      int rows = weightValues[n++];
      int cols = weightValues[n++];

      assert(rows == layer.weights.rows);
      assert(cols == layer.weights.cols);

      Mat w(rows, cols);
      for (int j = 0; j < rows; ++j) {
         for (int k = 0; k < cols; ++k) {
            w[j][k] = weightValues[n++];
         }
      }

      layer.weights = w;
   }
}

void Network::saveParameters(const std::string &biasFile,
                             const std::string &weightFile) const {
   std::ofstream bias_file(biasFile);
   std::ofstream weight_file(weightFile);

   bias_file << layers.size() - 1;
   for (int n = 1; n < layers.size(); ++n) {
      bias_file << ' ';

      auto &b = layers[n]->getBiases();
      bias_file << b.size();

      for (int i = 0; i < b.size(); ++i) {
         bias_file << ' ' << b[i];
      }
   }

   weight_file << layers.size() - 1;
   for (int n = 1; n < layers.size(); ++n) {
      weight_file << ' ';

      auto &w = layers[n]->getWeights();
      weight_file << w.rows << ' ' << w.cols;

      for (int i = 0; i < w.rows; ++i) {
         for (int j = 0; j < w.cols; ++j) {
            weight_file << ' ' << w[i][j];
         }
      }
   }

   bias_file.close();
   weight_file.close();
}

void Network::initializeWeightsAndBiases(InitializationStrategy init) {
   switch (init) {
   case InitializationStrategy::Random:
      for (auto &layer : layers) {

      }

      break;
   case InitializationStrategy::Gaussian: {
      for (auto &layer : layers) {
         int layerSize = layer->size();
         if (layer->isInputLayer()) {
            for (int i = 0; i < layerSize; ++i) {
               layer->biases[i] = 0.0f;
            }

            continue;
         }

         int inputSize = layer->previousLayer->size();

         // Initialize biases with a gaussian with sigma = 1.
         auto biasGaussian = createGaussianKernel1D(layerSize, 1.f);
         for (int i = 0; i < layerSize; ++i) {
            layer->biases[i] = biasGaussian(0, i);
         }

         // Initialize weights with a gaussian with sigma = 1 / sqrt(n_in).
         auto weightGaussian = createGaussianKernel1D(inputSize, 1.f / sqrt(inputSize));
         for (int i = 0; i < layerSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
               layer->weights(i, j) = weightGaussian(0, j);
            }
         }
      }

      break;
   }
   }
}

Vec Network::operator()(const Vec &inputActivation) const
{
   assert(inputActivation.size() == layers[0]->size()
      && "input size does not equal input layer size!");

   Vec a = inputActivation;
   for (int i = 1; i < layers.size(); ++i) {
      auto &layer = *layers[i];

      const Vec &b = layer.getBiases();
      const Mat &w = layer.getWeights();

      Mat z = (w * a) + b;
      a = layer.getActivationFn()(z);
   }

   return a;
}

Mat Network::operator()(const Mat &inputActivations) const
{
   Mat result(inputActivations.rows, layers.back()->size());
   for (int i = 0; i < inputActivations.rows; ++i) {
      Vec row = this->operator()(Vec(inputActivations, i));
      for (int j = 0; j < inputActivations.cols; ++j)
         result[i][j] = row[j];
   }

   return result;
}

std::pair<Vec (*)(const Vec &, const Vec &), Vec (*)(const Vec &, const Vec &)> Network::getCostAndDerivativeFn() const
{
   Vec (*cost)(const Vec&, const Vec&);
   Vec (*costDerivative)(const Vec&, const Vec&);

   switch (costFn) {
   case CostFunction::Quadratic:
      cost = quadraticCost;
      costDerivative = quadraticCostDerivative;
      break;
   case CostFunction::Crossentropy:
      cost = crossentropy;
      costDerivative = crossentropyDerivative;
      break;
   }

   return std::make_tuple(cost, costDerivative);
}

auto Network::getLossFn() const -> float(*)(const Vec&)
{
   switch (costFn) {
   case CostFunction::Quadratic:
      return combinedQuadraticCost;
   case CostFunction::Crossentropy:
      return combinedCrossEntropy;
   }
}

std::vector<Vec> Network::getEmptyBiasMatrix(float init) const
{
   std::vector<Vec> result;
   for (auto &layer : layers) {
      result.emplace_back(layer->size(), init);
   }

   return result;
}

std::vector<Mat> Network::getEmptyWeightMatrix(float init) const
{
   std::vector<Mat> result;
   for (auto &layer : layers) {
      auto &weights = layer->getWeights();
      result.emplace_back(weights.rows, weights.cols, init);
   }

   return result;
}

std::tuple<float, std::vector<Vec>, std::vector<Mat>>
Network::calculateLossAndGradient(const Mat &dataset, const Vec &labels) const
{
   auto costAndDerivative = getCostAndDerivativeFn();
   auto *cost = costAndDerivative.first;
   auto *costDerivative = costAndDerivative.second;
   auto *lossFn = getLossFn();

   float loss = 0.f;
   int outputLayerSize = layers.back()->size();
   int numLayers = layers.size();

   std::vector<std::vector<Vec>> nabla_bs;
   std::vector<std::vector<Mat>> nabla_ws;

   Vec expectedOutput(outputLayerSize);
   for (int sample = 0; sample < dataset.rows; ++sample) {
      int expectedLabel = (int)labels[sample];

      for (int i = 0; i < outputLayerSize; ++i) {
         expectedOutput[i] = i == expectedLabel ? 1.f : 0.f;
      }

      // --- Forward pass ---
      Vec a(dataset, sample);
      std::vector<Vec> activations { a };
      std::vector<Mat> zs;

      for (int i = 1; i < layers.size(); ++i) {
         auto &layer = *layers[i];

         const Vec &b = layer.getBiases();
         const Mat &w = layer.getWeights();

         Mat z = (w * a) + b;
         zs.push_back(z);

         a = layer.getActivationFn()(z);
         activations.push_back(a);
      }

      // --- Backward Pass ---
      // Compute the error of the last layer.
      Vec delta_L = costDerivative(activations.back(), expectedOutput)
         .mul(layers.back()->getActivationDerivativeFn()(zs.back()));

      float currentLoss = lossFn(cost(softmax(activations.back()), expectedOutput));
      loss += currentLoss;

      auto nabla_b = getEmptyBiasMatrix(0.f);
      auto nabla_w = getEmptyWeightMatrix(0.f);

      nabla_b.back() = delta_L;

      Vec a_T;
      cv::transpose(activations[activations.size() - 2], a_T);
      nabla_w.back() = delta_L * a_T;

      Mat w_T;
      for (int i = 2; i < numLayers; ++i) {
         auto &layer = layers[numLayers - i];

         Mat &z = zs[numLayers - 1 - i];
         Vec sp = layer->getActivationDerivativeFn()(z);

         // Compute loss for next layer.
         cv::transpose(layers[numLayers - i + 1]->getWeights(), w_T);

         delta_L = (w_T * delta_L).mul(sp);
         nabla_b[numLayers - i] = delta_L;

         cv::transpose(activations[numLayers - i - 1], a_T);
         nabla_w[numLayers - i] = delta_L * a_T;
      }

      nabla_bs.emplace_back(move(nabla_b));
      nabla_ws.emplace_back(move(nabla_w));
   }

   auto nabla_b = getEmptyBiasMatrix(0.f);
   auto nabla_w = getEmptyWeightMatrix(0.f);

   for (auto &vec : nabla_bs) {
      for (int n = 0; n < vec.size(); ++n) {
         auto &nb = vec[n];
         for (int i = 0; i < nb.size(); ++i) {
            nabla_b[n][i] += nb[i];
         }
      }
   }

   for (auto &vec : nabla_ws) {
      for (int n = 0; n < vec.size(); ++n) {
         auto &nw = vec[n];
         for (int i = 0; i < nw.rows; ++i) {
            for (int j = 0; j < nw.cols; ++j) {
               nabla_w[n][i][j] += nw[i][j];
            }
         }
      }
   }

   for (auto &val : nabla_b) {
      val /= nabla_bs.size();
   }

   for (auto &val : nabla_w) {
      val /= nabla_ws.size();
   }

   return { loss / (float)dataset.rows, move(nabla_b), move(nabla_w) };
}

template<class T>
static std::vector<T> slice(const std::vector<T> &vec, int from, int to)
{
   auto begin = vec.begin() + std::clamp(from, 0, (int)vec.size() - 1);
   auto end = vec.begin() + std::clamp(to, from, (int)vec.size() - 1);

   return std::vector<T>(begin, end);
}

void Network::SGD(const Mat &dataset, const Vec &labels,
                  int epochs, int miniBatchSize, float learningRate,
                  const Mat &testData, const Vec &testLabels) {
   int n = dataset.rows;
   int outputLayerSize = layers.back()->size();

   std::vector<std::pair<Vec, Vec>> trainingData(n);
   for (int i = 0; i < n; ++i) {
      Vec img(dataset, i);
      Vec expectedOutput(outputLayerSize);

      int expectedLabel = (int)labels[i];
      for (int j = 0; j < outputLayerSize; ++j) {
         expectedOutput[j] = j == expectedLabel ? 1.f : 0.f;
      }

      trainingData[i] = std::make_pair(img, expectedOutput);
   }

   std::cout << "Epoch: " << 0 << " | Accuracy: " << this->evaluate(testData, testLabels) << std::endl;

   auto nabla_b = getEmptyBiasMatrix(0.f);
   auto nabla_w = getEmptyWeightMatrix(0.f);

   std::vector<std::vector<std::pair<Vec, Vec>>> miniBatches;
   auto rng = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
   for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(trainingData.begin(), trainingData.end(), rng);

      for (int k = 0; k < n; k += miniBatchSize) {
         miniBatches.emplace_back(slice(trainingData, k, k + miniBatchSize));
      }

      float loss = 0.f;
      for (auto &batch : miniBatches) {
         loss = updateMiniBatch(batch, learningRate, nabla_b, nabla_w);
      }

//      std::cout << "\33[2K\r" << std::flush;
      float accuracy = this->evaluate(testData, testLabels);
      std::cout << "Epoch: " << (epoch + 1) << " | Loss: " << loss << " | Accuracy: " << accuracy << std::endl;

      miniBatches.clear();
   }
}

float Network::updateMiniBatch(const std::vector<std::pair<Vec, Vec>> &miniBatch,
                               float learningRate, std::vector<Vec> &nabla_b,
                               std::vector<Mat> &nabla_w) {
   float latestLoss = 0.f;
   for (auto &data : miniBatch) {
      auto [loss, delta_nb, delta_nw] = backprop(data.first, data.second);
      latestLoss = loss;

      // Update ∇b
      for (int n = 1; n < delta_nb.size(); ++n)
         for (int i = 0; i < nabla_b[n].size(); ++i)
            nabla_b[n][i] += delta_nb[n][i];

      // Update ∇w
      for (int n = 1; n < delta_nw.size(); ++n)
         for (int i = 0; i < nabla_w[n].rows; ++i)
            for (int j = 0; j < nabla_w[n].cols; ++j)
            nabla_w[n][i][j] += delta_nw[n][i][j];
   }

   // Update biases
   for (int n = 1; n < nabla_b.size(); ++n) {
      layers[n]->updateBiases(nabla_b[n], learningRate / miniBatch.size());
      layers[n]->updateWeights(nabla_w[n], learningRate / miniBatch.size());
   }

   // Reset ∇b
   for (int n = 1; n < nabla_b.size(); ++n)
      for (int i = 0; i < nabla_b[n].size(); ++i)
         nabla_b[n][i] = 0.f;

   // Reset ∇w
   for (int n = 1; n < nabla_w.size(); ++n)
      for (int i = 0; i < nabla_w[n].rows; ++i)
         for (int j = 0; j < nabla_w[n].cols; ++j)
            nabla_w[n][i][j] = 0.f;

   return latestLoss;
}

std::tuple<float, std::vector<Vec>, std::vector<Mat>>
Network::backprop(const Vec &x, const Vec &y)
{
   auto costAndDerivative = getCostAndDerivativeFn();
   auto *cost = costAndDerivative.first;
   auto *costDerivative = costAndDerivative.second;
   auto *lossFn = getLossFn();
   int numLayers = layers.size();

   // --- Forward pass ---
   Vec a = x;

   activations.clear();
   activations.push_back(a);

   zs.clear();

   for (int i = 1; i < layers.size(); ++i) {
      auto &layer = *layers[i];

      const Vec &b = layer.getBiases();
      const Mat &w = layer.getWeights();

      Mat z = (w * a) + b;
      zs.push_back(z);

      a = layer.getActivationFn()(z);
      activations.push_back(a);
   }

   // --- Backward Pass ---
   // Compute the error of the last layer.
   Vec delta_L = costDerivative(activations.back(), y)
      .mul(layers.back()->getActivationDerivativeFn()(zs.back()));

   float loss = lossFn(cost(softmax(activations.back()), y));

   auto nabla_b = getEmptyBiasMatrix(0.f);
   auto nabla_w = getEmptyWeightMatrix(0.f);

   nabla_b.back() = delta_L;

   Vec a_T;
   cv::transpose(activations[activations.size() - 2], a_T);
   nabla_w.back() = delta_L * a_T;

   Mat w_T;
   for (int i = 2; i < numLayers; ++i) {
      auto &layer = layers[numLayers - i];

      Mat &z = zs[numLayers - 1 - i];
      Vec sp = layer->getActivationDerivativeFn()(z);

      // Compute loss for next layer.
      cv::transpose(layers[numLayers - i + 1]->getWeights(), w_T);

      delta_L = (w_T * delta_L).mul(sp);
      nabla_b[numLayers - i] = delta_L;

      cv::transpose(activations[numLayers - i - 1], a_T);
      nabla_w[numLayers - i] = delta_L * a_T;
   }

   return { loss, move(nabla_b), move(nabla_w) };
}

void Network::applyGradient(const std::vector<Vec> &nabla_b,
                            const std::vector<Mat> &nabla_w,
                            float learningRate)
{
   for (int i = 1; i < layers.size(); ++i) {
      auto &layer = layers[i];
      layer->updateBiases(nabla_b[i], learningRate);
      layer->updateWeights(nabla_w[i], learningRate);
   }
}

[[maybe_unused]]
static void printAsDigit(const Vec &v)
{
   for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
         std::cerr<< (v[i * 28 + j] > 0.f ? "X" : "-");
      }
      std::cerr<<std::endl;
   }
}

float Network::evaluate(const Mat &dataset, const Vec &labels) const
{
   if (std::rand() == 3)
      printAsDigit(Vec());

   int correctSamples = 0;
   int outputLayerSize = layers.back()->size();

   for (int sample = 0; sample < dataset.rows; ++sample) {
      Vec inputActivation(dataset, sample);
      int expectedLabel = (int)labels[sample];

      Vec expectedOutput(outputLayerSize);
      for (int i = 0; i < outputLayerSize; ++i) {
         expectedOutput[i] = i == expectedLabel ? 1.f : 0.f;
      }

      Vec z = (*this)(inputActivation);
      Vec sm = softmax(z);

      int label = sm.argmax();
      if (label == expectedLabel)
         ++correctSamples;
   }

   return (float)correctSamples / (float)dataset.rows;
}

void Network::SGD(const Mat &dataset, const Vec &labels,
                  int epochs, int miniBatchSize, float learningRate,
                  bool evaluate) {
   std::vector<int> samples(dataset.rows);
   for (int i = 0; i < dataset.rows; ++i) {
      samples[i] = i;
   }

   auto rng = std::default_random_engine();
   Mat minibatch(miniBatchSize, dataset.cols);
   Vec batchLabels(miniBatchSize);

   for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(samples.begin(), samples.end(), rng);

      int rest = dataset.rows;
      int currentSample = 0;

      while (rest > 0) {
         int batchSize = std::min(rest, miniBatchSize);
         int firstSample = currentSample;
         int end = currentSample + batchSize;

         for (; currentSample < end; ++currentSample) {
            for (int j = 0; j < dataset.cols; ++j)
               minibatch[currentSample - firstSample][j] = dataset[samples[currentSample]][j];

            batchLabels[currentSample - firstSample] = labels[currentSample];
         }

         auto[loss, nabla_b, nabla_w] = calculateLossAndGradient(minibatch, batchLabels);

         if (epoch != 0 || currentSample != 0)
            std::cout << "\33[2K\r" << std::flush;

         std::cout << "Epoch: " << epoch << " | Iter: " << currentSample << " | Loss: " << loss;
         if (evaluate) {
            float acc = this->evaluate(minibatch, batchLabels);
            std::cout << " | Accuracy: " << acc;
         }

         applyGradient(nabla_b, nabla_w, learningRate / batchSize);
         rest -= batchSize;
      }
   }
}