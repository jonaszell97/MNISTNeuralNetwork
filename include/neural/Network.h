
#ifndef NEURALNETWORK_NETWORK_H
#define NEURALNETWORK_NETWORK_H

#include <opencv2/opencv.hpp>

#include <memory>

namespace neural {

using Mat = cv::Mat_<float>;

class Vec: public cv::Mat_<float> {
public:
   explicit Vec(int size = 0) : cv::Mat_<float>(size, 1)
   {}

   explicit Vec(int size, float init) : cv::Mat_<float>(size, 1, init)
   {}

   Vec(int size, float *data) : cv::Mat_<float>(size, 1, data)
   {}

   Vec(const cv::Mat_<float> &mat) : cv::Mat_<float>(mat)
   {
      assert(mat.cols == 1 && "not a vector");
   }

   Vec(const cv::Mat_<float> &mat, int row) : cv::Mat_<float>(mat.cols, 1)
   {
      for (int i = 0; i < mat.cols; ++i)
         (*this)[i] = mat(row, i);
   }

   Vec(const cv::MatExpr &mat) : cv::Mat_<float>(mat)
   {
      assert(this->cols == 1 && "not a vector");
   }

   Vec(const std::initializer_list<float> init) : cv::Mat_<float>(init.size(), 1, const_cast<float*>(init.begin()))
   {
      assert(this->cols == 1 && "not a vector");
   }

   const float &operator[](int idx) const
   {
      return this->cv::Mat_<float>::operator()(idx, 0);
   }

   const float &operator()(int idx) const
   {
      return this->cv::Mat_<float>::operator()(idx, 0);
   }

   float &operator[](int idx)
   {
      return this->cv::Mat_<float>::operator()(idx, 0);
   }

   float &operator()(int idx)
   {
      return this->cv::Mat_<float>::operator()(idx, 0);
   }

   Vec operator-(const Vec &rhs) const
   {
      return (cv::Mat_<float>)(((const cv::Mat_<float>&)*this) - ((const cv::Mat_<float>&)rhs));
   }

   Vec operator+(const Vec &rhs) const
   {
      return (cv::Mat_<float>)(((const cv::Mat_<float>&)*this) + ((const cv::Mat_<float>&)rhs));
   }

   Mat operator*(const Vec &rhs) const
   {
      return (((const cv::Mat_<float>&)*this) * ((const cv::Mat_<float>&)rhs));
   }

   Vec operator/(const Vec &rhs) const
   {
      return (cv::Mat_<float>)(((const cv::Mat_<float>&)*this) / ((const cv::Mat_<float>&)rhs));
   }

   float max() const;
   int argmax() const;

   float sqrMagnitude() const;
   float magnitude() const;

   int size() const { return this->rows; }
};

void dump(const Vec &v);
void dump(const Mat &m);

/// The identity activation function.
float identity(float);
Vec identity(const Vec&);

float identityDerivative(float);
Vec identityDerivative(const Vec&);

/// The sigmoid activation function.
float sigmoid(float);
Vec sigmoid(const Vec&);

/// The derivative of the sigmoid activation function.
float sigmoidDerivative(float);
Vec sigmoidDerivative(const Vec&);

/// The ReLU activation function.
float relu(float);
Vec relu(const Vec&);

/// The derivative ReLU activation function.
float reluDerivative(float);
Vec reluDerivative(const Vec&);

class Network;

class Layer {
public:
   enum Type {
      INPUT,
      OUTPUT,
      HIDDEN,
   };

   enum class ActivationFunction {
      Sigmoid,
      Identity,
      ReLU,
   };

private:
   /// The layer's type.
   Type type;

   /// The layer's activation function.
   ActivationFunction activationFn;

   /// The previous layer.
   Layer *previousLayer;

   /// The neurons activations.
   Vec activations;

   /// The neurons biases.
   Vec biases;

   /// The weights from this layer to the previous one.
   Mat weights;

public:
   explicit Layer(int size, ActivationFunction activationFn)
      : type(INPUT), activationFn(activationFn), previousLayer(nullptr),
        activations(size), biases(size), weights()
   {

   }

   Layer(int size, Layer &prev, ActivationFunction activationFn)
      : type(HIDDEN), activationFn(activationFn), previousLayer(&prev),
        activations(size), biases(size), weights(size, prev.size())
   {

   }

   Layer(const Layer&) = delete;
   Layer(Layer&&) = default;

   const Layer &operator=(const Layer&) = delete;
   Layer &operator=(Layer&&) = default;

   [[nodiscard]] int size() const { return activations.size(); }
   [[nodiscard]] Type getType() const { return type; }
   [[nodiscard]] bool isInputLayer() const { return type == INPUT; }
   [[nodiscard]] bool isHiddenLayer() const { return type == HIDDEN; }
   [[nodiscard]] bool isOutputLayer() const { return type == OUTPUT; }

   [[nodiscard]] Vec (*getActivationFn())(const Vec&)
   {
      switch (activationFn) {
      case ActivationFunction::Identity: return identity;
      case ActivationFunction::Sigmoid: return sigmoid;
      case ActivationFunction::ReLU: return relu;
      }
   }

   [[nodiscard]] Vec (*getActivationDerivativeFn())(const Vec&)
   {
      switch (activationFn) {
      case ActivationFunction::Identity: return identityDerivative;
      case ActivationFunction::Sigmoid: return sigmoidDerivative;
      case ActivationFunction::ReLU: return reluDerivative;
      }
   }

   [[nodiscard]] const Vec &getActivations() const { return activations; }
   [[nodiscard]] const Vec &getBiases() const { return biases; }
   [[nodiscard]] const Mat &getWeights() const { return weights; }

   void updateBiases(const Vec &delta, float learningRate);
   void updateWeights(const Mat &delta, float learningRate);

   friend class Network;
};

/// The cross-entropy cost function.
Vec crossentropy(const Vec &a, const Vec &y);
Vec crossentropyDerivative(const Vec &a, const Vec &y);
float combinedCrossEntropy(const Vec &loss);

/// The quadratic cost function.
Vec quadraticCost(const Vec &a, const Vec &y);
Vec quadraticCostDerivative(const Vec &a, const Vec &y);
float combinedQuadraticCost(const Vec &loss);

/// The softmax function.
Vec softmax(const Vec&);
Mat softmax(const Mat&);

class Network {
public:
   enum class CostFunction {
      Quadratic,
      Crossentropy,
   };

private:
   /// The network's cost function.
   CostFunction costFn;

   /// The network's layers.
   std::vector<std::unique_ptr<Layer>> layers;

   /// Caches used for backpropagation.
   std::vector<Vec> activations;
   std::vector<Mat> zs;

   Network(CostFunction costFn, std::vector<std::unique_ptr<Layer>> &&layers)
      : costFn(costFn), layers(move(layers))
   {
      if (!this->layers.empty())
         this->layers.back()->type = Layer::OUTPUT;
   }

   std::vector<Vec> getEmptyBiasMatrix(float init = 0.f) const;
   std::vector<Mat> getEmptyWeightMatrix(float init = 0.f) const;


   /// Do an SGD step for a mini batch.
   float updateMiniBatch(const std::vector<std::pair<Vec, Vec>> &miniBatch,
                         float learningRate, std::vector<Vec> &nabla_b,
                         std::vector<Mat> &nabla_w);

public:
   enum class InitializationStrategy {
      Random,
      Gaussian,
   };

   /// Helper class for creating a network.
   class Builder {
      /// The added layers so far.
      std::vector<std::unique_ptr<Layer>> layers;

      /// The cost function.
      CostFunction costFn;

      explicit Builder(CostFunction costFn)
         : layers(), costFn(costFn)
      {}

      friend class Network;

   public:
      Builder &addLayer(int size, Layer::ActivationFunction activationFn = Layer::ActivationFunction::Identity)
      {
         if (layers.empty()) {
            layers.emplace_back(std::make_unique<Layer>(size, activationFn));
         }
         else {
            layers.emplace_back(std::make_unique<Layer>(size, *layers.back(), activationFn));
         }

         return *this;
      }

      Network build(InitializationStrategy init = InitializationStrategy::Random)
      {
         Network net(costFn, move(layers));
         net.initializeWeightsAndBiases(init);

         return net;
      }
   };

   /// Create a network builder.
   static Builder create(CostFunction costFn = CostFunction::Crossentropy)
   {
      return Builder(costFn);
   }

   /// Load parameters from a text file.
   void loadParameters(const std::string &biasFile, const std::string &weightFile) const;

   /// Store parameters to a text file.
   void saveParameters(const std::string &biasFile, const std::string &weightFile) const;

   /// Return a specific layer.
   [[nodiscard]] const Layer &getLayer(unsigned i) const
   {
      assert(i < layers.size());
      return *layers[i];
   }

   std::pair<Vec(*)(const Vec&, const Vec&), Vec(*)(const Vec&, const Vec&)>
   getCostAndDerivativeFn() const;

   auto getLossFn() const -> float(*)(const Vec&);

   /// Initialize the weights and biases.
   void initializeWeightsAndBiases(InitializationStrategy init = InitializationStrategy::Random);

   /// Calculate the activation of the output layer for a specified input activation vector.
   Vec operator()(const Vec &inputActivation) const;

   /// Calculate the activation of the output layer for a specified vector of features.
   Mat operator()(const Mat &inputActivations) const;

   /// Calculate the current loss of the network along with the gradient.
   std::tuple<float, std::vector<Vec>, std::vector<Mat>>
   calculateLossAndGradient(const Mat &dataset, const Vec &labels) const;

   /// Calculate the current loss of the network along with the gradient for a single training sample.
   std::tuple<float, std::vector<Vec>, std::vector<Mat>>
   calculateLossAndGradient(const Vec &inputActivation, int expectedLabel) const;

   /// Apply the calculated gradient.
   void applyGradient(const std::vector<Vec> &nabla_b,
                      const std::vector<Mat> &nabla_w,
                      float learningRate);

   /// Perform a single backpropagation step.
   std::tuple<float, std::vector<Vec>, std::vector<Mat>> backprop(const Vec &x, const Vec &y);

   /// Evaluate the network's accuracy on a set of inputs.
   float evaluate(const Mat &dataset, const Vec &labels) const;

   /// Train the network using stochastic gradient descent.
   void SGD(const Mat &dataset, const Vec &labels,
            int epochs, int miniBatchSize, float learningRate,
            bool evaluate = false);

   /// Train the network using stochastic gradient descent.
   void SGD(const Mat &dataset, const Vec &labels,
            int epochs, int miniBatchSize, float learningRate,
            const Mat &testData, const Vec &testLabels);
};

} // namespace neural

#endif //NEURALNETWORK_NETWORK_H
