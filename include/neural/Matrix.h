
#ifndef NEURALNETWORK_MATRIX_H
#define NEURALNETWORK_MATRIX_H

#include <cassert>
#include <iostream>
#include <type_traits>

namespace neural {
namespace detail {

template<unsigned Dim, unsigned TotalSize, unsigned Size, unsigned ...Sizes>
class _Tensor: public _Tensor<Dim - 1, TotalSize * Size, Sizes...> {

};

template<unsigned Size, unsigned _TotalSize>
class _Tensor<0, _TotalSize, Size> {
protected:
   static constexpr unsigned TotalSize = _TotalSize * Size;

   /// The raw tensor data.
   float data[TotalSize];
};

template<unsigned...>
class _UnsignedParameterPack {};

template<class...>
class _TypeParameterPack {};

} // namespace detail

template<unsigned Size1, unsigned ...RestSizes>
class Tensor: public detail::_Tensor<sizeof...(RestSizes), 1, Size1, RestSizes...> {
private:
   template<class Index, class ...Indices, unsigned Size, unsigned ..._RestSizes>
   float& at_impl(unsigned idxProd,
                  detail::_TypeParameterPack<Index, Indices...> _indices,
                  detail::_UnsignedParameterPack<Size, _RestSizes...> _sizes,
                  unsigned idx1, Indices ...rest)
   {
      assert(idx1 < Size && "index is out of bounds!");
      return at_impl(idxProd * idx1,
                     detail::_TypeParameterPack<Indices...>(),
                     detail::_UnsignedParameterPack<_RestSizes...>(),
                     std::forward<Indices&&>(rest)...);
   }

   float& at_impl(unsigned idxProd,
                  detail::_TypeParameterPack<> _indices,
                  detail::_UnsignedParameterPack<> _sizes)
   {
      return this->data[idxProd];
   }

   // Vector specialization
   template<unsigned Dim>
   float print_impl(std::ostream &OS)
   {
      OS << "[";

      for (int i = 0; i < Dim; ++i) {
         if (i != 0)
            OS << "\n ";

         OS << this->at(i);
      }

      OS << "]";
   }

   // Matrix specialization
   template<unsigned Rows, unsigned Cols>
   float print_impl(std::ostream &OS)
   {
      OS << "[";

      for (int i = 0; i < Rows; ++i) {
         if (i != 0)
            OS << '\n';

         for (int j = 0; j < Rows; ++j) {
            if (j != 0)
               OS << ' ';

            OS << this->at(i, j);
         }
      }

      OS << "]";
   }

public:
   /// Initializer.
   explicit Tensor(float init = 0.f)
   {
      if (init == 0.f)
      {
         std::memset(this->data, 0, sizeof(this->data));
      }
      else
      {
         for (int i = 0; i < this->TotalSize; ++i)
            this->data[i] = init;
      }
   }

   /// Get an element from the tensor by indices.
   template<class ...Indices>
   auto at(unsigned idx1, Indices ...indices)
      -> typename std::enable_if<sizeof...(RestSizes) == sizeof...(Indices), float&>::type
   {
      assert(idx1 < Size1 && "index is out of bounds in dimension 0!");
      return at_impl(idx1, detail::_TypeParameterPack<Indices...>(),
                     detail::_UnsignedParameterPack<RestSizes...>(),
                     std::forward<Indices&&>(indices)...);
   }

   /// Print the tensor to a stream.
   void print(std::ostream &OS)
   {
      print_impl<Size1, RestSizes...>(OS);
   }

   /// Print the tensor to stderr.
   void dump()
   {
      print_impl<Size1, RestSizes...>(std::cerr);
   }
};

template<unsigned Dim>
using Vector = Tensor<Dim>;

template<unsigned Rows, unsigned Cols>
using Matrix = Tensor<Rows, Cols>;

} // namespace neural

#endif //NEURALNETWORK_MATRIX_H
