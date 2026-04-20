

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <map>
#include <queue>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "QnnTypes.h"
#include "qnn-api/QnnTypeMacros.hpp"
#include "geniex-utils.h"

template <typename T_QuantType>
bool floatToTfN(
    T_QuantType* out, float* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool tfNToFloat(
    float* out, T_QuantType* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool castToFloat(float* out, T_QuantType* in, size_t numElements);

template <typename T_QuantType>
bool castFromFloat(T_QuantType* out, float* in, size_t numElements);

template <typename T_QuantType>
bool floatToTfN(
    T_QuantType* out, float* in, int32_t offset, float scale, size_t numElements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "floatToTfN supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    return false;
  }

  size_t dataTypeSizeInBytes = sizeof(T_QuantType);
  size_t bitWidth            = dataTypeSizeInBytes * g_bitsPerByte;
  double trueBitWidthMax     = pow(2, bitWidth) - 1;
  double encodingMin         = offset * scale;
  double encodingMax         = (trueBitWidthMax + offset) * scale;
  double encodingRange       = encodingMax - encodingMin;

  for (size_t i = 0; i < numElements; ++i) {
    int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);
    if (quantizedValue < 0)
      quantizedValue = 0;
    else if (quantizedValue > (int)trueBitWidthMax)
      quantizedValue = (int)trueBitWidthMax;
    out[i] = static_cast<T_QuantType>(quantizedValue);
  }
  return true;
}

template bool floatToTfN<uint8_t>(
    uint8_t* out, float* in, int32_t offset, float scale, size_t numElements);

template bool floatToTfN<uint16_t>(
    uint16_t* out, float* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool tfNToFloat(
    float* out, T_QuantType* in, int32_t offset, float scale, size_t numElements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "tfNToFloat supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    return false;
  }
  for (size_t i = 0; i < numElements; i++) {
    double quantizedValue = static_cast<double>(in[i]);
    double offsetDouble   = static_cast<double>(offset);
    out[i]                = static_cast<double>((quantizedValue + offsetDouble) * scale);
  }
  return true;
}

template bool tfNToFloat<uint8_t>(
    float* out, uint8_t* in, int32_t offset, float scale, size_t numElements);

template bool tfNToFloat<uint16_t>(
    float* out, uint16_t* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool castToFloat(float* out, T_QuantType* in, size_t numElements) {
  if (nullptr == out || nullptr == in) {
    return false;
  }
  for (size_t i = 0; i < numElements; i++) {
    out[i] = static_cast<float>(in[i]);
  }
  return true;
}

template bool castToFloat<uint8_t>(float* out,
                                  uint8_t* in,
                                  size_t numElements);

template bool castToFloat<uint16_t>(float* out,
                                  uint16_t* in,
                                  size_t numElements);

template bool castToFloat<uint32_t>(float* out,
                                  uint32_t* in,
                                  size_t numElements);

template bool castToFloat<uint64_t>(float* out,
                                  uint64_t* in,
                                  size_t numElements);

template bool castToFloat<int8_t>(float* out,
                                int8_t* in,
                                size_t numElements);

template bool castToFloat<int16_t>(float* out,
                                  int16_t* in,
                                  size_t numElements);

template bool castToFloat<int32_t>(float* out,
                                  int32_t* in,
                                  size_t numElements);

template bool castToFloat<int64_t>(float* out,
                                  int64_t* in,
                                  size_t numElements);

template <typename T_QuantType>
bool castFromFloat(T_QuantType* out, float* in, size_t numElements) {
  if (nullptr == out || nullptr == in) {
    return false;
  }
  for (size_t i = 0; i < numElements; i++) {
    out[i] = static_cast<T_QuantType>(in[i]);
  }
  return true;
}

template bool castFromFloat<uint8_t>(uint8_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<uint16_t>(uint16_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<uint32_t>(uint32_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<uint64_t>(uint64_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<int8_t>(int8_t* out,
                                float* in,
                                size_t numElements);

template bool castFromFloat<int16_t>(int16_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<int32_t>(int32_t* out,
                                  float* in,
                                  size_t numElements);

template bool castFromFloat<int64_t>(int64_t* out,
                                  float* in,
                                  size_t numElements);

// Utility functions (keep these as standalone functions)
std::vector<std::string> split(const std::string& str) {
    std::vector<std::string> words;
    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(',', pos)) != std::string::npos) {
        std::string word = str.substr(prev, pos - prev);
        if (word.length() > 0) {
            words.push_back(word);
        }
        prev = ++pos;
    }
    std::string word = str.substr(prev, pos - prev);
    if (word.length() > 0) {
        words.push_back(word);
    }
    return words;
}

std::vector<float> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

void printTensorStats(const std::vector<std::vector<float>>& outputData) {
    for (size_t i = 0; i < outputData.size(); i++) {
        const auto& tensor = outputData[i];
        std::cout << "tensor " << i << " has " << tensor.size() << " elements" << std::endl;
        if (!tensor.empty()) {
            auto minElement = *std::min_element(tensor.begin(), tensor.end());
            auto maxElement = *std::max_element(tensor.begin(), tensor.end());
            size_t maxIndex = std::distance(tensor.begin(), std::max_element(tensor.begin(), tensor.end()));
            float sum = std::accumulate(tensor.begin(), tensor.end(), 0.0f);
            float mean = sum / tensor.size();
            std::cout << "Tensor " << i << " stats - Min: " << minElement << ", Max: " << maxElement << ", Mean: " << mean
                      << ", Max Index: " << maxIndex << std::endl;
            std::cout << "First 20 values: ";
            for (size_t j = 0; j < 20 && j < tensor.size(); j++) {
                std::cout << tensor[j];
                if (j < 19 && j < tensor.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Tensor " << i << " is empty." << std::endl;
        }
    }
}

size_t getTensorSize(const Qnn_Tensor_t* tensor)
{
    size_t numElems = 1;
    for (uint32_t d = 0; d < QNN_TENSOR_GET_RANK(tensor); ++d)
        numElems *= QNN_TENSOR_GET_DIMENSIONS(tensor)[d];

    auto it = g_dataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(tensor));
    if (it == g_dataTypeToSize.end())
        throw std::runtime_error("Unsupported tensor data type");
    return numElems * it->second;
}

void populateTensorSizeMap(std::unordered_map<std::string, size_t>& tensorSizeMap, const Qnn_Tensor_t* tensors,
                           uint32_t numTensors) {
    for (uint32_t i = 0; i < numTensors; ++i) {
        const Qnn_Tensor_t& tensor = tensors[i];
        std::string name = std::string(QNN_TENSOR_GET_NAME(tensor));
        tensorSizeMap[name] = getTensorSize(&tensor);
    }
}

void convertTensorDataToFloat(const Qnn_Tensor_t& tensor, void* bufferData, uint32_t totalElements, 
                            std::vector<float>& outputVector, size_t tensorIndex) {
    switch (QNN_TENSOR_GET_DATA_TYPE(&tensor)) {
        case QNN_DATATYPE_FLOAT_32: {
            float* typedBuffer = static_cast<float*>(bufferData);
            std::memcpy(outputVector.data(), typedBuffer, totalElements * sizeof(float));
        } break;
        case QNN_DATATYPE_UFIXED_POINT_16: {
            uint16_t* typedBuffer = static_cast<uint16_t*>(bufferData);
            auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
            
            if (quantParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                tfNToFloat(outputVector.data(), typedBuffer, 
                          quantParams.scaleOffsetEncoding.offset,
                          quantParams.scaleOffsetEncoding.scale,
                          totalElements);
            } else {
                std::cout << "Unsupported quantization encoding for output tensor " << tensorIndex << std::endl;
                castToFloat(outputVector.data(), typedBuffer, totalElements);
            }
        } break;
        case QNN_DATATYPE_UFIXED_POINT_8: {
            uint8_t* typedBuffer = static_cast<uint8_t*>(bufferData);
            auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
            
            if (quantParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                tfNToFloat(outputVector.data(), typedBuffer,
                          quantParams.scaleOffsetEncoding.offset,
                          quantParams.scaleOffsetEncoding.scale,
                          totalElements);
            } else {
                std::cout << "Unsupported quantization encoding for output tensor " << tensorIndex << std::endl;
                castToFloat(outputVector.data(), typedBuffer, totalElements);
            }
        } break;
        case QNN_DATATYPE_INT_32: {
            int32_t* typedBuffer = static_cast<int32_t*>(bufferData);
            castToFloat(outputVector.data(), typedBuffer, totalElements);
        } break;
        default:
            printf("Unsupported data type for output tensor %zu: %d\n", tensorIndex, QNN_TENSOR_GET_DATA_TYPE(&tensor));
            exit(1);
    }
}

void convertFloatDataToTensor(const Qnn_Tensor_t& tensor, void* bufferData, uint32_t totalElements, 
                            const std::vector<float>& inputVector, size_t tensorIndex) {
    switch (QNN_TENSOR_GET_DATA_TYPE(&tensor)) {
        case QNN_DATATYPE_FLOAT_32: {
            float* typedBuffer = static_cast<float*>(bufferData);
            std::memcpy(typedBuffer, inputVector.data(), totalElements * sizeof(float));
            // printf("Populated %u float elements to input tensor %zu\n", totalElements, tensorIndex);
        } break;
        case QNN_DATATYPE_UFIXED_POINT_16: {
            uint16_t* typedBuffer = static_cast<uint16_t*>(bufferData);
            auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
            
            if (quantParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                floatToTfN(typedBuffer, const_cast<float*>(inputVector.data()),
                          quantParams.scaleOffsetEncoding.offset,
                          quantParams.scaleOffsetEncoding.scale,
                          totalElements);
            } else {
                std::cout << "Unsupported quantization encoding for output tensor " << tensorIndex << std::endl;
                castFromFloat(typedBuffer, const_cast<float*>(inputVector.data()), totalElements);
            }
        } break;
        case QNN_DATATYPE_UFIXED_POINT_8: {
            uint8_t* typedBuffer = static_cast<uint8_t*>(bufferData);
            auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
            
            if (quantParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                floatToTfN(typedBuffer, const_cast<float*>(inputVector.data()),
                          quantParams.scaleOffsetEncoding.offset,
                          quantParams.scaleOffsetEncoding.scale,
                          totalElements);
            } else {
                std::cout << "Unsupported quantization encoding for output tensor " << tensorIndex << std::endl;
                castFromFloat(typedBuffer, const_cast<float*>(inputVector.data()), totalElements);
            }
        } break;
        case QNN_DATATYPE_INT_32: {
            int32_t* typedBuffer = static_cast<int32_t*>(bufferData);
            castFromFloat(typedBuffer, const_cast<float*>(inputVector.data()), totalElements);
        } break;
        default:
            printf("Unsupported data type for input tensor %zu: %d\n", tensorIndex, QNN_TENSOR_GET_DATA_TYPE(&tensor));
            exit(1);
    }
}