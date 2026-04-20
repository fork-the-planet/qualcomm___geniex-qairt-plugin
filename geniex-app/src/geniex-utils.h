

#pragma once

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
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "QnnTypes.h"
#include "QnnTypeMacros.hpp"

/**
 * @brief Utility functions for Nexa AI
 */
const size_t g_bitsPerByte = 8;
const std::map<Qnn_DataType_t, size_t> g_dataTypeToSize = {
    {QNN_DATATYPE_INT_8, 1},
    {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},
    {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},
    {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},
    {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},
    {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_FLOAT_64, 8},
    {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2},
    {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},
    {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
    {QNN_DATATYPE_BOOL_8, 1},
};

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

std::vector<std::string> split(const std::string& str);
std::vector<float> readBinaryFile(const std::string& filename);
void printTensorStats(const std::vector<std::vector<float>>& outputData);
size_t getTensorSize(const Qnn_Tensor_t* tensor);
void populateTensorSizeMap(std::unordered_map<std::string, size_t>& tensorSizeMap, const Qnn_Tensor_t* tensors,
                           uint32_t numTensors);
void convertTensorDataToFloat(const Qnn_Tensor_t& tensor, void* bufferData, uint32_t totalElements, 
                              std::vector<float>& outputVector, size_t tensorIndex);
void convertFloatDataToTensor(const Qnn_Tensor_t& tensor, void* bufferData, uint32_t totalElements, 
                              const std::vector<float>& inputVector, size_t tensorIndex);
