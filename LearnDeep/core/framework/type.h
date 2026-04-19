#ifndef _TENSORFLOW_CORE_FRAMEWORK_TYPE_
#define _TENSORFLOW_CORE_FRAMEWORK_TYPE_

#include <cstdint>
#include <stdfloat>

typedef enum DataType {
  tf_bfloat16,
  tf_float16,
  tf_float32,
  tf_float64,
  tf_int8,
  tf_int16,
  tf_int32,
  tf_int64,
  tf_uint8,
  tf_uint16,
  tf_uint32,
  tf_uint64,
  tf_bool,
} DataType;

#endif // _TENSORFLOW_CORE_FRAMEWORK_TYPE_

template<DataType T>
struct EnumToDataType;

template<>
struct EnumToDataType<tf_bfloat16>{
  using type = std::bfloat16_t;
};

template<>
struct EnumToDataType<tf_float16>{
  using type = std::float16_t;
};

template<>
struct EnumToDataType<tf_float32>{
  using type = std::float32_t;
};
template<>
struct EnumToDataType<tf_float64>{
  using type = std::float64_t;
};

template<>
struct EnumToDataType<tf_int8>{
  using type = int8_t;
};
template<>
struct EnumToDataType<tf_int16>{
  using type = int16_t;
};

template<>
struct EnumToDataType<tf_int32>{
  using type = int32_t;
};
template<>
struct EnumToDataType<tf_int64>{
  using type = int64_t;
};

template<>
struct EnumToDataType<tf_uint8>{
  using type = uint8_t;
};

template<>
struct EnumToDataType<tf_uint16>{
  using type = uint16_t;
};

template<>
struct EnumToDataType<tf_uint32>{
  using type = uint32_t;
};

template<>
struct EnumToDataType<tf_uint64>{
  using type = uint64_t;
};