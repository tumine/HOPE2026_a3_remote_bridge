#include "a3_pingpong/onnx_actor.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a3_pingpong {
namespace {

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

Ort::SessionOptions MakeSessionOptions() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

std::string ExpectedJointOrderCsv() {
  const auto layout = robot_io::MakeA3Layout31();
  std::string output;
  for (std::size_t index = 0; index < layout.names.size(); ++index) {
    if (index != 0) output.push_back(',');
    output += layout.names[index];
  }
  return output;
}

void ValidateTensor(const Ort::TypeInfo& type_info,
                    std::int64_t expected_width,
                    const char* label) {
  const auto tensor = type_info.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string(label) + " must be float32");
  }
  const auto shape = tensor.GetShape();
  if (shape.size() != 2 || (shape[0] != -1 && shape[0] != 1) ||
      shape[1] != expected_width) {
    throw std::runtime_error(std::string(label) + " has an invalid shape");
  }
}

}  // namespace

struct OnnxActor::Impl {
  explicit Impl(const std::string& model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "a3_pingpong_onnx"),
        session_options(MakeSessionOptions()),
        session(env, model_path.c_str(), session_options) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error(
          "model_53000 actor must have exactly one input and one output");
    }

    auto allocated_input = session.GetInputNameAllocated(0, allocator);
    auto allocated_output = session.GetOutputNameAllocated(0, allocator);
    input_name_value = allocated_input ? allocated_input.get() : "";
    output_name_value = allocated_output ? allocated_output.get() : "";
    if (input_name_value != "observation" ||
        output_name_value != "raw_action") {
      throw std::runtime_error(
          "ONNX names must be observation -> raw_action");
    }
    ValidateTensor(session.GetInputTypeInfo(0),
                   static_cast<std::int64_t>(kPingpongObservationDim),
                   "ONNX observation input");
    ValidateTensor(session.GetOutputTypeInfo(0),
                   static_cast<std::int64_t>(kPingpongActionDim),
                   "ONNX raw_action output");

    const auto metadata = session.GetModelMetadata();
    const auto require_metadata = [&](const char* key,
                                      const std::string& expected) {
      auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
      const std::string actual = value ? value.get() : "";
      if (actual != expected) {
        throw std::runtime_error(std::string("ONNX metadata mismatch for ") +
                                 key + ": expected " + expected +
                                 ", got " + actual);
      }
    };
    const auto require_metadata_if_present = [&](const char* key,
                                                 const std::string& expected) {
      auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
      if (!value) return;
      const std::string actual = value.get();
      if (actual != expected) {
        throw std::runtime_error(std::string("ONNX metadata mismatch for ") +
                                 key + ": expected " + expected +
                                 ", got " + actual);
      }
    };
    require_metadata("contract_name", "hope_pingpong");
    require_metadata("obs_dim", "111");
    require_metadata("action_dim", "31");
    require_metadata("control_rate_hz", "50");
    require_metadata("observation_normalization", "none");
    require_metadata("base_station_semantics", "moving_lateral_target_v1");
    // The bundle also stores checkpoint identity in its manifest/provenance.
    // Accept the exporter omission while validating these fields if present.
    require_metadata_if_present(
        "policy_generation", "moving_base_model_53000_v1");
    require_metadata_if_present("checkpoint_iteration", "53000");
    require_metadata("joint_order", ExpectedJointOrderCsv());
  }

  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name_value;
  std::string output_name_value;
};

OnnxActor::OnnxActor(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

OnnxActor::~OnnxActor() = default;
OnnxActor::OnnxActor(OnnxActor&&) noexcept = default;
OnnxActor& OnnxActor::operator=(OnnxActor&&) noexcept = default;

bool OnnxActor::Run(const PingpongObservation& observation,
                    PingpongAction& raw_action,
                    std::string* reason) noexcept {
  if (!impl_) {
    SetReason(reason, "ONNX actor is not initialized");
    return false;
  }
  if (!std::all_of(observation.begin(), observation.end(),
                   [](float value) { return std::isfinite(value); })) {
    SetReason(reason, "ONNX observation contains NaN or infinity");
    return false;
  }

  try {
    const std::array<std::int64_t, 2> shape = {
        1, static_cast<std::int64_t>(kPingpongObservationDim)};
    const auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(observation.data()), observation.size(),
        shape.data(), shape.size());
    const std::array<const char*, 1> input_names = {
        impl_->input_name_value.c_str()};
    const std::array<const char*, 1> output_names = {
        impl_->output_name_value.c_str()};
    auto outputs = impl_->session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input, 1,
        output_names.data(), 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      SetReason(reason, "ONNX actor returned an invalid output");
      return false;
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementCount() != raw_action.size()) {
      SetReason(reason, "ONNX raw_action output shape or dtype changed");
      return false;
    }
    const float* values = outputs[0].GetTensorData<float>();
    std::copy_n(values, raw_action.size(), raw_action.begin());
    if (!std::all_of(raw_action.begin(), raw_action.end(),
                     [](float value) { return std::isfinite(value); })) {
      SetReason(reason, "ONNX raw_action contains NaN or infinity");
      return false;
    }
    SetReason(reason, "valid");
    return true;
  } catch (const std::exception& error) {
    SetReason(reason, std::string("ONNX inference failed: ") + error.what());
    return false;
  } catch (...) {
    SetReason(reason, "ONNX inference failed with an unknown exception");
    return false;
  }
}

const std::string& OnnxActor::input_name() const noexcept {
  return impl_->input_name_value;
}

const std::string& OnnxActor::output_name() const noexcept {
  return impl_->output_name_value;
}

}  // namespace a3_pingpong
