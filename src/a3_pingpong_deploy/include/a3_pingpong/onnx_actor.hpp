#pragma once

#include "a3_pingpong/pingpong_observation_builder.hpp"

#include <memory>
#include <string>

namespace a3_pingpong {

// CPU ONNX Runtime wrapper for the frozen model_48000 actor contract:
// observation float32[1,111] -> raw_action float32[1,31].
class OnnxActor {
 public:
  explicit OnnxActor(const std::string& model_path);
  ~OnnxActor();

  OnnxActor(const OnnxActor&) = delete;
  OnnxActor& operator=(const OnnxActor&) = delete;
  OnnxActor(OnnxActor&&) noexcept;
  OnnxActor& operator=(OnnxActor&&) noexcept;

  bool Run(const PingpongObservation& observation,
           PingpongAction& raw_action,
           std::string* reason = nullptr) noexcept;

  const std::string& input_name() const noexcept;
  const std::string& output_name() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace a3_pingpong
