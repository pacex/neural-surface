/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   frequency.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the frequency encoding of NeRF [Mildenhall et al. 2020].
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include "tiny_obj_loader.h"
#include <curand.h>
#include <curand_kernel.h>


#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include <inttypes.h>

namespace tcnn {

template <typename T>
__global__ void vertex_encoding(
	const uint32_t num_elements,
	const uint32_t n_features,
	const uint32_t n_faces,
	const uint32_t n_vertices,
	const uint32_t output_width,
	tinyobj::index_t* indices,
	T* features,
	MatrixView<const float> data_in,
	MatrixView<T> data_out)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / output_width;
	const uint32_t j = encoded_index - i * output_width;

	if (j >= n_features) {
		// Padding
		data_out(j, i) = 1;
		return;
	}

	if (data_in(0, i) < 0.f) {
		data_out(j, i) = 0;
		return;
	}

	// Decode face id
	uint32_t faceId = *((uint32_t*)&data_in(0, i));
	assert(faceId < n_faces);

	// Get Barycentric Coordinates
	T w0 = data_in(1, i);
	T w1 = data_in(2, i);
	T w2 = (T)1.0f - w0 - w1;

	// Look up vertex IDs
	uint32_t f0 = indices[3 * faceId + 0].vertex_index;
	uint32_t f1 = indices[3 * faceId + 1].vertex_index;
	uint32_t f2 = indices[3 * faceId + 2].vertex_index;
	assert(f0 < n_vertices);
	assert(f1 < n_vertices);
	assert(f2 < n_vertices);

	// Interpolate feature vectors
	T interp = w0 * features[n_features * f0 + j] + w1 * features[n_features * f1 + j] + w2 * features[n_features * f2 + j];

	data_out(j, i) = interp;		
}

template <typename T>
__global__ void vertex_encoding_backward(
	const uint32_t num_elements,
	MatrixView<const T> dL_dy,
	const float* dy_dx,
	MatrixView<float> dL_dx
) {
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	// TODO: implement backward pass
}

template <typename T>
__global__ void init_features(
	const uint32_t num_elements,
	curandState* crs,
	T* features
) {
	const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= num_elements)
		return;

	// Initialze features uniformly in [-b,b]
	float b = 0.02f;
	curand_init(1337, idx, 0, &crs[idx]);
	features[idx] = (T)((fmodf(curand_uniform(&crs[idx]), 1.0f) - 0.5f) * b);
}



template <typename T>
class VertexEncoding : public Encoding<T> {
public:
	VertexEncoding(uint32_t n_features, uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, std::vector<tinyobj::index_t> indices)
		: n_features{ n_features }, m_n_dims_to_encode{ n_dims_to_encode }, n_vertices{ n_vertices }, n_faces{ n_faces }
	{
		m_n_output_dims = n_features;
		features = GPUMemory<T>(n_vertices * n_features);

		curandState* crs;
		cudaMalloc(&crs, sizeof(curandState) * n_vertices * n_features);

		linear_kernel(init_features<T>, 0, nullptr, n_vertices * n_features, crs, features.data());

		cudaFree(crs);
		this->indices = GPUMemory<tinyobj::index_t>(indices.size());
		this->indices.copy_from_host(indices);
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();

		if (!output || padded_output_width() == 0) {
			return forward;
		}

		if (prepare_input_gradients) {
			forward->dy_dx = GPUMatrix<float>{n_features, input.n(), stream};
		}

		linear_kernel(vertex_encoding<T>, 0, stream,
			input.n() * padded_output_width(),
			n_features,
			n_faces,
			n_vertices,
			padded_output_width(),
			indices.data(),
			features.data(),
			input.view(),
			output->view()
		);

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		if (!dL_dinput) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		linear_kernel(vertex_encoding_backward<T>, 0, stream,
			input.n() * m_n_dims_to_encode,
			dL_doutput.view(),
			forward.dy_dx.data(),
			dL_dinput->view()
		);
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	json hyperparams() const override {
		return {
			{"otype", "Vertex"},
			{"n_features", n_features},
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	uint32_t m_n_dims_to_encode;
	uint32_t n_features;
	uint32_t n_vertices;
	uint32_t n_faces;
	GPUMemory<T> features;

	GPUMemory<tinyobj::index_t> indices;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

template <typename T>
VertexEncoding<T>* create_vertex_encoding(uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, std::vector<tinyobj::index_t> indices, const json& encoding) {
	return new VertexEncoding<T>(encoding.value("n_features", 2u), n_dims_to_encode, n_vertices, n_faces, indices);
}

}
