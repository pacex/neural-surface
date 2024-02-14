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

enum OutputConstruction { lin_interp, concat };

template <typename T>
__global__ void vertex_encoding(
	const uint32_t num_elements,
	const uint32_t n_features,
	const uint32_t n_frequencies,
	const uint32_t n_faces,
	const uint32_t n_vertices,
	const OutputConstruction constr,
	const uint32_t output_width,
	const uint32_t padded_output_width,
	tinyobj::index_t* indices,
	float* vertices,
	T* features,
	MatrixView<const float> data_in,
	MatrixView<T> data_out)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / padded_output_width;
	const uint32_t j = encoded_index - i * padded_output_width;

	if (j >= output_width) {
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

	switch (constr) {
	case lin_interp:
	
		// Interpolate feature vectors
		data_out(j, i) = w0 * features[n_features * f0 + j] + w1 * features[n_features * f1 + j] + w2 * features[n_features * f2 + j];

		break;

	case concat:
		uint32_t vertex = j / n_features;

		if (vertex >= 1) {
			// Positional Component
			const T v = w0 * (T)vertices[3 * f0 + (j % 3)] + w1 * (T)vertices[3 * f1 + (j % 3)] + w2 * (T)vertices[3 * f2 + (j % 3)];
			const uint32_t log2_frequency = (j / 2) % n_frequencies;

			const float phase_shift = (j % 2) * (PI / 2);

			const float x = scalbnf(v, log2_frequency);
			const float input = x * PI + phase_shift;

			data_out(j, i) = (T)__sinf(input);
		}
		else {
			// Feature Component
			
			// Interpolate feature vectors
			data_out(j, i) = w0 * features[n_features * f0 + j] + w1 * features[n_features * f1 + j] + w2 * features[n_features * f2 + j];
		}

		break;
	}

	return;	
}

template <typename T>
__global__ void vertex_encoding_backward(
	const uint32_t num_elements,
	const uint32_t output_width,
	const uint32_t n_features,
	const uint32_t n_faces,
	const uint32_t n_vertices,
	const OutputConstruction constr,
	tinyobj::index_t* indices,
	T* features,
	MatrixView<const float> data_in,
	MatrixView<const T> dL_dy,
	const float* dy_dx
	//MatrixView<float> dL_dx
) {
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	// TODO: implement backward pass
	const uint32_t i = encoded_index / output_width;
	const uint32_t j = encoded_index - i * output_width;

	if (data_in(0, i) < 0.f) {
		return;
	}

	// Decode face id
	uint32_t faceId = *((uint32_t*)&data_in(0, i));
	assert(faceId < n_faces);

	// Get Barycentric Coordinates
	T w0 = data_in(1, i);
	T w1 = data_in(2, i);
	T w2 = (T)1.0f - w0 - w1;

	T gradient;

	// Look up vertex IDs
	uint32_t f0 = indices[3 * faceId + 0].vertex_index;
	uint32_t f1 = indices[3 * faceId + 1].vertex_index;
	uint32_t f2 = indices[3 * faceId + 2].vertex_index;
	assert(f0 < n_vertices);
	assert(f1 < n_vertices);
	assert(f2 < n_vertices);

	switch (constr) {

		/*
			LINEAR INTERPOLATION
		*/
	case lin_interp:
		

		gradient = dL_dy(j, i);
		atomicAdd(&features[n_features * f0 + j], -gradient * w0);
		atomicAdd(&features[n_features * f1 + j], -gradient * w1);
		atomicAdd(&features[n_features * f2 + j], -gradient * w2);
		break;

		/*
			CONCATENATION
		*/
	case concat:
		if (j >= n_features)
			break;

		gradient = dL_dy(j, i);
		atomicAdd(&features[n_features * f0 + j], -gradient * w0);
		atomicAdd(&features[n_features * f1 + j], -gradient * w1);
		atomicAdd(&features[n_features * f2 + j], -gradient * w2);
		break;
	}

	return;
}

template <typename T>
class VertexEncoding : public Encoding<T> {
public:
	// TODO: remove n_faces parameter
	VertexEncoding(uint32_t n_features, uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, OutputConstruction output_construction, std::vector<tinyobj::index_t> indices, std::vector<float> vertices)
		: m_n_features{ n_features }, m_n_dims_to_encode{ n_dims_to_encode }, m_n_vertices{ n_vertices }, m_n_faces{ n_faces }, m_output_construction{ output_construction }
	{
		switch (m_output_construction) {
		case lin_interp:
			m_n_output_dims = n_features;
			break;
		case concat:
			m_n_output_dims = n_features + m_n_frequencies;
			break;
		default:
			m_n_output_dims = n_features;
		}

		m_n_params = n_features * n_vertices;

		m_indices = GPUMemory<tinyobj::index_t>(indices.size());
		m_indices.copy_from_host(indices);

		m_vertices = GPUMemory<float>(vertices.size());
		m_vertices.copy_from_host(vertices);
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
			forward->dy_dx = GPUMatrix<float>{m_n_features, input.n(), stream};
		}

		linear_kernel(vertex_encoding<T>, 0, stream,
			input.n() * padded_output_width(),
			m_n_features,
			m_n_frequencies,
			m_n_faces,
			m_n_vertices,
			m_output_construction,
			m_n_output_dims,
			padded_output_width(),
			m_indices.data(),
			m_vertices.data(),
			use_inference_params ? this->inference_params() : this->params(),
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
		const uint32_t num_elements = input.n();
		if ((!dL_dinput && param_gradients_mode == GradientMode::Ignore) || num_elements == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		linear_kernel(vertex_encoding_backward<T>, 0, stream,
			input.n() * m_n_output_dims,
			m_n_output_dims,
			m_n_features,
			m_n_faces,
			m_n_vertices,
			m_output_construction,
			m_indices.data(),
			use_inference_params ? this->inference_params() : this->params(),
			input.view(),
			dL_doutput.view(),
			forward.dy_dx.data()
			//dL_dinput->view()
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

	void set_params_impl(T* params, T* inference_params, T* gradients) override { }

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize the hashgrid from the GPU, because the number of parameters can be quite large.
		generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
	}

	size_t n_params() const override {
		return m_n_params;
	}

	json hyperparams() const override {
		return {
			{"otype", "Vertex"},
			{"n_features", m_n_features},
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	OutputConstruction m_output_construction;

	uint32_t m_n_dims_to_encode;
	uint32_t m_n_frequencies = 24;
	uint32_t m_n_params;
	uint32_t m_n_features;
	uint32_t m_n_vertices;
	uint32_t m_n_faces;

	GPUMemory<tinyobj::index_t> m_indices;
	GPUMemory<float> m_vertices;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

template <typename T>
VertexEncoding<T>* create_vertex_encoding(uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, std::vector<tinyobj::index_t> indices, std::vector<float> vertices, const json& encoding) {
	const std::string output_construction_str = encoding.value("output_construction", "lin_interp");
	OutputConstruction output_construction;

	if (equals_case_insensitive(output_construction_str, "lin_interp"))
		output_construction = lin_interp;
	else if (equals_case_insensitive(output_construction_str, "concat"))
		output_construction = concat;
	else
		output_construction = lin_interp;

	return new VertexEncoding<T>(encoding.value("n_features", 2u), n_dims_to_encode, n_vertices, n_faces, output_construction, indices, vertices);
}

}
