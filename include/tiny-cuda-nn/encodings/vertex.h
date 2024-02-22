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
struct FeatureRef{
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
	T w0;
	T w1;
	T w2;
};

template <typename T>
__device__ FeatureRef<T> computeFeatureRef(uint32_t faceId, T w0, T w1, T w2, uint32_t level, uint32_t* offset, uint32_t* meta) {

	uint32_t face_stride = meta[0];
	uint32_t level_offset = meta[1 + 2 * level];
	uint32_t n_subdiv = meta[1 + 2 * level + 1];

	// Compute 'barycentric' identifiers to feature vectors at adjacent vertices
	uint32_t w0_low = static_cast<uint32_t>(std::floorf((T)(n_subdiv + 1) * w0));
	uint32_t w0_high = static_cast<uint32_t>(std::ceilf((T)(n_subdiv + 1) * w0));
	uint32_t w1_low = static_cast<uint32_t>(std::floorf((T)(n_subdiv + 1) * w1));
	uint32_t w1_high = static_cast<uint32_t>(std::ceilf((T)(n_subdiv + 1) * w1));
	uint32_t w2_low = static_cast<uint32_t>(std::floorf((T)(n_subdiv + 1) * w2));
	uint32_t w2_high = static_cast<uint32_t>(std::ceilf((T)(n_subdiv + 1) * w2));

	// adjacent 'sub-vertices' depend on if sub triangle is edge aligned to original triangle
	uint32_t vertices_adj_local_aligned_unaligned[2][9] = { { w0_low, w1_high, w2_high, w0_high, w1_low, w2_high, w0_high, w1_high, w2_low },
															{ w0_high, w1_low, w2_low, w0_low, w1_high, w2_low, w0_low, w1_low, w2_high} };

	bool edgeAligned = (w0_low + w1_low + w2_low) % 2 == n_subdiv % 2;
	uint32_t* vertices_adj_local;
	vertices_adj_local = vertices_adj_local_aligned_unaligned[edgeAligned];

	/* For all three adjacent feature vectors:
	*	- compute their position in the offset buffer from 'barycentric' identifiers
	*	- look up feature vector offset from precomputed buffer
	*/
	FeatureRef<T> result;
	uint32_t invW0, W1;

	invW0 = (n_subdiv + 1) - vertices_adj_local[0];
	W1 = vertices_adj_local[1];
	result.f0 = offset[faceId * face_stride + level_offset + (invW0 * (invW0 + 1) / 2 + W1)];
	invW0 = (n_subdiv + 1) - vertices_adj_local[3];
	W1 = vertices_adj_local[4];
	result.f1 = offset[faceId * face_stride + level_offset + (invW0 * (invW0 + 1) / 2 + W1)];
	invW0 = (n_subdiv + 1) - vertices_adj_local[6];
	W1 = vertices_adj_local[7];
	result.f2 = offset[faceId * face_stride + level_offset + (invW0 * (invW0 + 1) / 2 + W1)];

	// Compute local barycentric coordinates on sub triangle
	T w0_local = (T)std::fmodf((T)(n_subdiv + 1) * w0, 1.0f);
	T w1_local = (T)std::fmodf((T)(n_subdiv + 1) * w1, 1.0f);
	T w2_local = (T)std::fmodf((T)(n_subdiv + 1) * w2, 1.0f);

	result.w0 = (T)edgeAligned * w0_local + (T)!edgeAligned * ((T)1.0f - w0_local);
	result.w1 = (T)edgeAligned * w1_local + (T)!edgeAligned * ((T)1.0f - w1_local);
	result.w2 = (T)edgeAligned * w2_local + (T)!edgeAligned * ((T)1.0f - w2_local);

	return result;
}


template <typename T>
__global__ void vertex_encoding(
	const uint32_t num_instances,
	const uint32_t n_features,
	const uint32_t n_levels,
	const uint32_t n_frequencies,
	const uint32_t n_faces,
	const uint32_t n_vertices,
	const OutputConstruction constr,
	const uint32_t output_width,
	const uint32_t padded_output_width,
	tinyobj::index_t* indices,
	float* vertices,
	T* features,
	uint32_t* offset,
	uint32_t* meta,
	MatrixView<const float> data_in,
	MatrixView<T> data_out)
{
	const uint32_t encoded_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t level = blockIdx.y; // number of _added_ features per edge

	const uint32_t i = encoded_index / n_features;
	const uint32_t j = encoded_index - i * n_features + level * n_features;

	if (i >= num_instances) return;
	if (j >= padded_output_width) return;

	if (level >= n_levels) {
		// Padding
		data_out(j, i) = 1;
		return;
	}

	if (data_in(0, i) < 0.f) {
		data_out(j, i) = 0;
		return;
	}

	const uint32_t feature_entry = j % n_features;

	// Decode face id
	uint32_t faceId = *((uint32_t*)&data_in(0, i));

	T w0 = (T)data_in(1, i);
	T w1 = (T)data_in(2, i);
	T w2 = (T)1.0f - w0 - w1;

	FeatureRef<T> fRef = computeFeatureRef(faceId,
		w0, w1, w2, level, offset, meta);

	

	switch (constr) {
	case lin_interp:
	
		// Interpolate feature vectors
		data_out(j, i) = fRef.w0 * features[n_features * fRef.f0 + feature_entry]
			+ fRef.w1 * features[n_features * fRef.f1 + feature_entry]
			+ fRef.w2 * features[n_features * fRef.f2 + feature_entry];

		break;

	case concat:
		uint32_t vertex = j / n_features;

		if (vertex >= 3) {
			// Positional Component
			const T v = fRef.w0 * (T)vertices[3 * fRef.f0 + (j % 3)] + fRef.w1 * (T)vertices[3 * fRef.f1 + (j % 3)] + fRef.w2 * (T)vertices[3 * fRef.f2 + (j % 3)];
			const uint32_t log2_frequency = (j / 2) % n_frequencies;

			const float phase_shift = (j % 2) * (PI / 2);

			const float x = scalbnf(v, log2_frequency);
			const float input = x * PI + phase_shift;

			data_out(j, i) = (T)__sinf(input);
		}
		else {
			// Feature Component
			uint32_t f = indices[3 * faceId + vertex].vertex_index;
			data_out(j, i) = features[n_features * f + (j % n_features)];
		}

		break;
	}

	return;	
}

template <typename T>
__global__ void vertex_encoding_backward(
	const uint32_t num_instances,
	const uint32_t output_width,
	const uint32_t n_features,
	const uint32_t n_levels,
	const uint32_t n_faces,
	const uint32_t n_vertices,
	const OutputConstruction constr,
	tinyobj::index_t* indices,
	T* features,
	uint32_t* offset,
	uint32_t* meta,
	MatrixView<const float> data_in,
	MatrixView<const T> dL_dy,
	const float* dy_dx
	//MatrixView<float> dL_dx
) {
	const uint32_t encoded_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t level = blockIdx.y;

	const uint32_t i = encoded_index / n_features;
	const uint32_t j = encoded_index - i * n_features + level * n_features;

	if (i >= num_instances) return;
	if (j >= output_width) return;

	const uint32_t feature_entry = j % n_features;

	if (data_in(0, i) < 0.f) {
		return;
	}

	// Decode face id
	uint32_t faceId = *((uint32_t*)&data_in(0, i));

	// Get Barycentric Coordinates
	T w0 = data_in(1, i);
	T w1 = data_in(2, i);
	T w2 = (T)1.0f - w0 - w1;

	T gradient;

	FeatureRef<T> fRef = computeFeatureRef(faceId,
		w0, w1, w2, level, offset, meta);

	switch (constr) {

		/*
			LINEAR INTERPOLATION
		*/
	case lin_interp:
		gradient = dL_dy(j, i);
		atomicAdd(&features[n_features * fRef.f0 + feature_entry], -gradient * fRef.w0);
		atomicAdd(&features[n_features * fRef.f1 + feature_entry], -gradient * fRef.w1);
		atomicAdd(&features[n_features * fRef.f2 + feature_entry], -gradient * fRef.w2);
		break;

		/*
			CONCATENATION
		*/
	case concat:
		if (j >= n_features * 3)
			break;
		uint32_t vertex = j / n_features;
		uint32_t f = indices[3 * faceId + vertex].vertex_index;

		gradient = dL_dy(j, i);
		atomicAdd(&features[n_features * f + (j % n_features)], -gradient);
		break;
	}

	return;
}

template <typename T>
class VertexEncoding : public Encoding<T> {
public:
	// TODO: remove n_faces parameter
	VertexEncoding(uint32_t n_features, uint32_t n_levels, uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, OutputConstruction output_construction, std::vector<tinyobj::index_t> indices, std::vector<float> vertices)
		: m_n_features{ n_features }, m_n_levels{ n_levels }, m_n_dims_to_encode { n_dims_to_encode}, m_n_vertices{ n_vertices }, m_n_faces{ n_faces }, m_output_construction{ output_construction }
	{
		printf("Vertex Encoding with %i feature vector entries and %i feature hierarchy levels.\n", m_n_features, m_n_levels);

		// TODO: implement hashing
		m_max_features_per_level = 16 * n_vertices;

		switch (m_output_construction) {
		case lin_interp:
			m_n_output_dims = n_features * m_n_levels;
			break;
		case concat:
			m_n_output_dims = 3 * n_features + m_n_frequencies;
			break;
		default:
			m_n_output_dims = n_features;
		}

		m_indices = GPUMemory<tinyobj::index_t>(indices.size());
		m_indices.copy_from_host(indices);

		m_vertices = GPUMemory<float>(vertices.size());
		m_vertices.copy_from_host(vertices);

		m_n_params = computeFeatureOffset(indices, &m_offset, &m_metadata) * n_features;
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

		const uint32_t N_THREADS = 512;
		const uint32_t N_ELEMENTS = input.n() * m_n_features;
		const dim3 blocks = { div_round_up(N_ELEMENTS, N_THREADS), div_round_up(padded_output_width(), m_n_features), 1};

		
		vertex_encoding<T><<<blocks, N_THREADS, 0, stream>>>(
			input.n(),
			m_n_features,
			m_n_levels,
			m_n_frequencies,
			m_n_faces,
			m_n_vertices,
			m_output_construction,
			m_n_output_dims,
			padded_output_width(),
			m_indices.data(),
			m_vertices.data(),
			use_inference_params ? this->inference_params() : this->params(),
			m_offset.data(),
			m_metadata.data(),
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

		const uint32_t N_THREADS = 512;
		const uint32_t N_ELEMENTS = input.n() * m_n_features;
		const dim3 blocks = { div_round_up(N_ELEMENTS, N_THREADS), m_n_levels, 1 };


		vertex_encoding_backward<T> <<<blocks, N_THREADS, 0, stream>>> (
			input.n(),
			m_n_output_dims,
			m_n_features,
			m_n_levels,
			m_n_faces,
			m_n_vertices,
			m_output_construction,
			m_indices.data(),
			use_inference_params ? this->inference_params() : this->params(),
			m_offset.data(),
			m_metadata.data(),
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

	OutputConstruction m_output_construction;		// Interpolate or concatenate feature vectors

	uint32_t m_n_dims_to_encode;					// FaceId, w0, w1
	
	uint32_t m_n_params;							// Total number of feature vectors
	uint32_t m_n_features;							// Number of entries per feature vector
	uint32_t m_n_levels;							// Number of feature hierarchy levels
	uint32_t m_max_features_per_level;				// Max number of unique feature vectors per level

	uint32_t m_n_vertices;							
	uint32_t m_n_faces;

	uint32_t m_n_frequencies = 12;					// Number of frequencies to encode position in (only applied in concat mode)

	GPUMemory<tinyobj::index_t> m_indices;
	GPUMemory<float> m_vertices;
	GPUMemory<uint32_t> m_offset;
	GPUMemory<uint32_t> m_metadata;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;

	uint32_t computeFeatureOffset(std::vector<tinyobj::index_t> indices, GPUMemory<uint32_t>* offset, GPUMemory<uint32_t>* meta) {

		/* Precompute offset of feature vectors in memory
		*
		* offset layout:
		* |0		|face_stride	|2*face_stride	... |n_faces*face_stride	|
		*			/				\
		*			|f_00 f_01 f_02|f_10 f_11 ... f_1n_features_level|...
		*
		* (f_ij * n_features) serves as index to feature vector in params
		*
		* metadata layout:
		* |face_stride level_offset_0 n_subdiv_0 level_offset_1 n_subdiv_1 ... level_offset_n-1 n_subdiv_n-1|
		*
		* returns number of unique feature vectors
		*/

		// TODO: hashing

		// TODO: make this hyperparam
		auto level_subdiv = [](uint32_t l) {
			return std::pow(2, l) - 1;
		};

	
		uint32_t face_stride = 0; // How many feature offsets need to be stored per face
		for (size_t l = 0; l < m_n_levels; l++) {
			uint32_t n = level_subdiv(l);
			face_stride += (n + 2) * (n + 3) / 2;
		}

		std::vector<uint32_t> offset_host(m_n_faces * face_stride);	// Offset buffer
		std::vector<uint32_t> meta_host(1 + m_n_levels * 2);		// Metadata buffer
		meta_host[0] = face_stride;

		uint32_t level_offset = 0;
		uint32_t unique_feature = 0;	// Index of next unique feature vector


		for (size_t l = 0; l < m_n_levels; l++) { // Iterate over subdivision levels, l=0: no subdivision

			uint32_t n_subdiv = level_subdiv(l);
			meta_host[1 + 2 * l] = level_offset;
			meta_host[1 + 2 * l + 1] = n_subdiv;


			uint32_t n_features_level = (n_subdiv + 2) * (n_subdiv + 3) / 2; // #feature vectors per face on level l

			// temp storage for indices to feature vectors that are shared across faces at vertices or along edges
			std::vector<uint32_t> verts(m_n_vertices);
			for (size_t i = 0; i < m_n_vertices; i++)
				verts[i] = 0xffffffff;

			std::vector<uint32_t> edges(m_n_vertices * m_n_vertices * n_subdiv);
			for (size_t i = 0; i < m_n_vertices * m_n_vertices * n_subdiv; i++)
				edges[i] = 0xffffffff;

			for (size_t j = 0; j < m_n_faces; j++) { // At each level: Iterate over faces

				uint32_t v0 = indices[3 * j + 0].vertex_index;
				uint32_t v1 = indices[3 * j + 1].vertex_index;
				uint32_t v2 = indices[3 * j + 2].vertex_index;

				for (size_t f = 0; f < n_features_level; f++) { // For each face, at each subdivision level: iterate over all features within it

					// Compute integer 'barycentric' feature identifiers
					uint32_t invW0 = std::floorf(-0.5f + std::sqrtf(0.25f + 2 * f));
					uint32_t W0 = (n_subdiv + 1) - invW0;
					uint32_t W1 = f - (invW0 * (invW0 + 1) / 2);
					uint32_t W2 = (n_subdiv + 1) - W0 - W1;
					assert(W0 + W1 + W2 == (n_subdiv + 1));

					// Shared feature at vertex
					if (W0 == (n_subdiv + 1) || W1 == (n_subdiv + 1) || W2 == (n_subdiv + 1)) {
						uint32_t v = W0 > 0 ? v0 : (W1 > 0 ? v1 : v2);
						if (verts[v] == 0xffffffff) { // Not seen before -> assign new unique feature
							offset_host[j * face_stride + level_offset + f] = unique_feature;
							verts[v] = unique_feature;
							unique_feature++;
						}
						else { // Seen before -> reference assigned feature
							offset_host[j * face_stride + level_offset + f] = verts[v];
						}
					}

					// Shared feature at edge
					else if (W0 == 0 || W1 == 0 || W2 == 0) {
						uint32_t e0 = W0 == 0 ? std::max(v1, v2) : (W1 == 0 ? std::max(v0, v2) : std::max(v0, v1));
						uint32_t e1 = W0 == 0 ? std::min(v1, v2) : (W1 == 0 ? std::min(v0, v2) : std::min(v0, v1));
						uint32_t W = e0 == v0 ? W0 : (e0 == v1 ? W1 : W2);

						if (edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1] == 0xffffffff) { // Not seen before -> assign new unique feature
							offset_host[j * face_stride + level_offset + f] = unique_feature;
							edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1] = unique_feature;
							unique_feature++;
						}
						else { // Seen before -> reference assigned feature
							offset_host[j * face_stride + level_offset + f] = edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1];
						}
					}

					// Non-shared feature -> new unique feature
					else {
						offset_host[j * face_stride + level_offset + f] = unique_feature;
						unique_feature++;
					}

					assert(level_offset + f < face_stride);
				}
			}

			level_offset += n_features_level;

			// Fix memory alignment
			uint32_t alignment = 1u << 3;	// not 100% sure what the memory alignment requirements are but this seems to work
			unique_feature += (alignment - (unique_feature % alignment)) % alignment;
		}

		*offset = GPUMemory<uint32_t>(offset_host.size());
		offset->copy_from_host(offset_host);
		*meta = GPUMemory<uint32_t>(meta_host.size());
		meta->copy_from_host(meta_host);

		return unique_feature; // return #unique features
	}
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

	return new VertexEncoding<T>(encoding.value("n_features", 2u), encoding.value("n_levels", 1u), n_dims_to_encode, n_vertices, n_faces, output_construction, indices, vertices);
}

}
