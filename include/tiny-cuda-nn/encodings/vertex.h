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
struct FeatureRef{
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
	T w0;
	T w1;
	T w2;
};

template <typename T>
__device__ T remap(T x, T lowIn, T highIn, T lowOut, T highOut) {
	return lowOut + (x - lowIn) * (highOut - lowOut) / (highIn - lowIn);
}

template <typename T>
__device__ T getBinCenter(int q, int z, T s) {

	// Calculate the center of the bin
	T bin_center = s * (static_cast<T>(q - z) - static_cast<T>(0.5f));

	return bin_center;
}

template <typename T>
__device__ T mapToBinCenter(T feature, int z, T s) {

	// Calculate the index of the bin that feature falls into
	int q = static_cast<int>(roundf(((float)feature / (float)s + (float)z + 0.5f)));

	// Calculate the center of the bin
	T bin_center = getBinCenter(q, z, s);

	return bin_center;
}

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
	const uint32_t n_faces,
	const uint32_t output_width,
	const uint32_t padded_output_width,
	tinyobj::index_t* indices,
	T* features,
	uint32_t* offset,
	uint32_t* meta,
	MatrixView<const float> data_in,
	MatrixView<T> data_out,
	curandState* crs,
	const uint32_t n_bins,
	const int z,
	const T s,
	const uint32_t n_quant_iterations,
	const uint32_t iter_count)
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

	const T w0 = (T)data_in(1, i);
	const T w1 = (T)data_in(2, i);
	const T w2 = (T)1.0f - w0 - w1;

	FeatureRef<T> fRef = computeFeatureRef(faceId,
		w0, w1, w2, level, offset, meta);

	const bool fQuantEnable = n_quant_iterations > 0u;
	const bool fQuantAddNoise = iter_count < n_quant_iterations;

	T f0 = features[n_features * fRef.f0 + feature_entry];
	T f1 = features[n_features * fRef.f1 + feature_entry];
	T f2 = features[n_features * fRef.f2 + feature_entry];

	if (fQuantEnable) {
		if (fQuantAddNoise) {
			// Add uniform noise to simulate quantisation
			T half_bin_width = s * static_cast<T>(0.5f);
			T offset0 = remap<T>((T)fmodf(curand_uniform(&crs[n_features * fRef.f0 + feature_entry]), 1.0f), (T)0.f, (T)1.f, -half_bin_width, half_bin_width);
			T offset1 = remap<T>((T)fmodf(curand_uniform(&crs[n_features * fRef.f1 + feature_entry]), 1.0f), (T)0.f, (T)1.f, -half_bin_width, half_bin_width);
			T offset2 = remap<T>((T)fmodf(curand_uniform(&crs[n_features * fRef.f2 + feature_entry]), 1.0f), (T)0.f, (T)1.f, -half_bin_width, half_bin_width);

			f0 += (T)offset0;
			f1 += (T)offset1;
			f2 += (T)offset2;
		}
		else {
			// Map features to bin centers
			f0 = mapToBinCenter(f0, z, s);
			f1 = mapToBinCenter(f1, z, s);
			f2 = mapToBinCenter(f2, z, s);
		}
	}

	// Interpolate feature vectors
	data_out(j, i) = fRef.w0 * f0
		+ fRef.w1 * f1
		+ fRef.w2 * f2;

	return;	
}

template <typename T>
__global__ void vertex_encoding_backward(
	const uint32_t num_instances,
	const uint32_t output_width,
	const uint32_t n_features,
	const uint32_t n_levels,
	const uint32_t n_faces,
	tinyobj::index_t* indices,
	T* features,
	uint32_t* offset,
	uint32_t* meta,
	MatrixView<const float> data_in,
	MatrixView<const T> dL_dy,
	const float* dy_dx,
	//MatrixView<float> dL_dx
	bool updateWeights
) {
	const uint32_t encoded_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t level = blockIdx.y;

	const uint32_t i = encoded_index / n_features;
	const uint32_t j = encoded_index - i * n_features + level * n_features;

	if (i >= num_instances) return;
	if (j >= output_width) return;

	const uint32_t feature_entry = j % n_features;

	if (data_in(0, i) < 0.f || !updateWeights) {
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

	gradient = dL_dy(j, i);
	atomicAdd(&features[n_features * fRef.f0 + feature_entry], -gradient * fRef.w0);
	atomicAdd(&features[n_features * fRef.f1 + feature_entry], -gradient * fRef.w1);
	atomicAdd(&features[n_features * fRef.f2 + feature_entry], -gradient * fRef.w2);

	return;
}

template <typename T>
__global__ void clamp_features(
	const uint32_t num_instances,
	T* features,
	int z,
	T s,
	uint32_t n_bins
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_instances) return;

	// Calculate half width of each bin
	T half_bin_width = s * static_cast<T>(0.5f);
	T bin0_low = getBinCenter(0, z, s) - half_bin_width;
	T binn_high = getBinCenter(n_bins - 1, z, s) + half_bin_width;

	features[idx] = clamp<T>(features[idx], bin0_low, binn_high - static_cast<T>(0.001f));
	return;
}

template <typename T>
class VertexEncoding : public Encoding<T> {
public:
	// TODO: remove n_faces parameter
	VertexEncoding(uint32_t n_features, uint32_t n_levels, uint32_t n_dims_to_encode, uint32_t n_faces, uint32_t n_vertices, uint32_t max_features_per_level, uint32_t n_quant_bins, uint32_t n_quant_iterations, std::vector<tinyobj::index_t> indices)
		: m_n_features{ n_features }, m_n_levels{ n_levels }, m_n_dims_to_encode{ n_dims_to_encode }, m_n_faces{ n_faces }, m_n_vertices{ n_vertices }, m_max_features_per_level{ max_features_per_level }, m_n_bins{ n_quant_bins }, m_n_quant_iterations{ n_quant_iterations }
	{
		printf("Vertex Encoding: n_features = %i, n_levels = %i, max_fs_per_level = 2^%i\n", m_n_features, m_n_levels, (int)std::log2(m_max_features_per_level));
		if (m_n_quant_iterations == 0)
			printf("Feature Quantisation: disabled\n");
		else
			printf("Feature Quantisation: enabled | n_quant_bins = %i, n_quant_iterations = %i\n", m_n_bins, m_n_quant_iterations);

		m_n_output_dims = n_features * m_n_levels;
		
		m_indices = GPUMemory<tinyobj::index_t>(indices.size());
		m_indices.copy_from_host(indices);

		m_n_params = computeFeatureOffset(indices, &m_offset, &m_metadata) * n_features;

		// Feature Quantisation
		m_z = (int)m_n_bins / 2;
		m_s = 2.0f / static_cast<float>(m_n_bins);

		cudaMalloc(&m_crs, sizeof(curandState) * m_n_params);
		setup_kernel<<<1, m_n_params>>>(m_n_params, m_crs);
	}

	~VertexEncoding() {
		cudaFree(m_crs);
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
		const uint32_t totalThreads = input.n() * padded_output_width();

		vertex_encoding<T><<<blocks, N_THREADS, 0, stream>>>(
			input.n(),
			m_n_features,
			m_n_levels,
			m_n_faces,
			m_n_output_dims,
			padded_output_width(),
			m_indices.data(),
			use_inference_params ? this->inference_params() : this->params(),
			m_offset.data(),
			m_metadata.data(),
			input.view(),
			output->view(),
			m_crs,
			m_n_bins,
			m_z,
			m_s,
			m_n_quant_iterations,
			m_iteration_count
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


		const bool update_weights = m_iteration_count < m_n_quant_iterations || m_n_quant_iterations <= 0u;

		vertex_encoding_backward<T> << <blocks, N_THREADS, 0, stream >> > (
			input.n(),
			m_n_output_dims,
			m_n_features,
			m_n_levels,
			m_n_faces,
			m_indices.data(),
			use_inference_params ? this->inference_params() : this->params(),
			m_offset.data(),
			m_metadata.data(),
			input.view(),
			dL_doutput.view(),
			forward.dy_dx.data(),
			//dL_dinput->view()
			update_weights
			);
		
		if (m_iteration_count < m_n_quant_iterations) {
			clamp_features<T> <<<1, n_params(), 0, stream>>> (n_params(),
				use_inference_params ? this->inference_params() : this->params(), m_z, m_s, m_n_bins);
		}

		m_iteration_count++;
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

	uint32_t m_n_dims_to_encode;					// FaceId, w0, w1
	
	uint32_t m_n_params;							// Total number of feature vectors
	uint32_t m_n_features;							// Number of entries per feature vector
	uint32_t m_n_levels;							// Number of feature hierarchy levels
	uint32_t m_max_features_per_level;	// Max number of unique feature vectors per level
							
	uint32_t m_n_faces;
	uint32_t m_n_vertices;

	curandState* m_crs;
	uint32_t m_n_bins;						// Number of quantisation bins
	int m_z;
	T m_s;

	uint32_t m_n_quant_iterations;			// Number of training iterations with simulated quantisation
	uint32_t m_iteration_count = 0u;

	GPUMemory<tinyobj::index_t> m_indices;
	GPUMemory<uint32_t> m_offset;
	GPUMemory<uint32_t> m_metadata;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;

	struct Edge {
		uint32_t toVertex;
		uint32_t subdiv;
		uint32_t offset;
		Edge(uint32_t toVertex, uint32_t subdiv, uint32_t offset) : toVertex(toVertex), subdiv(subdiv), offset(offset) {

		}
	};

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
		std::vector<uint32_t> offset_hashed(m_n_faces * face_stride);
		std::vector<uint32_t> meta_host(1 + m_n_levels * 2);		// Metadata buffer
		meta_host[0] = face_stride;

		uint32_t level_offset = 0;
		uint32_t unique_feature = 0;	// Index of next unique feature vector

		std::vector<uint32_t> n_unique_features_level(m_n_levels);


		for (size_t l = 0; l < m_n_levels; l++) { // Iterate over subdivision levels, l=0: no subdivision

			uint32_t n_subdiv = level_subdiv(l);
			meta_host[1 + 2 * l] = level_offset;
			meta_host[1 + 2 * l + 1] = n_subdiv;

			uint32_t level_first_feature = unique_feature;


			uint32_t n_features_level = (n_subdiv + 2) * (n_subdiv + 3) / 2; // #feature vectors per face on level l

			// temp storage for indices to feature vectors that are shared across faces at vertices or along edges
			std::vector<uint32_t> verts(m_n_vertices);
			for (size_t i = 0; i < m_n_vertices; i++)
				verts[i] = 0xffffffff;

			std::vector<std::vector<Edge*>> edges(m_n_vertices);
			for (size_t i = 0; i < m_n_vertices; i++)
				edges[i] = std::vector<Edge*>();

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

						bool found = false;
						for (size_t e = 0; e < edges[e0].size(); e++) {
							Edge* candidate = edges[e0][e];
							if (candidate->toVertex == e1 && candidate->subdiv == W - 1) {
								// Seen before -> reference assigned feature
								offset_host[j * face_stride + level_offset + f] = candidate->offset;
								found = true;
								break;
							}
						}
						if (!found) {
							// Not seen before -> assign new unique feature
							offset_host[j * face_stride + level_offset + f] = unique_feature;
							Edge* toE1 = new Edge(e1, W - 1, unique_feature);
							edges[e0].push_back(toE1);
							unique_feature++;
						}

						/*
						if (edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1] == 0xffffffff) { // Not seen before -> assign new unique feature
							offset_host[j * face_stride + level_offset + f] = unique_feature;
							edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1] = unique_feature;
							unique_feature++;
						}
						else { // Seen before -> reference assigned feature
							offset_host[j * face_stride + level_offset + f] = edges[e0 * m_n_vertices * n_subdiv + e1 * n_subdiv + W - 1];
						}*/
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
			n_unique_features_level[l] = unique_feature - level_first_feature;
		}

		// Hash offsets
		level_offset = 0;
		uint32_t hashed_unique_feature = 0;

		for (size_t l = 0; l < m_n_levels; l++) { // Iterate over subdivision levels, l=0: no subdivision

			uint32_t n_subdiv = level_subdiv(l);

			uint32_t n_features_level = (n_subdiv + 2) * (n_subdiv + 3) / 2; // #feature vectors per face on level l

			if (n_unique_features_level[l] <= m_max_features_per_level) {
				for (size_t j = 0; j < m_n_faces; j++) { // At each level: Iterate over faces
					for (size_t f = 0; f < n_features_level; f++) { // For each face, at each subdivision level: iterate over all features within it
						offset_hashed[j * face_stride + level_offset + f] = offset_host[j * face_stride + level_offset + f];
					}
				}
				hashed_unique_feature += n_unique_features_level[l];
			}
			else {
				for (size_t j = 0; j < m_n_faces; j++) { // At each level: Iterate over faces
					for (size_t f = 0; f < n_features_level; f++) { // For each face, at each subdivision level: iterate over all features within it
						offset_hashed[j * face_stride + level_offset + f] =
							hashOffset(offset_host[j * face_stride + level_offset + f]) + hashed_unique_feature;
					}
				}
				hashed_unique_feature += m_max_features_per_level;
			}

			level_offset += n_features_level;
		}

		*offset = GPUMemory<uint32_t>(offset_hashed.size());
		offset->copy_from_host(offset_hashed);
		*meta = GPUMemory<uint32_t>(meta_host.size());
		meta->copy_from_host(meta_host);

		printf("n_unique_features = %i\n", hashed_unique_feature * m_n_features);
		return hashed_unique_feature; // return #unique features
		
	}

	uint32_t hashOffset(uint32_t offset) {

		//const uint32_t big_prime = 2654435761u;
		//return (offset * big_prime) % m_max_features_per_level;
		return offset % m_max_features_per_level;
	}
};

template <typename T>
VertexEncoding<T>* create_vertex_encoding(uint32_t n_dims_to_encode, uint32_t n_vertices, uint32_t n_faces, std::vector<tinyobj::index_t> indices, std::vector<float> vertices, const json& encoding) {

	return new VertexEncoding<T>(encoding.value("n_features", 2u), encoding.value("n_levels", 1u), n_dims_to_encode, n_faces, n_vertices, encoding.value("max_features_level", 1u << 14), encoding.value("n_quant_bins", 16u), encoding.value("n_quant_iterations", 0u), indices);
}

}
