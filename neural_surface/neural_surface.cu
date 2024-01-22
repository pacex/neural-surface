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

/** @file   mlp-learning-an-image.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Sample application that uses the tiny cuda nn framework to learn a
            2D function that represents an image.
 */

#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace tcnn;
using precision_t = network_precision_t;

GPUMemory<float> load_image(const std::string& filename, int& width, int& height) {
	// width * height * RGBA
	float* out = load_stbi(&width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
__global__ void to_ldr(const uint64_t num_elements, const uint32_t n_channels, const uint32_t stride, const T* __restrict__ in, uint8_t* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint64_t pixel = i / n_channels;
	const uint32_t channel = i - pixel * n_channels;

	out[i] = (uint8_t)(powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f/2.2f) * 255.0f + 0.5f);
}

template <typename T>
void save_image(const T* image, int width, int height, int n_channels, int channel_stride, const std::string& filename) {
	GPUMemory<uint8_t> image_ldr(width * height * n_channels);
	linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride, image, image_ldr.data());

	std::vector<uint8_t> image_ldr_host(width * height * n_channels);
	CUDA_CHECK_THROW(cudaMemcpy(image_ldr_host.data(), image_ldr.data(), image_ldr.size(), cudaMemcpyDeviceToHost));

	save_stbi(image_ldr_host.data(), width, height, n_channels, filename.c_str());
}

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float* __restrict__ xs_and_ys, float* __restrict__ result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx = i * 2;

	float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx+1]);
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i) {
		result[output_idx + i] = 1;
	}
}

// Generate training data

#define N_INPUT_DIMS 3;
#define N_OUTPUT_DIMS 3;

__global__ void setup_kernel(uint32_t n_elements, curandState* state, int iter) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1337 + iter, idx, 0, &state[idx]);
}

__global__ void generate_face_positions(uint32_t n_elements, uint32_t n_faces, curandState* crs, tinyobj::index_t* indices, float* vertices, float* result) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements)
		return;

	int output_idx = idx * N_INPUT_DIMS;

	// Pick random face
	// TODO: weight by face area
	float r = fmodf(curand_uniform(crs + idx), 1.0f);
	r *= n_faces - 1.f;
	int faceId = (int)r;

	int iv1, iv2, iv3; // Indices of adjacent vertices
	iv1 = indices[3 * faceId + 0].vertex_index;
	iv2 = indices[3 * faceId + 1].vertex_index;
	iv3 = indices[3 * faceId + 2].vertex_index;
	assert(iv1 == indices[3 * faceId + 0].texcoord_index);
	assert(iv2 == indices[3 * faceId + 1].texcoord_index);
	assert(iv3 == indices[3 * faceId + 2].texcoord_index);

	result[output_idx + 0] = (float)faceId;
	//result[output_idx + 0] = (float)iv1;
	//result[output_idx + 1] = (float)iv2;
	//result[output_idx + 2] = (float)iv3;

	float alpha, beta, gamma; // Barycentric coordinates
	// TODO: is this actually uniform??
	alpha = curand_uniform(crs + idx);
	beta = curand_uniform(crs + idx) * (1.0f - alpha);
	gamma = 1.0f - alpha - beta;


	result[output_idx + 1] = alpha;
	result[output_idx + 2] = beta;
	//result[output_idx + 3] = alpha;
	//result[output_idx + 4] = beta;
	//result[output_idx + 5] = gamma;
}

__global__ void rescale_faceIds(uint32_t n_elements, uint32_t n_faces, float* training_batch, float* result) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements)
		return;

	int input_idx = idx * N_INPUT_DIMS;
	int output_idx = idx * N_INPUT_DIMS;

	float scale = 1.0f / (float)n_faces;

	result[input_idx + 0] = training_batch[output_idx + 0] * scale;

}

__global__ void generate_training_target(uint32_t n_elements, cudaTextureObject_t texture, float* training_batch, tinyobj::index_t* indices, float* texcoords, float* result) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements)
		return;

	int input_idx = idx * N_INPUT_DIMS;
	int output_idx = idx * N_OUTPUT_DIMS;

	int iv1, iv2, iv3, faceId;
	float w1, w2, w3;
	//iv1 = (int)training_batch[input_idx + 0];
	//iv2 = (int)training_batch[input_idx + 1];
	//iv3 = (int)training_batch[input_idx + 2];
	//w1 = training_batch[input_idx + 3];
	//w2 = training_batch[input_idx + 4];
	//w3 = training_batch[input_idx + 5];

	faceId = training_batch[input_idx + 0];
	iv1 = indices[3 * faceId + 0].texcoord_index;
	iv2 = indices[3 * faceId + 1].texcoord_index;
	iv3 = indices[3 * faceId + 2].texcoord_index;

	w1 = training_batch[input_idx + 1];
	w2 = training_batch[input_idx + 2];
	w3 = 1.0f - w1 - w2;

	vec2 uv1, uv2, uv3;
	uv1 = vec2(texcoords[2 * iv1 + 0], texcoords[2 * iv1 + 1]);
	uv2 = vec2(texcoords[2 * iv2 + 0], texcoords[2 * iv2 + 1]);
	uv3 = vec2(texcoords[2 * iv3 + 0], texcoords[2 * iv3 + 1]);

	vec2 uv_interp = w1 * uv1 + w2 * uv2 + w3 * uv3;

	float4 val = tex2D<float4>(texture, uv_interp.x, uv_interp.y);
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

}

int main(int argc, char* argv[]) {
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		if (argc < 2) {
			std::cout << "USAGE: " << argv[0] << " " << "path-to-image.jpg [path-to-optional-config.json]" << std::endl;
			std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
			return 0;
		}

		json config = {
			{"loss", {
				{"otype", "RelativeL2"}
			}},
			{"optimizer", {
				{"otype", "Adam"},
				// {"otype", "Shampoo"},
				{"learning_rate", 1e-2},
				{"beta1", 0.9f},
				{"beta2", 0.99f},
				{"l2_reg", 0.0f},
				// The following parameters are only used when the optimizer is "Shampoo".
				{"beta3", 0.9f},
				{"beta_shampoo", 0.0f},
				{"identity", 0.0001f},
				{"cg_on_momentum", false},
				{"frobenius_normalization", true},
			}},
			{"encoding", {
				{"otype", "HashGrid"},
				{"n_levels", 16},
				{"n_features_per_level", 2},
				{"log2_hashmap_size", 15},
				{"base_resolution", 16},
				{"per_level_scale", 1.5},
			}},
			{"network", {
				{"otype", "FullyFusedMLP"},
				// {"otype", "CutlassMLP"},
				{"n_neurons", 64},
				{"n_hidden_layers", 4},
				{"activation", "ReLU"},
				{"output_activation", "None"},
			}},
		};

		if (argc >= 3) {
			std::cout << "Loading custom json config '" << argv[2] << "'." << std::endl;
			std::ifstream f{argv[2]};
			config = json::parse(f, nullptr, true, /*skip_comments=*/true);
		}

		/* ======================
		*  === LOAD .OBJ FILE ===
		*  ======================
		*/

		std::string path = "data/objects/wheatley.obj";

		std::cout << "Loading " << path << "..." << std::flush;
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		std::string warn;
		// Expect '.mtl' file in the same directory and triangulate meshes
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str());
		if (!err.empty())
		{ // `err` may contain warning message.
			std::cerr << err << std::endl;
		}
		if (!warn.empty())
		{
			std::cerr << warn << std::endl;
		}
		if (!ret)
		{
			std::cerr << "Loading .obj file failed." << std::endl;
			exit(1);
		}

		int shapeIndex = 1;
		int n_vertices = attrib.vertices.size() / 3;
		assert(n_vertices == attrib.texcoords.size() / 2);

		int n_indices = shapes[shapeIndex].mesh.indices.size();
		int n_faces = n_indices / 3;

		// write vertices and indices to GPU memory
		GPUMemory<float> vertices(n_vertices * 3);
		vertices.copy_from_host(attrib.vertices);
		GPUMemory<float> texcoords(n_vertices * 2);
		texcoords.copy_from_host(attrib.texcoords);

		GPUMemory<tinyobj::index_t> indices(n_indices);
		indices.copy_from_host(shapes[shapeIndex].mesh.indices);

		/* ==============================
		*  === LOAD REFERENCE TEXTURE ===
		*  ==============================
		*/

		int width, height;
		GPUMemory<float> image = load_image(argv[1], width, height);

		// Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = image.data();
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		resDesc.res.pitch2D.width = width;
		resDesc.res.pitch2D.height = height;
		resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;

		cudaTextureObject_t texture;
		CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));


		// Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
		//             function will be eventually possible.

		//int sampling_width = width;
		//int sampling_height = height;

		// Uncomment to fix the resolution of the training task independent of input image
		// int sampling_width = 1024;
		// int sampling_height = 1024;

		//uint32_t n_coords = sampling_width * sampling_height;
		//uint32_t n_coords_padded = next_multiple(n_coords, BATCH_SIZE_GRANULARITY);

		//GPUMemory<float> sampled_image(n_coords * 3);
		//GPUMemory<float> xs_and_ys(n_coords_padded * 2);

		/*
		std::vector<float> host_xs_and_ys(n_coords * 2);
		for (int y = 0; y < sampling_height; ++y) {
			for (int x = 0; x < sampling_width; ++x) {
				int idx = (y * sampling_width + x) * 2;
				host_xs_and_ys[idx+0] = (float)(x + 0.5) / (float)sampling_width;
				host_xs_and_ys[idx+1] = (float)(y + 0.5) / (float)sampling_height;
			}
		}

		xs_and_ys.copy_from_host(host_xs_and_ys.data());

		linear_kernel(eval_image<3>, 0, nullptr, n_coords, texture, xs_and_ys.data(), sampled_image.data());

		save_image(sampled_image.data(), sampling_width, sampling_height, 3, 3, "reference.jpg");
		*/

		/* =======================
		*  === TRAIN THE MODEL ===
		*  =======================
		*/

		// Various constants for the network and optimization
		const uint32_t batch_size = 1 << 18;
		const uint32_t n_training_steps = argc >= 4 ? atoi(argv[3]) : 10000000;
		const uint32_t n_input_dims = N_INPUT_DIMS; // (v1, v2, v3, alpha, beta, gamma)
		const uint32_t n_output_dims = N_OUTPUT_DIMS; // RGB color

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		default_rng_t rng{1337};
		// Auxiliary matrices for training

		GPUMatrix<float> training_batch_raw(n_input_dims, batch_size);
		GPUMatrix<float> training_batch(n_input_dims, batch_size);
		GPUMatrix<float> training_target(n_output_dims, batch_size);

		// Auxiliary matrices for evaluation
		//GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
		//GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

		json encoding_opts = config.value("encoding", json::object());
		json loss_opts = config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts = config.value("network", json::object());

		std::shared_ptr<Loss<precision_t>> loss{ create_loss<precision_t>(loss_opts) };
		std::shared_ptr<Optimizer<precision_t>> optimizer{ create_optimizer<precision_t>(optimizer_opts) };
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

		auto model = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

		uint32_t interval = 10;

		for (uint32_t i = 0; i < n_training_steps; ++i) {
			bool print_loss = i % interval == 0;
			bool visualize_learned_func = argc < 5 && i % interval == 0;

			/* ===============================
			*  === GENERATE TRAINING BATCH ===
			*  ===============================
			*/
			{
				// RNG
				curandState* crs;
				cudaMalloc(&crs, sizeof(curandState) * batch_size);
				linear_kernel(setup_kernel, 0, training_stream, batch_size, crs, i);

				// Generate Surface Points - training input
				linear_kernel(generate_face_positions, 0, training_stream, batch_size, n_faces, crs, indices.data(), vertices.data(), training_batch_raw.data());
				linear_kernel(rescale_faceIds, 0, training_stream, batch_size, n_faces, training_batch_raw.data(), training_batch.data());

				// Sample reference texture at surface points - training output
				linear_kernel(generate_training_target, 0, training_stream, batch_size, texture, training_batch_raw.data(), indices.data(), texcoords.data(), training_target.data());

				cudaFree(crs);

				// Debug
				//std::vector<float> in = training_batch_obj.to_cpu_vector();
				//std::vector<float> out = training_target_obj.to_cpu_vector();
			}

			/* =========================
			*  === RUN TRAINING STEP ===
			*  =========================
			*/

			{
				//auto ctx = trainer->training_step(training_stream, training_batch, training_target);

				auto ctx_obj = model->training_step(training_stream, training_batch, training_target);

				if (i % std::min(interval, (uint32_t)100) == 0) {
					tmp_loss += model->loss(training_stream, *ctx_obj);
					++tmp_loss_counter;
				}
			}




			// Debug outputs
			{
				
				if (print_loss) {
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					std::cout << "Step#" << i << ": " << "loss=" << tmp_loss/(float)tmp_loss_counter << " time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

					tmp_loss = 0;
					tmp_loss_counter = 0;
				}

				if (visualize_learned_func) {
					//network->inference(inference_stream, inference_batch, prediction);
					//auto filename = fmt::format("{}.jpg", i);
					//std::cout << "Writing '" << filename << "'... ";
					//save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, filename);
					//std::cout << "done." << std::endl;
				}

				// Don't count visualizing as part of timing
				// (assumes visualize_learned_pdf is only true when print_loss is true)
				if (print_loss) {
					begin = std::chrono::steady_clock::now();
				}
			}

			if (print_loss && i > 0 && interval < 1000) {
				interval *= 10;
			}
		}

		// Dump final image if a name was specified
		if (argc >= 5) {
			//network->inference(inference_stream, inference_batch, prediction);
			//save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, argv[4]);
		}

		free_all_gpu_memory_arenas();

		// If only the memory arenas pertaining to a single stream are to be freed, use
		//free_gpu_memory_arena(stream);
	} catch (const std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}

