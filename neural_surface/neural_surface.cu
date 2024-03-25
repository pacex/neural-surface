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
#include <sstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <tiny-cuda-nn/encodings/vertex.h>
#include <tiny-cuda-nn/encoding.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace tcnn;
using precision_t = network_precision_t;

struct Texture {
	bool valid;
	cudaTextureObject_t texture;
	int width;
	int height;
	cudaResourceDesc resDesc;
	cudaTextureDesc texDesc;
	GPUMemory<float> image;
};

struct Material {
	float diffuse[3];
	Texture map_Kd;
	Texture map_Bump;
};

struct EvalResult {
	float MSE;
	uint32_t n_floats;
};


#define N_INPUT_DIMS 3
#define N_OUTPUT_DIMS 6

GPUMemory<float> load_image(const std::string& filename, int& width, int& height) {
	// width * height * RGBA
	float* out = load_stbi(&width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
__global__ void to_ldr(const uint64_t num_elements, const uint32_t n_channels, const uint32_t stride, const uint32_t offset, const T* __restrict__ in, uint8_t* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint64_t pixel = i / n_channels;
	const uint32_t channel = i - pixel * n_channels + offset;

	out[i] = (uint8_t)(powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f/2.2f) * 255.0f + 0.5f);
}

template <typename T>
void save_image(const T* image, int width, int height, int n_channels, int channel_stride, int channel_offset, const std::string& filename) {

	std::cout << filename << "... " << std::endl;
	GPUMemory<uint8_t> image_ldr(width * height * n_channels);
	linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride, channel_offset, image, image_ldr.data());

	std::vector<uint8_t> image_ldr_host(width * height * n_channels);
	CUDA_CHECK_THROW(cudaMemcpy(image_ldr_host.data(), image_ldr.data(), image_ldr.size(), cudaMemcpyDeviceToHost));

	save_stbi(image_ldr_host.data(), width, height, n_channels, filename.c_str());
}

template <typename T>
void save_images(const T* data, int width, int height, const std::string& filename) {

	std::cout << "Writing image files... " << std::endl;
	for (int i = 0; i < N_OUTPUT_DIMS / 3; i++) {
		save_image(data, width, height, 3, N_OUTPUT_DIMS, i * 3, fmt::format("images/ch{}_", i) + filename);
	}
	std::cout << "Done." << std::endl;
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

__global__ void setup_kernel(uint32_t n_elements, curandState* state) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockDim.x * blockIdx.y;

	if (idx >= n_elements)
		return;

	curand_init(1337, idx, 0, &state[idx]);
}

template <typename T>
__global__ void generate_face_positions(uint32_t n_elements, uint32_t n_faces, curandState* crs, tinyobj::index_t* indices, float* vertices, float* cdf, float* result) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements)
		return;

	int output_idx = idx * 3;

	// Pick random face
	// TODO: weight by face area
	float r = fmodf(curand_uniform(&crs[idx]), 1.0f);

	/* 
	uint32_t faceId = n_faces - 1;

	for (uint32_t i = 0; i < n_faces; i++) {
		if (r < cdf[i]) {
			faceId = i;
			break;
		}
	}
	*/

	uint32_t faceId = (uint32_t)(r * n_faces);


	int iv1, iv2, iv3; // Indices of adjacent vertices
	iv1 = indices[3 * faceId + 0].vertex_index;
	iv2 = indices[3 * faceId + 1].vertex_index;
	iv3 = indices[3 * faceId + 2].vertex_index;
	//assert(iv1 == indices[3 * faceId + 0].texcoord_index);
	//assert(iv2 == indices[3 * faceId + 1].texcoord_index);
	//assert(iv3 == indices[3 * faceId + 2].texcoord_index);

	result[output_idx + 0] = *((float*) &faceId);
	//result[output_idx + 0] = (float)iv1;
	//result[output_idx + 1] = (float)iv2;
	//result[output_idx + 2] = (float)iv3;

	// Barycentric coordinates
	T alpha, beta, gamma;
	// TODO: is this actually uniform??
	alpha = (T)curand_uniform(&crs[idx]);
	alpha = alpha - (T)(long)alpha;
	beta = (T)curand_uniform(&crs[idx]) * ((T)1.0f - (T)alpha);
	beta = beta - (T)(long)beta;
	gamma = (T)1.0f - alpha - beta;


	result[output_idx + 1] = (float)alpha;
	result[output_idx + 2] = (float)beta;
	//result[output_idx + 3] = alpha;
	//result[output_idx + 4] = beta;
	//result[output_idx + 5] = gamma;
}

__global__ void rescale_faceIds(uint32_t n_elements, uint32_t n_faces, float* training_batch, float* result) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements)
		return;

	int input_idx = idx * 3;
	int output_idx = idx * 3;

	uint32_t faceId = *((uint32_t*) &training_batch[output_idx + 0]);

	result[input_idx + 0] = (float)faceId / (float)n_faces;
	result[input_idx + 1] = training_batch[output_idx + 1];
	result[input_idx + 2] = training_batch[output_idx + 2];

}

__global__ void generate_training_target(uint32_t n_elements, uint32_t n_faces, Material* materials, int* material_ids, float* training_batch, tinyobj::index_t* indices, float* texcoords, float* result) {

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

	faceId = *((uint32_t*)&training_batch[input_idx + 0]);
	if (faceId < 0 || faceId >= n_faces) {
		result[output_idx + 0] = 0.f;
		result[output_idx + 1] = 0.f;
		result[output_idx + 2] = 0.f;
		return;
	}

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

	// This needs to match N_OUTPUT_DIMS

	if (materials[material_ids[faceId]].map_Kd.valid) {
		cudaTextureObject_t texture_diffuse = materials[material_ids[faceId]].map_Kd.texture;
		float4 val_diff = tex2D<float4>(texture_diffuse, uv_interp.x, uv_interp.y);
		result[output_idx + 0] = val_diff.x;
		result[output_idx + 1] = val_diff.y;
		result[output_idx + 2] = val_diff.z;
	}
	else {
		result[output_idx + 0] = materials[material_ids[faceId]].diffuse[0];
		result[output_idx + 1] = materials[material_ids[faceId]].diffuse[1];
		result[output_idx + 2] = materials[material_ids[faceId]].diffuse[2];
	}
	if (materials[material_ids[faceId]].map_Bump.valid) {
		cudaTextureObject_t texture_bump = materials[material_ids[faceId]].map_Bump.texture;
		float4 val_bump = tex2D<float4>(texture_bump, uv_interp.x, uv_interp.y);
		result[output_idx + 3] = val_bump.x;
		result[output_idx + 4] = val_bump.y;
		result[output_idx + 5] = val_bump.z;
	}
	else {
		result[output_idx + 3] = 0.5f;
		result[output_idx + 4] = 0.5f;
		result[output_idx + 5] = 1.0f;
	}

}

std::vector<std::string> splitString(std::string input, char delimiter) {
	std::vector<std::string> result;
	std::istringstream stream(input);
	std::string token;

	while (std::getline(stream, token, delimiter)) {
		result.push_back(token);
	}

	return result;
}

EvalResult trainAndEvaluate(json config, GPUMemory<tinyobj::index_t>* indices, std::vector<tinyobj::index_t> indices_host,
	GPUMemory<float>* vertices, std::vector<float> vertices_host, GPUMemory<float>* texcoords, GPUMemory<float>* cdf,
	GPUMemory<Material>* materials, GPUMemory<int>* material_ids,
	int sampleWidth, int sampleHeight, GPUMemory<float>* test_batch, long* training_time_ms,
	uint32_t training_iterations) {
	try {

		EvalResult res;

		/* =======================
		*  === TRAIN THE MODEL ===
		*  =======================
		*/

			// Various constants for the network and optimization
		const uint32_t batch_size = 1 << 18;

		const uint32_t n_training_steps = training_iterations + 1;

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		default_rng_t rng{ 1337 };
		// Auxiliary matrices for training

		GPUMatrix<float> training_batch_raw(N_INPUT_DIMS, batch_size);
		GPUMatrix<float> training_batch(N_INPUT_DIMS, batch_size);
		GPUMatrix<float> training_target(N_OUTPUT_DIMS, batch_size);

		json encoding_opts = config.value("encoding", json::object());
		json loss_opts = config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts = config.value("network", json::object());

		uint32_t n_vertices = vertices_host.size() / 3;
		uint32_t n_faces = indices_host.size() / 3;

		std::shared_ptr<Loss<precision_t>> loss{ create_loss<precision_t>(loss_opts) };
		std::shared_ptr<Optimizer<precision_t>> optimizer{ create_optimizer<precision_t>(optimizer_opts) };
		std::shared_ptr<Encoding<precision_t>> encoding{ create_vertex_encoding<precision_t>(N_INPUT_DIMS, n_vertices, n_faces, indices_host, vertices_host, encoding_opts) };
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(encoding, N_OUTPUT_DIMS, network_opts);
		//std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(std::shared_ptr<Encoding<precision_t>>{create_encoding<precision_t>(n_input_dims, encoding_opts)}, n_output_dims, network_opts);

		auto model = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point end;


		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

		// RNG
		curandState* crs;
		cudaMalloc(&crs, sizeof(curandState) * batch_size);
		linear_kernel(setup_kernel, 0, training_stream, batch_size, crs);

		uint32_t interval = 10;

		for (uint32_t i = 0; i < n_training_steps; ++i) {
			bool print_loss = i % interval == 0;
			bool visualize_learned_func = /*(i % interval == 0) || */(i == (n_training_steps - 1));
			bool writeEvalResult = i == (n_training_steps - 1);

			/* ===============================
			*  === GENERATE TRAINING BATCH ===
			*  ===============================
			*/
			
			// Generate Surface Points - training input
			linear_kernel(generate_face_positions<precision_t>, 0, training_stream, batch_size, n_faces, crs, indices->data(), vertices->data(), cdf->data(), training_batch_raw.data());
			//linear_kernel(rescale_faceIds, 0, training_stream, batch_size, n_faces, training_batch_raw.data(), training_batch.data());

				// Sample reference texture at surface points - training output
				linear_kernel(generate_training_target, 0, training_stream, batch_size, n_faces, materials->data(), material_ids->data(), training_batch_raw.data(), indices->data(), texcoords->data(), training_target.data());

			/* =========================
			*  === RUN TRAINING STEP ===
			*  =========================
			*/

			
			//auto ctx = trainer->training_step(training_stream, training_batch, training_target);

			auto ctx_obj = model->training_step(training_stream, training_batch_raw, training_target);

			if (i % std::min(interval, (uint32_t)100) == 0) {
				tmp_loss += model->loss(training_stream, *ctx_obj);
				++tmp_loss_counter;
			}
			




			// Debug outputs
			{

				if (writeEvalResult) {
					res.MSE = tmp_loss / (float)tmp_loss_counter;
					res.n_floats = encoding.get()->n_params();
				}

				if (print_loss) {
					end = std::chrono::steady_clock::now();
					std::cout << "Step#" << i << ": " << "loss=" << tmp_loss / (float)tmp_loss_counter << " time=" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

					tmp_loss = 0;
					tmp_loss_counter = 0;
				}

				if (visualize_learned_func) {

					// Auxiliary matrices for evaluation
					GPUMatrix<float> prediction(N_OUTPUT_DIMS, sampleWidth * sampleHeight);
					GPUMatrix<float> inference_batch(test_batch->data(), N_INPUT_DIMS, sampleWidth * sampleHeight);

					cudaThreadSynchronize();
					std::chrono::steady_clock::time_point evalStart = std::chrono::steady_clock::now();
					network->inference(inference_stream, inference_batch, prediction);
					cudaThreadSynchronize();
					std::chrono::steady_clock::time_point evalEnd = std::chrono::steady_clock::now();
					std::cout << "Evaluation time = " << std::chrono::duration_cast<std::chrono::microseconds>(evalEnd - evalStart).count() << "[microseconds]" << std::endl;

					std::vector<float> debug = prediction.to_cpu_vector();
					auto filename = fmt::format("nl{}_nmf{}_nbins{}_niter{}.jpg",
						encoding_opts.value("n_levels", 1u), std::log2(encoding_opts.value("max_features_level", 1u << 14)),
						encoding_opts.value("n_quant_bins", 16u), training_iterations - 5000u);
					save_images(prediction.data(), sampleWidth, sampleHeight, filename);
				}

				

				// Don't count visualizing as part of timing
				// (assumes visualize_learned_pdf is only true when print_loss is true)
				if (print_loss) {
					//begin = std::chrono::steady_clock::now();
				}
			}

			if (print_loss && i > 0 && interval < 1000) {
				interval *= 10;
			}
		}

		*training_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

		cudaFree(crs);
		free_all_gpu_memory_arenas();

		return res;
	}
	catch (const std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

}

void loadTexture(std::string path, Texture* tex) {

	try {
		tex->image = load_image(path, tex->width, tex->height);
	}
	catch (const std::exception& e) {
		tex->valid = false;
		return;
	}

	// Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
	memset(&tex->resDesc, 0, sizeof(tex->resDesc));
	tex->resDesc.resType = cudaResourceTypePitch2D;
	tex->resDesc.res.pitch2D.devPtr = tex->image.data();
	tex->resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	tex->resDesc.res.pitch2D.width = tex->width;
	tex->resDesc.res.pitch2D.height = tex->height;
	tex->resDesc.res.pitch2D.pitchInBytes = tex->width * 4 * sizeof(float);

	memset(&tex->texDesc, 0, sizeof(tex->texDesc));
	tex->texDesc.filterMode = cudaFilterModeLinear;
	tex->texDesc.normalizedCoords = true;
	tex->texDesc.addressMode[0] = cudaAddressModeClamp;
	tex->texDesc.addressMode[1] = cudaAddressModeClamp;
	tex->texDesc.addressMode[2] = cudaAddressModeClamp;

	CUDA_CHECK_THROW(cudaCreateTextureObject(&tex->texture, &tex->resDesc, &tex->texDesc, nullptr));

	tex->valid = true;
}

int main(int argc, char* argv[]) {
	
	uint32_t compute_capability = cuda_compute_capability();
	if (compute_capability < MIN_GPU_ARCH) {
		std::cerr
			<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
			<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
	}

	/*
	if (argc < 2) {
		std::cout << "USAGE: " << argv[0] << " " << "path-to-image.jpg [path-to-optional-config.json]" << std::endl;
		std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
		return 0;
	}*/

	/*
	if (argc >= 3) {
		std::cout << "Loading custom json config '" << argv[2] << "'." << std::endl;
		std::ifstream f{argv[2]};
		config = json::parse(f, nullptr, true, /*skip_comments=*//*true);
	}*/

	/* =========================
	*  === LAUNCH PARAMETERS ===
	*  =========================
	*/

	std::string object_basedir = "data/objects/barramundifish/";
	std::string object_path = object_basedir + "barramundifish.obj";
	//std::string texture_path = object_basedir + "BarramundiFish_baseColor.png";
	std::string sample_path = "data/objects/sample_fish.csv";


	/* ======================
	*  === LOAD .OBJ FILE ===
	*  ======================
	*/

	std::cout << "Loading " << object_path << "..." << std::flush;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> mats;
	std::string err;
	std::string warn;
	// Expect '.mtl' file in the same directory and triangulate meshes
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, object_path.c_str(), object_basedir.c_str());
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

	int shapeIndex = 0;
	tinyobj::mesh_t mesh = shapes[shapeIndex].mesh;
	int n_vertices = attrib.vertices.size() / 3;
	int n_texcoords = attrib.texcoords.size() / 2;

	int n_indices = mesh.indices.size();
	int n_faces = n_indices / 3;

	// Generate histogram of face areas to sample surface points
	std::vector<float> histogram(n_faces);
	float area_sum = 0.f;
	for (size_t i = 0; i < n_faces; i++) {
		vec3 v1(attrib.vertices[3 * mesh.indices[3 * i + 0].vertex_index + 0],
			attrib.vertices[3 * mesh.indices[3 * i + 0].vertex_index + 1],
			attrib.vertices[3 * mesh.indices[3 * i + 0].vertex_index + 2]);

		vec3 v2(attrib.vertices[3 * mesh.indices[3 * i + 1].vertex_index + 0],
			attrib.vertices[3 * mesh.indices[3 * i + 1].vertex_index + 1],
			attrib.vertices[3 * mesh.indices[3 * i + 1].vertex_index + 2]);

		vec3 v3(attrib.vertices[3 * mesh.indices[3 * i + 2].vertex_index + 0],
			attrib.vertices[3 * mesh.indices[3 * i + 2].vertex_index + 1],
			attrib.vertices[3 * mesh.indices[3 * i + 2].vertex_index + 2]);

		float area = 0.5f * length(cross(v2 - v1, v3 - v1));
		histogram[i] = area;
		area_sum += area;
	}

	// Create CDF from histogram
	std::vector<float> cdf_host(n_faces);
	float c_prob = 0.f;
	for (size_t i = 0; i < n_faces; i++) {
		c_prob += histogram[i] / area_sum;
		cdf_host[i] = c_prob;
	}

	// write vertices, indices and cdf to GPU memory
	GPUMemory<float> vertices(n_vertices * 3);
	vertices.copy_from_host(attrib.vertices);
	GPUMemory<float> texcoords(n_texcoords * 2);
	texcoords.copy_from_host(attrib.texcoords);
	GPUMemory<float> cdf(n_faces);
	cdf.copy_from_host(cdf_host);

	GPUMemory<tinyobj::index_t> indices(n_indices);
	std::vector<tinyobj::index_t> indices_host = shapes[shapeIndex].mesh.indices;
	indices.copy_from_host(indices_host);

	/* ==========================
	*  === LOAD MATERIAL DATA ===
	*  ==========================
	*/
	int n_materials = mats.size();
	std::vector<Material> materials_host(n_materials);
	GPUMemory<Material> materials(n_materials);
	GPUMemory<int> material_ids(n_faces);

	for (int i = 0; i < n_materials; i++) {
		materials_host[i].diffuse[0] = mats[i].diffuse[0];
		materials_host[i].diffuse[1] = mats[i].diffuse[1];
		materials_host[i].diffuse[2] = mats[i].diffuse[2];
		loadTexture(object_basedir + mats[i].diffuse_texname, &materials_host[i].map_Kd);
		loadTexture(object_basedir + mats[i].bump_texname, &materials_host[i].map_Bump);
	}

	materials.copy_from_host(materials_host);
	material_ids.copy_from_host(mesh.material_ids);

	/* =======================
	*  === LOAD TEST INPUT ===
	*  =======================
	*/
	const bool testInput = true;

	// Vector to store the floats
		
	int sampleWidth, sampleHeight;
	sampleWidth = 0;
	sampleHeight = 0;
	std::vector<float> surface_positions;

	if (testInput) {

		std::ifstream file(sample_path);

		if (!file.is_open()) {
			std::cerr << "Error opening file: " << sample_path << std::endl;
			return 1;
		}		

		// Read width and height
		std::string firstLine;	

		if (std::getline(file, firstLine)) {
			std::vector<std::string> w_and_h = splitString(firstLine, ',');
			sampleWidth = std::stoi(w_and_h[0]);
			sampleHeight = std::stoi(w_and_h[1]);
		}
		else {
			std::cerr << "Error reading sample.csv" << std::endl;
			return 1;
		}

		// Continue reading the remaining lines
		std::string line;
		while (std::getline(file, line)) {
			std::vector<std::string> s_pos = splitString(line, ',');
			uint32_t faceId = static_cast<uint32_t>(std::stoul(s_pos[0]));
			surface_positions.push_back(*((float*) &faceId));
			surface_positions.push_back(std::stof(s_pos[1]));
			surface_positions.push_back(std::stof(s_pos[2]));

		}

		// Close the file
		file.close();
	}

	// Write surface_positions to GPU memory
	GPUMemory<float> test_batch(sampleWidth * sampleHeight * 3);
	test_batch.copy_from_host(surface_positions.data());

	if (testInput) {
			
		cudaStream_t test_generator_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&test_generator_stream));
		GPUMatrix<float> test_generator_batch(test_batch.data(), N_OUTPUT_DIMS, sampleWidth * sampleHeight);
		GPUMatrix<float> test_generator_result(N_OUTPUT_DIMS, sampleWidth * sampleHeight);
		linear_kernel(generate_training_target, 0, test_generator_stream, sampleWidth * sampleHeight, n_faces, materials.data(), material_ids.data(), test_generator_batch.data(), indices.data(), texcoords.data(), test_generator_result.data());

		auto filename = "data_generator_test.jpg";
		save_images(test_generator_result.data(), sampleWidth, sampleHeight, filename);
			
	}


	/* =========================
		TRAINING AND EVALUATION
	   =========================
	*/

	uint32_t n_test_cases = 27;
	//uint32_t ns_level[] = {   1,1,1,1,1, 1,		2,2,2,2,2, 2,	3,3,3,3,3, 3,	4,4,4,4,4, 4,	5,5,5,5,	6,6,6,	7,7 };
	//uint32_t ns_feature[] = { 1,2,4,8,16,32,	1,2,4,8,16,32,	1,2,4,8,16,32,	1,2,4,8,16,32,	1,2,4,8,	1,2,4,	1,2 };

	uint32_t ns_level[] = {		2, 3, 4, 5, 6, 7,	8, 9, 10,2, 3, 4,	5 ,6 ,7 ,8 ,9 ,10,	2, 3, 4, 5, 6, 7,	8, 9, 10 };
	uint32_t ns_feature[] = {	2, 2, 2, 2, 2, 2,	2, 2, 2, 2, 2, 2,	2, 2, 2, 2, 2, 2,	2, 2, 2, 2, 2, 2,	2, 2, 2 };
	uint32_t max_fs_level[] = { 18,18,18,18,18,18,	18,18,18,19,19,19,	19,19,19,19,19,19,	20,20,20,20,20,20,	20,20,20 };

	uint32_t ns_bins[] = { 32,32,64,16,			   32,32,32,32,			   64, 64, 64 , 64 };
	uint32_t ns_iter[] = { 5250, 5250, 5250, 6000, 5250, 5500, 5750, 6000, 5250, 5500, 5750, 6000 };

	for (size_t j = 0; j < 1; j++) {

		std::ofstream outCsv;
		std::string csvFileName = fmt::format("evaluation_nbins{}_niter{}.csv", ns_bins[j], ns_iter[j] - 5000u);
		csvFileName = "evaluation_moreChannels.csv";
		outCsv.open(csvFileName);
		outCsv << "n_levels, n_features, max_fs_level, n_floats, loss, training_time_ms\n";

		for (size_t i = 0; i < n_test_cases; i++) {

			printf("============================\n");
			printf("    TEST CASE %i / %i\n", i + 1, n_test_cases);
			printf("============================\n");

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
				{"otype", "Vertex"},
				{"n_features", ns_feature[i]},
				{"n_levels", ns_level[i]},
				{"max_features_level", 1u << max_fs_level[i]},
				{"n_quant_bins", ns_bins[j]},
				{"n_quant_iterations", 0}
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

		long training_time_ms;
		EvalResult res = trainAndEvaluate(config, &indices, indices_host, &vertices, attrib.vertices, &texcoords, &cdf,
			&materials, &material_ids, sampleWidth, sampleHeight, &test_batch, &training_time_ms, /*ns_iter[j]*/5000);

			outCsv << fmt::format("{},{},{},{},{},{}\n", ns_level[i], ns_feature[i], max_fs_level[i], res.n_floats, res.MSE, training_time_ms);
		}

		outCsv.close();
	}

	return EXIT_SUCCESS;
}

