#include<ComputeManager.hpp>

#define BUFFER_ELEMENTS 32

#include <random>

int main(int argc, char* argv[]) {
	/*
		Prepare storage buffers
	*/
	std::vector<float> computeInput1(BUFFER_ELEMENTS);
	std::vector<float> computeInput2(BUFFER_ELEMENTS);
	std::vector<float> computeOutput(BUFFER_ELEMENTS);

	// Fill input data
	std::random_device rd;
    	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);
	auto func = [&gen, &dist] { return dist(gen); };
	std::generate(computeInput1.begin(), computeInput1.end(), func);
	std::generate(computeInput2.begin(), computeInput2.end(), func);

	const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(float);
	ComputeManager *manager = new ComputeManager();

	DeviceMemoryBlock hostMemory[2], deviceMemory[2];
	hostMemory[0].size = bufferSize;
	deviceMemory[0].size = bufferSize;
	manager->createBuffer(&hostMemory[0], CPU_BUFFER);
	manager->createBuffer(&deviceMemory[0], GPU_BUFFER);
	hostMemory[1].size = bufferSize;
	deviceMemory[1].size = bufferSize;
	manager->createBuffer(&hostMemory[1], CPU_BUFFER);
	manager->createBuffer(&deviceMemory[1], GPU_BUFFER);

	manager->blockMemoryCopy(&hostMemory[0], computeInput1.data(), MEMORY_USER_TO_BLOCK);
	manager->stageMemorycpy(&hostMemory[0], &deviceMemory[0]);
	manager->blockMemoryCopy(&hostMemory[1], computeInput2.data(), MEMORY_USER_TO_BLOCK);
	manager->stageMemorycpy(&hostMemory[1], &deviceMemory[1]);
	
	manager->preparePipeline(&deviceMemory[0], &deviceMemory[1]);
	manager->compute(&deviceMemory[1]);
	
	manager->stageMemorycpy(&deviceMemory[1], &hostMemory[1]);
	manager->blockMemoryCopy(&hostMemory[1], computeOutput.data(), MEMORY_BLOCK_TO_USER);
	manager->clean(&deviceMemory[0]);
	manager->clean(&hostMemory[0]);
	manager->clean(&deviceMemory[1]);
	manager->clean(&hostMemory[1]);

	// Output buffer contents
	printf("input1: [");
	for (auto v : computeInput1) {
		printf("%.2f ", v);
	}
	std::cout << "]" << std::endl;

	// Output buffer contents
	printf("input2: [");
	for (auto v : computeInput2) {
		printf("%.2f ", v);
	}
	std::cout << "]" << std::endl;

	printf("output: [");
	for (auto v : computeOutput) {
		printf("%.2f ", v);
	}
	std::cout << "]" << std::endl;
	delete(manager);
	return 0;
}
