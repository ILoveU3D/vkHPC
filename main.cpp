#include<ComputeManager.hpp>

#define BUFFER_ELEMENTS 32

int main(int argc, char* argv[]) {
	/*
		Prepare storage buffers
	*/
	std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
	std::vector<uint32_t> computeOutput(BUFFER_ELEMENTS);

	// Fill input data
	uint32_t n = 0;
	std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

	const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);
	ComputeManager *manager = new ComputeManager();

	// VkBuffer deviceBuffer, hostBuffer;
	// VkDeviceMemory deviceMemory, hostMemory;
	DeviceMemoryBlock hostMemory, deviceMemory;
	hostMemory.size = bufferSize;
	deviceMemory.size = bufferSize;
	manager->createBuffer(CPU_BUFFER, &hostMemory);
	manager->createBuffer(GPU_BUFFER, &deviceMemory);

	manager->blockMemoryCopy(&hostMemory, computeInput.data(), MEMORY_USER_TO_BLOCK);
	manager->stageMemorycpy(&hostMemory, &deviceMemory);
	
	manager->preparePipeline(&deviceMemory);
	manager->compute(&hostMemory, &deviceMemory);
	
	manager->blockMemoryCopy(&hostMemory, computeOutput.data(), MEMORY_BLOCK_TO_USER);
	manager->clean(&deviceMemory);
	manager->clean(&hostMemory);

	// Output buffer contents
	printf("Compute input:\n");
	for (auto v : computeInput) {
		printf("%d \t", v);
	}
	std::cout << std::endl;

	printf("Compute output:\n");
	for (auto v : computeOutput) {
		printf("%d \t", v);
	}
	std::cout << std::endl;
	delete(manager);
	return 0;
}