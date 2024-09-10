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
	manager->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &hostMemory, computeInput.data());
	manager->flushMemory(&hostMemory);
	manager->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &deviceMemory);
	manager->stageMemorycpy(&hostMemory, &deviceMemory);
	manager->preparePipeline(&deviceMemory);
	manager->compute(&hostMemory, &deviceMemory);
	manager->outputMemorycpy(&hostMemory, computeOutput.data());
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