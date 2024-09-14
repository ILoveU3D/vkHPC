#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>

#include <vulkan/vulkan.h>

#include "CommandLineParser.hpp"
#include "VulkanInitializers.hpp"
#include "utils.hpp"

class ComputeManager
{
public:
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	uint32_t queueFamilyIndex;
	VkPipelineCache pipelineCache;
	VkQueue queue;
	VkCommandPool commandPool;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout[2];
	VkDescriptorSet descriptorSet[2];
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule shaderModule;
	CommandLineParser commandLineParser;

	VkResult createBuffer(DeviceMemoryBlock *block, BufferFlag flag){
		VkBufferUsageFlags usageFlags;
		VkMemoryPropertyFlags memoryPropertyFlags;
		switch (flag){
		case GPU_BUFFER:
			usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			break;
		case CPU_BUFFER:
			usageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			memoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			break;
		}
		
		// Create the buffer handle
		VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, block->size);
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &block->buffer));

		// Create the memory backing up the buffer handle
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
		VkMemoryRequirements memReqs;
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vkGetBufferMemoryRequirements(device, block->buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Find a memory type index that fits the properties of the buffer
		bool memTypeFound = false;
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
			if ((memReqs.memoryTypeBits & 1) == 1) {
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
					memAlloc.memoryTypeIndex = i;
					memTypeFound = true;
					break;
				}
			}
			memReqs.memoryTypeBits >>= 1;
		}
		assert(memTypeFound);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &block->memory));

		VK_CHECK_RESULT(vkBindBufferMemory(device, block->buffer, block->memory, 0));

		return VK_SUCCESS;
	}

	VkResult stageMemorycpy(DeviceMemoryBlock* srcBlock, DeviceMemoryBlock* dstBlock){
		// Copy to staging buffer
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VkCommandBuffer copyCmd;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));

		VkBufferCopy copyRegion = {};
		copyRegion.size = srcBlock->size;
		vkCmdCopyBuffer(copyCmd, srcBlock->buffer, dstBlock->buffer, 1, &copyRegion);
		VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));
		
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCmd;
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);

		return VK_SUCCESS;
	}

	VkResult preparePipeline(DeviceMemoryBlock* deviceMemory1, DeviceMemoryBlock* deviceMemory2){
		// descriptorSetLayout0
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout[0]));
		// descriptorSetLayout1
                setLayoutBindings = {
                        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
                };
                descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout[1]));

		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, descriptorSetLayout, 2);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, descriptorSet));

		VkDescriptorBufferInfo bufferDescriptor1 = { deviceMemory1->buffer, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo bufferDescriptor2 = { deviceMemory2->buffer, 0, VK_WHOLE_SIZE };
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bufferDescriptor1),
			vks::initializers::writeDescriptorSet(descriptorSet[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &bufferDescriptor2),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(descriptorSetLayout, 2);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));

		// Create pipeline
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);

		// Pass SSBO size via specialization constant
		struct SpecializationData {
			uint32_t BUFFER_ELEMENT_COUNT = 32;
		} specializationData;
		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(SpecializationData), &specializationData);

		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = loadShader((std::string(SHADER_PATH) + "headless.comp.spv").c_str(), device);
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &specializationInfo;

		assert(shaderStage.module != VK_NULL_HANDLE);
		computePipelineCreateInfo.stage = shaderStage;
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));
		
		return VK_SUCCESS;
	}

	VkResult compute(DeviceMemoryBlock* deviceMemory){
		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VkCommandBuffer commandBuffer;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

		// Fence for compute CB sync
		VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

		// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.buffer = deviceMemory->buffer;
		bufferBarrier.size = VK_WHOLE_SIZE;
		bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_HOST_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, descriptorSet, 0, 0);

		vkCmdDispatch(commandBuffer, 32, 1, 1);

		// Barrier to ensure that shader writes are finished before buffer is read back from GPU
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		bufferBarrier.buffer = deviceMemory->buffer;
		bufferBarrier.size = VK_WHOLE_SIZE;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

		// Submit compute work
		vkResetFences(device, 1, &fence);
		const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &commandBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		return VK_SUCCESS;
	}

	VkResult clean(DeviceMemoryBlock *block){
		vkDestroyBuffer(device, block->buffer, nullptr);
		vkFreeMemory(device, block->memory, nullptr);
		return VK_SUCCESS;
	}

	VkResult blockMemoryCopy(DeviceMemoryBlock *block, void* data, MemoryCopyFlag flag){
		// Make device writes visible to the host
		void *mapped;
		vkMapMemory(device, block->memory, 0, VK_WHOLE_SIZE, 0, &mapped);
		VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
		mappedRange.memory = block->memory;
		mappedRange.offset = 0;
		mappedRange.size = VK_WHOLE_SIZE;
		vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);

		switch (flag){
		case MEMORY_BLOCK_TO_USER:
			memcpy(data, mapped, block->size);
			break;
		case MEMORY_USER_TO_BLOCK:
			memcpy(mapped, data, block->size);
			break;
		}

		vkFlushMappedMemoryRanges(device, 1, &mappedRange);
		vkUnmapMemory(device, block->memory);
		return VK_SUCCESS;
	}

	ComputeManager()
	{
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "hpc";
		appInfo.apiVersion = VK_API_VERSION_1_3;

		/*
			Vulkan instance creation (without surface extensions)
		*/
		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pApplicationInfo = &appInfo;
		VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

		/*
			Vulkan device creation
		*/
		// Physical device (always use first)
		uint32_t deviceCount = 0;
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
		physicalDevice = physicalDevices[0];

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

		// Request a single compute queue
		const float defaultQueuePriority(0.0f);
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				queueFamilyIndex = i;
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = i;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
				break;
			}
		}
		// Create logical device
		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

		// Get a compute queue
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

		// Compute command pool
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));
	}

	~ComputeManager()
	{
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout[0], nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout[1], nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineCache(device, pipelineCache, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, shaderModule, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}
};
