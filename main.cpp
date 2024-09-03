#include<ComputeManager.hpp>

int main(int argc, char* argv[]) {
	commandLineParser.add("help", { "--help" }, 0, "Show help");
	commandLineParser.add("shaders", { "-s", "--shaders" }, 1, "Select shader type to use (glsl or hlsl)");
	commandLineParser.parse(argc, argv);
	if (commandLineParser.isSet("help")) {
		commandLineParser.printHelp();
		std::cin.get();
		return 0;
	}
	ComputeManager *manager = new ComputeManager();
	delete(manager);
	return 0;
}