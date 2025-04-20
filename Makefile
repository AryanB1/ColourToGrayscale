all:
	nvcc -std=c++17 -ccbin=g++-10 -o grayscale main.cu `pkg-config --cflags --libs opencv4`
