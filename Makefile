CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS ?= -fopenmp

TARGET := prefix_scan
SRC := src/main.cpp src/scan.cpp
HDR := src/scan.h src/scan.hpp

.PHONY: all clean run test benchmark plot

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --n 1048579 --threads 4 --repeats 9 --mode both --scan-type exclusive

test: $(TARGET)
	./$(TARGET) --test

benchmark: $(TARGET)
	./scripts/run_benchmarks.sh
	python3 scripts/plot_results.py results.csv

plot:
	python3 scripts/plot_results.py results.csv

clean:
	rm -f $(TARGET) results.csv charts/*.svg
