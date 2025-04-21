all: main neural_setup

main: src/main.cpp src/include/app/screen.cpp src/include/core/neural.cpp src/include/core/knn.cpp
	g++ -I src/include -L src/lib -o main src/main.cpp src/include/app/screen.cpp src/include/core/neural.cpp src/include/core/knn.cpp -lmingw32 -lSDL2main -lSDL2

neural_setup: src/neural_setup.cpp src/include/core/neural.cpp
	g++ -I src/include -L src/lib -o neural_setup src/neural_setup.cpp src/include/core/neural.cpp -lmingw32 -lSDL2main -lSDL2

clean:
	rm -f main neural_setup
