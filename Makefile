all :
	g++ -I src/include -L src/lib -o main src/main.cpp src/include/app/screen.cpp src/include/core/neural.cpp src/include/core/knn.cpp -lmingw32 -lSDL2main -lSDL2