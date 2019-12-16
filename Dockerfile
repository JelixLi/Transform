FROM gcc
COPY hello.cpp .
RUN  gcc hello.cpp -o hello
