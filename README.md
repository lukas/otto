# otto

Install node
Install python

## Getting Started

### Step 1: Install whisper and llama

Install [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
Install [llama.cpp](https://github.com/ggerganov/llama.cpp)

In root directory run

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama
make
```

```
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper
make
```

### Step 2 Get a whisper and llama model to run

### Step 3: Run backend

```
./app.py
```

### Step 4: Run frontend

```
cd frontend && yarn start
```

## Fine Tuning Llama model