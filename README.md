# otto

Install node
Install python

## Getting Started

### Step 1: Install whisper and llama

#### Install [llama.cpp](https://github.com/ggerganov/llama.cpp)

In root directory run

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama
make
```

#### Download a llama model

```
cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

#### Install [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

```
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper
make
```

#### Download a whisper model

```
bash ./models/download-ggml-model.sh base.en
```

### Step 2: Run backend

```
./app.py
```

### Step 3: Run frontend

```
cd frontend && yarn start
```

## Fine Tuning Llama model
