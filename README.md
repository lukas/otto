# otto

Install node
Install python

## Getting Started

### Step 0: Install python, npm, yarn

On linux
```
sudo apt install python
sudo apt install npm
```

On mac
```
brew install python
brew install npm
```

```
npm install --global yarn
```

### Step 1: Install whisper and llama

#### Install [llama.cpp](https://github.com/ggerganov/llama.cpp)

In root directory run

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

#### Download a llama model
Store the model in llama.cpp/models
```
cd llama.cpp/models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

#### Download a finetuned model
Store the fine-tuned model in llama.cpp/models
```
wget https://api.wandb.ai/artifactsV2/gcp-us/l2k2/QXJ0aWZhY3Q6NjAyMDE2NzM0/25bd8f78a12839913fd2c4c49c3f9c22/ggml-finetuned-model-q4_0.gguf
```

#### Install [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
In root directory run
```
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
make stream
```

On linux you may need to run:

```
sudo apt-get install libsdl2-dev
```

#### Download a whisper model
Store the model in whisper.cpp/models
```
bash ./models/download-ggml-model.sh base.en
```

Feel free to add more llama or whisper models to play with. If you add a different llm, update llms.json with the model file and correct prompt format.

### Step 2: Install python libraries

```
pip install -r requirements.txt
pip install -r server/skills/requirements_skills.txt
```

### Step 3: Install yarn packages

```
cd frontend 
yarn install
```

### Step 4: Run backend

```
python server/app.py
```

### Step 5: Run frontend

```
cd frontend && yarn start
```

## Fine Tuning Llama model
