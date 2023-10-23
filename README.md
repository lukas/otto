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
make stream
```

On linux you may need to run:

```
sudo apt-get install libsdl2-dev
```

#### Download a whisper model

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
