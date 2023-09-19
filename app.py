import requests
import json
import subprocess
import time
import re
import threading
import yaml
import os
from requests.exceptions import ChunkedEncodingError
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from markupsafe import escape
from generate_grammar import generate_grammar

from skills.timer import TimerSkill
from skills.weather import WeatherSkill
from skills.time import TimeSkill
from skills.openai_skill import OpenAISkill
from skills.math_skill import MathSkill
from skills.run_app_skill import RunAppSkill


url = "http://127.0.0.1:8080/completion"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
transcribe_filename = "transcript.txt"
grammar_filename = "commands_kwargs.gbnf"
llm_log_file = "llm_log.txt"


llm_settings = {"temperature": 0.8, "n_predict": 20,
                "force_grammar": True, "model": ""}

transcribe_settings = {'threads': '4', 'step': '3000', 'length': '10000', 'keep': '200',
                       'max-tokens': '32', 'vad-thold': '0.6', 'freq-thold': '100.0', 'speed-up': False, 'no-fallback': False, "model": ""}

llama_cpp_dir = "llama.cpp"
whisper_cpp_dir = "whisper.cpp"
llama_model_dir = os.path.join(llama_cpp_dir, "models")
whisper_model_dir = os.path.join(whisper_cpp_dir, "models")

prompt_setup = ""

old_prompts = []
old_responses = []

stop_talking = False
speak_flag = False
truth_flag = True
reset_dialog_flag = False

available_llm_models = []
available_transcribe_models = []
prompt_presets = []

sleeping = True
sleep_time_in_seconds = 60 * 10
last_action_time = time.time()

wake_words = ["otto", "auto", "wake"]

run_llm_on_new_transcription = True
skills = [TimerSkill, WeatherSkill, TimeSkill,
          OpenAISkill, RunAppSkill, MathSkill]
skill_instances = []


def strip_whitespace_from_promt(prompt):
    return prompt.replace("\n", " ").replace("\t", " ").strip()


def load_prompt_presets():
    global prompt_presets
    global prompt_setup
    with open("system_prompt_presets.yaml") as f:
        prompt_presets = yaml.load(f, Loader=yaml.FullLoader)

    if (prompt_presets == []):
        raise ("No prompt presets found in file system_prompt_presets.yaml")

    if prompt_setup == "":
        prompt_setup = prompt_presets[0]["prompt"]


def load_available_transcribe_models():
    global available_transcribe_models
    global transcribe_settings
    with open("transcribe_models.json") as f:
        known_transcribe_models = json.load(f)

    if known_transcribe_models == []:
        raise RuntimeError(
            "No whisper models found in file whisper_models.json")

    available_transcribe_models = []
    for model in known_transcribe_models:
        if (os.path.isfile(os.path.join(whisper_model_dir, model["model"]))):
            available_transcribe_models.append(model)

    if available_transcribe_models == []:
        raise RuntimeError(
            "Need to download at least one whisper model, see README.md")

    transcribe_settings["model"] = available_transcribe_models[0]["model"]


def load_available_llm_models():
    global prompt_setup
    global available_llm_models
    global llm_settings
    with open("llms.json") as f:
        known_llm_models = json.load(f)

    if known_llm_models == []:
        raise RuntimeError("No LLM models found in file llms.json")

    available_llm_models = []
    for model in known_llm_models:
        if (os.path.isfile(os.path.join(llama_model_dir, model["model"]))):
            available_llm_models.append(model)

    if available_llm_models == []:
        raise RuntimeError(
            "Need to download at least one LLM model and add model to llms.json, see README.md")
    llm_settings['model'] = available_llm_models[0]["model"]


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def emptyaudio(audiostr):
    if (audiostr.strip() == ""):
        return True

    if (audiostr.strip().startswith("[") or audiostr.strip().startswith("(")):
        return True
    else:
        return False


def cleanup_text_to_speak(text):
    # remove text inside parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    # remove text inside asterisks
    text = re.sub(r'\*[^)]*\*', '', text)
    return text


def message(text):
    print("Sending message ", text)
    socket_io.emit("message", text)


def load_skills():
    global skills
    global skill_instances
    for skill in skills:
        try:
            skill_instance = skill(message)
            skill_instances.append(skill_instance)
        except Exception as e:
            print("Error loading skill: ", e)


def parse_function_call(call_str: str) -> (str, dict[str, str]):
    call_str = call_str.strip()

    # Regular expression to match function name and arguments
    match = re.match(r'(\w+)\((.*)\)', call_str)

    if not match:
        return None

    function_name = match.group(1)

    # Extract arguments
    args_str = match.group(2)

    # Splitting by comma outside of quotes
    args = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', args_str)

    param_dict = {}
    if (args != ['']):
        for arg in args:
            key, value = arg.split("=")
            param_dict[key.strip()] = value.strip().strip('"')

    return function_name, param_dict


def function_call(function_name: str, args: dict[str, str]):
    global last_action_time

    socket_io.emit("function_call", {
                   "function_name": function_name, "args": args})

    action_happened = True

    if function_name != "other":
        for skill in skill_instances:
            if function_name == skill.function_name:
                skill.start(args)
                break

        if (action_happened):
            last_action_time = time.time()


def function_call_str(function_call_str):

    results = parse_function_call(function_call_str)
    if (results == None):
        return

    function_name, args = results

    function_call(function_name, args)


def end_response(response):
    function_call_str(response)


def end_response_chunk(left_to_read):
    if (speak_flag):
        cur_time = time.time()
        subprocess.run(
            ["say", cleanup_text_to_speak(left_to_read)])
        elapsed_time = time.time() - cur_time
        print("Elapsed time: ", elapsed_time)

        # if (number_people_talking() > 1):
        #    print("Stopping because multiple people are talking")
        #    break


def generate_prompt_chat(prompt_setup, user_prompt, old_prompts, old_responses):
    chat_prompt = ""
    for old_prompt, old_response in zip(old_prompts, old_responses):
        chat_prompt += f"[INST]{old_prompt}[/INST]\n{old_response}\n"

    prompt_header = f"""<s>[INST] <<SYS>>
        {prompt_setup}<</SYS>>[/INST]
        
    """
    prompt = f"{prompt_header} {chat_prompt}\n [INST]{user_prompt}[/INST]\n"
    return prompt


def generate_prompt_instruct(prompt_setup, user_prompt, old_prompts, old_responses):
    prompt_header = f"""### System:
{prompt_setup}
"""

    chat_prompt = ""
    for old_prompt, old_response in zip(old_prompts, old_responses):
        chat_prompt += f"### User: {old_prompt}\n### Assistant: {old_response}\n"

    prompt = f"{prompt_header} {chat_prompt}\n### User: {user_prompt}\n### Assistant: "
    return prompt


def generate_prompt(prompt_setup, user_prompt, old_prompts, old_responses, prompt_generator):

    if (prompt_generator == "instruct"):
        return generate_prompt_instruct(prompt_setup, user_prompt, old_prompts, old_responses)
    elif (prompt_generator == "chat"):
        return generate_prompt_chat(prompt_setup, user_prompt, old_prompts, old_responses)
    raise (
        f"Unknown LLM model prompt generator {prompt_generator}")


def log_llm(user_prompt, response):
    with open(llm_log_file, "a") as llm_log:
        print(
            f"### User: {user_prompt}\n### Assistant: {response}\n", file=llm_log)


def llm(user_prompt):
    global stop_talking
    global llm_settings
    prompt = generate_prompt(prompt_setup, user_prompt,
                             old_prompts, old_responses, get_prompt_generator_for_model(llm_settings["model"]))
    print("LLM -> ", user_prompt)

    socket_io.emit("prompt_setup", prompt_setup)
    socket_io.emit("old_prompts", {
                   "old_prompts": old_prompts, "old_responses": old_responses})
    socket_io.emit("user_prompt", user_prompt)

    socket_io.emit("prompt", prompt)

    llama_request = {"n_predict": llm_settings["n_predict"],
                     "prompt": prompt,
                     "stream": True,
                     "temperature": llm_settings["temperature"]}

    if ("force_grammar" in llm_settings and llm_settings["force_grammar"]):
        # with open(grammar_filename) as f:
        #     grammar_string = f.read()
        #     llama_request['grammar'] = grammar_string
        grammar_string = generate_grammar(skills)
        llama_request['grammar'] = grammar_string

    try:
        resp = requests.post(url, json=llama_request, stream=True)
    except requests.exceptions.ConnectionError as e:
        socket_io.emit("error", "LLM Connection Error")
        print("LLM Connection Error")
        return

    try:
        response = ""
        left_to_read = ""
        for line in resp.iter_lines():
            if line:
                try:
                    line = line.decode('utf-8')
                    content = line.split(": ", 1)[1]
                    data = json.loads(content)
                except Exception as e:
                    print(f"Couldn't parse line {line} - exeception {e} ")
                    continue

                token = data['content']

                if not line:
                    break

                socket_io.emit("response", token)
                response += token
                left_to_read += token
                if (token == "." or token == "?" or token == "!" or token == ","):
                    end_response_chunk(left_to_read)
                    left_to_read = ""

                if (stop_talking):
                    break
    except requests.exceptions.ChunkedEncodingError as e:
        socket_io.emit("error", "LLM Connection Error")
        print("LLM Connection Error")
        return
    finally:
        stop_talking = False

    if left_to_read != "":
        end_response_chunk(left_to_read)
        log_llm(user_prompt, response)
        end_response(response)
        left_to_read = ""

    if (reset_dialog_flag):
        pass
    else:
        old_responses.append(response)
        old_prompts.append(user_prompt)
    # llm_log.flush()


last_tts_line = ""


def listen():
    global run_llm_on_new_tts
    global sleep_time_in_seconds
    global sleeping
    global last_tts_line
    p = subprocess.Popen(
        ["tail", "-1", transcribe_filename], stdout=subprocess.PIPE)
    try:
        line = p.stdout.readline()
    except Exception as e:
        print("Error reading transcription: " + e)

    if (line == last_tts_line):  # don't call llm twice on the same line if it returns quickly
        return

    last_tts_line = line

    print("Seeing line: ", line)

    try:
        line = line.decode('utf-8')
        line = strip_ansi_codes(line)
        socket_io.emit("transcribe", line)

    except Exception as e:
        print("Error reading transcription: " + e)
        socket_io.emit("error", "Error reading transcription: " + e)
        return
    # lines.append(line)

    # if last_action was more than sleep_timer secongs ago, go to sleep
    if (time.time() - last_action_time > sleep_time_in_seconds):
        print("Going to sleep")
        socket_io.emit("sleeping", str(True))
        sleeping = True

    if (sleeping):
        line_words = re.split(r'\W+', line.lower())
        for word in wake_words:
            if (word in line_words):
                print("Waking up")
                socket_io.emit("sleeping", str(False))
                last_tts_line = line  # don't call llm on the wake word
                sleeping = False
                break
    else:
        print("Awake, running on new transcription ",
              run_llm_on_new_transcription)
        if run_llm_on_new_transcription:
            line = line.strip()
            print("line: ", line)

            # Example line [_BEG_] - He can sit.[_TT_42][_TT_42] - He wants to do it?[_TT_125]<|endoftext|>
            # if (line.endswith("<|endoftext|>")):
            # remove everything in line inside of []

            line = re.sub(r'\[[^]]*\]', '', line)
            # remove everything in line inside of <>
            line = re.sub(r'\<[^>]*\>', '', line)
            print(f"cleaned line: {line} emptyaudio {emptyaudio(line)}")
            if not emptyaudio(line):
                llm(line)


def listen_loop():
    while True:
        try:
            listen()
        except ChunkedEncodingError as e:
            print("Error reading transcription: " + e)

        time.sleep(0.2)


transcribe_thread = None
transcribe_process = None


def run_transcribe():
    global transcribe_process
    global transcribe_settings

    transcribe_args = []

    for setting, value in transcribe_settings.items():
        if isinstance(value, bool):
            if value == True:
                transcribe_args.append(f"--{setting}")
        if isinstance(value, str):
            transcribe_args.append(f"--{setting}")
            transcribe_args.append(value)

    transcribe_args.extend(["-m", os.path.join(whisper_model_dir,
                                               transcribe_settings["model"]),
                            "--print-special",
                            "-f", transcribe_filename])

    transcribe_command = [os.path.join(
        whisper_cpp_dir, "stream")] + transcribe_args
    socket_io.emit("transcribe_command", transcribe_command)

    transcribe_process = subprocess.Popen(transcribe_command,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)
    while transcribe_process != None:
        line = transcribe_process.stdout.readline()
        line = line.decode('utf-8')
        line = strip_ansi_codes(line)
        socket_io.emit("raw_transcription", line)
        time.sleep(0.1)

    print("Exiting transcribe process")


llm_thread = None
llm_process = None


def run_llm():
    global llm_process
    global llm_settings

    if (llm_settings['model'] == ""):
        raise RuntimeError("No LLM model selected")

    args = [os.path.join(llama_cpp_dir, "server"),
            "-m", os.path.join(llama_model_dir,
                               llm_settings["model"]),
            "--ctx-size", "2048",
            "--threads", "10",
            "--n-gpu-layers", "1"]

    llm_process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    while llm_process != None:
        line = llm_process.stdout.readline()
        line = line.decode('utf-8')
        socket_io.emit("llm_stdout", line)
        time.sleep(0.1)


def get_prompt_generator_for_model(model: str):
    global available_llm_models
    # find model file in available_llm_models and return prompt_generator
    model = next(
        (item for item in available_llm_models if item["model"] == model), None)
    if (model == None):
        raise RuntimeError(
            "Could not find prompt generator for model: " + model)
    return model["prompt_generator"]


def update_llm_settings(new_llm_settings: dict):
    global llm_settings
    llm_settings = {**llm_settings, **new_llm_settings}


def update_transcribe_settings(new_transcribe_settings: dict):
    global transcribe_settings
    transcribe_settings = {**transcribe_settings, **new_transcribe_settings}


app = Flask(__name__, static_url_path='', static_folder='frontend/build')
# CORS(app)

# @app.route("/", defaults={'path':''})
# @cross_origin()
# def serve(path):
#     return send_from_directory(app.static_folder,'index.html')


# @app.route("/listening")
# def test():
#     return(str(listening))

# @app.route("/prompt")
# def prompt():
#     return(prompt_header)


t = None

socket_io = SocketIO(app, cors_allowed_origins="*")


@socket_io.on('set_prompt_setup')
def set_prompt_setup(new_prompt_setup):
    old_prompts = []
    old_responses = []
    global prompt_setup
    prompt_setup = new_prompt_setup


@socket_io.on('manual_prompt')
def manual_prompt(user_prompt):
    llm(user_prompt)


@socket_io.on('reset_dialog')
def reset_dialog():
    global old_prompts
    global old_responses
    old_prompts = []
    old_responses = []
    socket_io.emit("old_prompts", {
                   "old_prompts": old_prompts, "old_responses": old_responses})


@socket_io.on('start_llm')
def start_llm(llm_settings: dict = {}):
    global llm_thread
    stop_llm()

    update_llm_settings(llm_settings)

    llm_thread = threading.Thread(target=run_llm)
    llm_thread.start()


@socket_io.on('stop_llm')
def stop_llm():
    global llm_process
    global llm_thread
    if (llm_process != None):
        llm_process.terminate()
        llm_process = None


@socket_io.on('stop_talking')
def stop_talking():
    global stop_talking
    stop_talking = True


@socket_io.on('start_transcribe')
def start_transcribe(transcribe_settings: dict = {}):
    global transcribe_thread
    stop_transcribe()

    update_transcribe_settings(transcribe_settings)

    transcribe_thread = threading.Thread(target=run_transcribe)
    transcribe_thread.start()


@socket_io.on('stop_transcribe')
def stop_transcribe():
    global transcribe_process
    global transcribe_thread
    if (transcribe_process != None):
        transcribe_process.terminate()
        transcribe_process = None


@socket_io.on('start_automation')
def start_automation(args):
    # global llm_model
    global llm_settings
    global transcribe_model

    # if ('model_file' in args):
    #     llm_model = next(
    #         (item for item in available_llm_models if item["model_file"] == args['model_file']), None)
    #     if (llm_model == None):
    #         raise (
    #             f"Could not find llm model {args['llm_model']} in available_models")

    if ('transcribe_model' in args):
        # find the first model in available_transcribe_models where model_file equals transcribe_model
        transcribe_model = next(
            (item for item in available_transcribe_models if item["model"] == args['transcribe_model']), None)
        if (transcribe_model == None):
            raise (
                f"Could not find transcribe model {args['transcribe_model']} in available_models"
            )

    if 'llm_settings' in args:
        update_llm_settings(args['llm_settings'])

    if 'transcribe_settings' in args:
        update_transcribe_settings(args['transcribe_settings'])

    if args['run_transcribe'] == True:
        start_transcribe()

    global run_llm_on_new_transcription
    if args['run_llm'] == True:
        start_llm()
        run_llm_on_new_transcription = True

    global reset_dialog_flag
    if args['reset_dialog'] == True:
        reset_dialog_flag = True
        reset_dialog()
    elif args['reset_dialog'] == False:
        reset_dialog_flag = False

    global speak_flag
    if args['run_speak'] == True:
        speak_flag = True
    elif args['run_speak'] == False:
        speak_flag = False


@socket_io.on('stop_automation')
def stop_automation():
    global run_llm_on_new_transcription
    run_llm_on_new_transcription = False


@socket_io.on('connect')
def handle_connect():
    print('new connection')


@socket_io.on('request_status')
def update_status():
    # global listening
    global run_llm_on_new_transcription
    global speak_flag
    global reset_dialog_flag
    global llm_settings
    global transcribe_setting
    global available_transcribe_models
    global prompt_presets
    status = {
        "sleeping": str(sleeping),
        "run_llm": run_llm_on_new_transcription,
        "speak_flag": speak_flag,
        "reset_dialog_flag": reset_dialog_flag,
        "llm_settings": llm_settings,
        "transcribe_settings": transcribe_settings,
        "available_llm_models": available_llm_models,
        "prompt_presets": prompt_presets,
        "available_transcribe_models": available_transcribe_models
    }
    print("Updating Status", status)
    socket_io.emit("server_status", status)


@socket_io.on('call')
def call(call_str):
    function_call_str(call_str)


if __name__ == '__main__':
    load_skills()
    load_available_llm_models()
    load_available_transcribe_models()
    load_prompt_presets()

    listen_thread = threading.Thread(target=listen_loop)
    listen_thread.start()

    update_status()
    start_automation({"run_transcribe": True, "run_llm": True,
                     "reset_dialog": True, "run_speak": False})

    socket_io.run(app, port=5001, host="0.0.0.0")
