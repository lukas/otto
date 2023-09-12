import requests
import json
import subprocess
import time
import re
import threading
import yaml
from requests.exceptions import ChunkedEncodingError
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from markupsafe import escape

from hooks.timer import TimerHook


url = "http://127.0.0.1:8080/completion"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
transcribe_filename = "transcript.txt"
grammar_filename = "commands.gbnf"

# diart_filename = "file.rttm"
prompt_setup = """"""

old_prompts = []
old_responses = []

stop_talking = False
speak_flag = False
truth_flag = True
reset_dialog_flag = False

available_models = []
prompt_presets = []

# modes: normal, transcribe, factcheck, timer, calendar, weather, news
mode = "normal"


def load_prompt_presets():
    global prompt_presets
    with open("system_prompt_presets.yaml") as f:
        prompt_presets = yaml.load(f, Loader=yaml.FullLoader)


def load_available_models():
    global available_models
    global llm_model
    with open("llms.json") as f:
        available_models = json.load(f)

    if available_models == []:
        raise ("No LLM models found in llms.json")

    llm_model = available_models[0]
    print("LLM Model set ", llm_model)


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def emptyaudio(audiostr):
    if (audiostr.strip() == ""):
        return True

    if (audiostr.strip().startswith("[") or audiostr.strip().startswith("(")):
        return True
    else:
        return False


# def number_people_talking(lookback_duration=1.0):
#     newest_timestamp = 0
#     speakers = set()
#     p = subprocess.Popen(
#         ["tail", "-10", "-r", diart_filename], stdout=subprocess.PIPE)
#     diart_lines = ""
#     for line in p.stdout.readlines():
#         line = line.decode('utf-8')
#         diart_lines += line + "\n"
#         data = line.split(" ")
#         timestamp = float(data[3])
#         duration = float(data[4])
#         speaker = data[7]
#         if (newest_timestamp == 0):
#             newest_timestamp = timestamp+duration
#         else:
#             if (timestamp > newest_timestamp - lookback_duration):
#                 pass
#             else:
#                 break

#         speakers.add(speaker)

#     diart_log.write(diart_lines)

#     diart_log.write("Speakers: {speakers}")
#     diart_log.flush()
#     return len(speakers)


# llm_log_data = ""
def cleanup_text_to_speak(text):
    # remove text inside parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    # remove text inside asterisks
    text = re.sub(r'\*[^)]*\*', '', text)
    return text


def message(text):
    socket_io.emit("message", text)


timer = TimerHook(message)


def end_response(response):
    global mode

    clean_response = response.lower().strip()
    if (re.match(r'^\[[a-z ]+\]', clean_response)):

        rest_of_response = clean_response.split("]", 1)[1]
        if (clean_response.startswith("[fact check]")):
            mode = "factcheck"
            socket_io.emit('factcheck', 'start')
        elif (clean_response.startswith("[timer]")):
            timer.start(rest_of_response)
        elif (clean_response.startswith("[calender]")):
            socket_io.emit('calendar')
        elif (clean_response.startswith("[weather]")):
            socket_io.emit('weather')
        elif (clean_response.startswith("[news]")):
            socket_io.emit('news')
        elif (clean_response.startswith("[transcribe]")):
            mode = "transcribe"
            socket_io.emit('transcribe')


def end_response_chunk(left_to_read):
    if (truth_flag):
        print("Left to read: ", left_to_read)
        left_to_read = left_to_read.strip()
        if (left_to_read.lower().startswith("true")):
            print("FC TRUE")
            socket_io.emit('factcheck', 'true')
        if (left_to_read.lower().startswith("false")):
            print("FC FALSE")

            socket_io.emit('factcheck', 'false')

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
        chat_prompt += f"### User:\n{old_prompt}\n### Assistant:n{old_response}\n"

    prompt = f"{prompt_header} {chat_prompt}\n### User:\n{user_prompt}\n### Assistant:\n"
    return prompt


def generate_prompt(prompt_setup, user_prompt, old_prompts, old_responses, prompt_generator):

    if (prompt_generator == "instruct"):
        return generate_prompt_instruct(prompt_setup, user_prompt, old_prompts, old_responses)
    elif (prompt_generator == "chat"):
        return generate_prompt_chat(prompt_setup, user_prompt, old_prompts, old_responses)
    raise (
        f"Unknown LLM model prompt generator {prompt_generator}")


def llm(user_prompt):
    global stop_talking
    prompt = generate_prompt(prompt_setup, user_prompt,
                             old_prompts, old_responses, llm_model["prompt_generator"])
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
        with open(grammar_file) as f:
            grammar_string = f.read()
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
                line = line.decode('utf-8')
                content = line.split(": ", 1)[1]
                data = json.loads(content)
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
        end_response(response)
        left_to_read = ""

    if (reset_dialog_flag):
        pass
    else:
        old_responses.append(response)
        old_prompts.append(user_prompt)
    # llm_log.flush()


lastprompt = ""

run_llm_on_new_tts = True


def listen():
    global run_llm_on_new_tts
    p = subprocess.Popen(
        ["tail", "-1", transcribe_filename], stdout=subprocess.PIPE)
    try:
        line = p.stdout.readline()
        line = line.decode('utf-8')
        line = strip_ansi_codes(line)
        socket_io.emit("tts", line)
        if (mode == "transcribe"):
            socket_io.emit('transcribe', line)
    except Exception as e:
        print("Error reading transcription: " + e)
        socket_io.emit("error", "Error reading transcription: " + e)
        return
    # lines.append(line)

    if run_llm_on_new_tts:
        if not emptyaudio(line):
            print("Calling llm with line ", line)
            llm(line)
            print("Done talking")
            time.sleep(1.0)
            print("Done sleeping")


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
    transcribe_process = subprocess.Popen(["./whisper.cpp/stream",
                                           "-m", "./whisper.cpp/models/ggml-base.en.bin",
                                           "-f", transcribe_filename],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)
    while transcribe_process != None:
        line = transcribe_process.stdout.readline()
        line = line.decode('utf-8')
        socket_io.emit("transcribe_stdout", line)
        time.sleep(0.1)


llm_thread = None
llm_process = None

llm_model = {"model_file": "", "prompt_generator": ""}
llm_settings = {"temperature": 0.8, "n_predict": 10}


def run_llm():
    global llm_process
    global llm_model
    global llm_settings

    if (llm_model == {"model_file": "", "prompt_generator": ""}):
        raise ("No LLM model selected")

    args = ["./llama.cpp/server",
            "-m", "llama.cpp/" +
            llm_model["model_file"],
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
def start_llm():
    global llm_thread
    stop_llm()

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
def start_transcribe():
    global transcribe_thread
    stop_transcribe()

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
    # global listening
    # global t

    print("Starting Automation: ", args)

    global llm_model
    global llm_settings
    if ('llm_model' in args):
        # find the first model in available_models where model_file equals llm_model
        llm_model = next(
            (item for item in available_models if item["model_file"] == args['llm_model']), None)
        print("LLM Model Set: ", llm_model)
        if (llm_model == None):
            raise (
                f"Could not find llm model {args['llm_model']} in available_models")

    if 'llm_settings' in args:
        if 'temperature' in args['llm_settings']:
            llm_settings['temperature'] = args['llm_settings']['temperature']
        if 'n_predict' in args['llm_settings']:
            llm_settings['n_predict'] = args['llm_settings']['n_predict']

    if args['run_tts'] == True:
        start_transcribe()

    if args['run_llm'] == True:
        start_llm()
        run_llm_on_new_tts = True

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
    global run_llm_on_new_tts
    # global t
    run_llm_on_new_tts = False
    # t = None


@socket_io.on('connect')
def handle_connect():
    print('new connection')


@socket_io.on('request_status')
def update_status():
    # global listening
    global run_llm_on_new_tts
    global speak_flag
    global reset_dialog_flag
    global llm_model
    global available_models
    global prompt_presets
    status = {
        "run_llm": run_llm_on_new_tts,
        "speak_flag": speak_flag,
        "reset_dialog_flag": reset_dialog_flag,
        "llm_model": llm_model["model_file"],
        "available_models": available_models,
        "prompt_presets": prompt_presets
    }
    print("Updating Status", status)
    socket_io.emit("server_status", status)


def update_tts():
    p = subprocess.Popen(
        ["tail", "-10", transcribe_filename], stdout=subprocess.PIPE)
    tts = p.communicate()[0]
    tts = tts.decode('utf-8')
    socket_io.emit("tts", tts)


def update_tts_loop():
    while (1):
        update_tts()
        time.sleep(0.2)


if __name__ == '__main__':
    load_available_models()
    load_prompt_presets()

    listen_thread = threading.Thread(target=listen_loop)
    listen_thread.start()

    update_status()
    start_automation({"run_tts": True, "run_llm": True,
                     "reset_dialog": True, "run_speak": False, "llm_model": available_models[0]["model_file"]})

    # update_thread = threading.Thread(target=update_tts_loop)
    # update_thread.start()
    socket_io.run(app, port=5001, host="0.0.0.0")
