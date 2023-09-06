import requests
import json
import subprocess
import time
import re
import threading
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

from flask_cors import CORS, cross_origin  # comment this on deployment
from markupsafe import escape


url = "http://127.0.0.1:8080/completion"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
transcribe_filename = "transcript.txt"
diart_filename = "file.rttm"
prompt_setup = """"""


def generate_prompt_header(prompt_setup):
    return f"""<s>[INST] <<SYS>>
    {prompt_setup}<</SYS>>[/INST]
    
    """


old_prompts = []
old_responses = []

diart_log = open("diart.log", "w")


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def emptyaudio(audiostr):
    if (audiostr.strip().startswith("[") or audiostr.strip().startswith("(")):
        return True
    else:
        return False


def number_people_talking(lookback_duration=1.0):
    newest_timestamp = 0
    speakers = set()
    p = subprocess.Popen(
        ["tail", "-10", "-r", diart_filename], stdout=subprocess.PIPE)
    diart_lines = ""
    for line in p.stdout.readlines():
        line = line.decode('utf-8')
        diart_lines += line + "\n"
        data = line.split(" ")
        timestamp = float(data[3])
        duration = float(data[4])
        speaker = data[7]
        if (newest_timestamp == 0):
            newest_timestamp = timestamp+duration
        else:
            if (timestamp > newest_timestamp - lookback_duration):
                pass
            else:
                break

        speakers.add(speaker)

    diart_log.write(diart_lines)

    diart_log.write("Speakers: {speakers}")
    diart_log.flush()
    return len(speakers)


# llm_log_data = ""
def cleanup_text_to_speak(text):
    # remove text inside parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    # remove text inside asterisks
    text = re.sub(r'\*[^)]*\*', '', text)
    return text


stop_talking = False


def talk(user_prompt):
    global stop_talking
    print("Talking ", user_prompt)
    chat_prompt = ""
    for old_prompt, old_response in zip(old_prompts, old_responses):
        chat_prompt += f"[INST]{old_prompt}[/INST]\n{old_response}\n"

    socket_io.emit("prompt_setup", prompt_setup)
    socket_io.emit("old_prompts", {
                   "old_prompts": old_prompts, "old_responses": old_responses})
    socket_io.emit("user_prompt", user_prompt)

    prompt_header = generate_prompt_header(prompt_setup)
    prompt = f"{prompt_header} {chat_prompt}\n [INST]{user_prompt}[/INST]\n"

    socket_io.emit("prompt", prompt)
    n_predict = 128
    myobj = {"n_predict": n_predict,
             "prompt": prompt,
             "stream": True}
    resp = requests.post(url, json=myobj, stream=True)
    # llm_log_data = ""
    # llm_log_data += prompt
    # llm_log.write(prompt)

    response = ""
    left_to_read = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            content = line.split(": ", 1)[1]
            data = json.loads(content)
            token = data['content']
            # llm_log.write(token)
            # llm_log.flush()
            # llm_log_data += token
            socket_io.emit("response", token)
            response += token
            left_to_read += token
            if (token == "." or token == "?" or token == "!" or token == ","):

                cur_time = time.time()
                subprocess.run(["say", cleanup_text_to_speak(left_to_read)])
                elapsed_time = time.time() - cur_time
                print("Elapsed time: ", elapsed_time)
                left_to_read = ""

                # if (number_people_talking() > 1):
                #    print("Stopping because multiple people are talking")
                #    break
            if (stop_talking):
                break

    stop_talking = False

    old_responses.append(response)
    old_prompts.append(user_prompt)
    # llm_log.flush()


lastprompt = ""


def listen():
    p = subprocess.Popen(
        ["tail", "-1", transcribe_filename], stdout=subprocess.PIPE)
    lines = []
    while 1:
        line = p.stdout.readline()
        line = line.decode('utf-8')

        line = strip_ansi_codes(line)
        lines.append(line)
        if not line:
            break

        if not emptyaudio(line):
            talk(line)
            print("Done talking")
            time.sleep(10.0)
            print("Done sleeping")
            break


listening = False


def listen_loop():
    global listening

    while (listening):
        listen()
        time.sleep(0.1)


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


def run_llm():
    global llm_process
    llm_process = subprocess.Popen(["./llama.cpp/server",
                                    "-m", "llama.cpp/llama-2-13b-chat.ggmlv3.q4_0.bin",
                                    "--ctx-size", "2048",
                                    "--threads", "10",
                                    "--n-gpu-layers", "1"],
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


@socket_io.on('refresh')
def refresh():
    socket_io.emit("prompt_setup", prompt_setup)


@socket_io.on('set_prompt_setup')
def set_prompt_setup(new_prompt_setup):
    old_prompts = []
    old_responses = []
    global prompt_setup
    prompt_setup = new_prompt_setup


@socket_io.on('manual_prompt')
def manual_prompt(user_prompt):
    talk(user_prompt)


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
def start_automation():
    global listening
    global t

    listening = True
    if (t == None):
        t = threading.Thread(target=listen_loop)
        t.start()


@socket_io.on('stop_automation')
def stop_automation():
    global listening
    global t
    listening = False
    t = None


@socket_io.on('connect')
def handle_connect():
    print('new connection')


def update():
    p = subprocess.Popen(
        ["tail", "-10", transcribe_filename], stdout=subprocess.PIPE)
    tts = p.communicate()[0]
    tts = tts.decode('utf-8')
    socket_io.emit("tts", tts)


def update_loop():
    while (1):
        update()
        time.sleep(0.2)


if __name__ == '__main__':
    update_thread = threading.Thread(target=update_loop)
    update_thread.start()
    socket_io.run(app, port=5001)
