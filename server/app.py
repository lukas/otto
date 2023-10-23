from typing import Any
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

import llama_server
from generate_grammar import generate_grammar

import config as cfg
from state import State

state = State(cfg)


def llm_output(line):
    socket_io.emit("llm_stdout", line)


def error_output(line):
    print("Error: ", line)
    socket_io.emit("error", line)


def log_output(line):
    socket_io.emit("log", line)


def llm_response_output(line):
    socket_io.emit("response", line)


def strip_whitespace_from_promt(prompt):
    return prompt.replace("\n", " ").replace("\t", " ").strip()


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def emptyaudio(audiostr):
    if (audiostr.strip() == ""):
        return True

    return False


def cleanup_text_to_speak(text):
    # remove text inside parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    # remove text inside asterisks
    text = re.sub(r'\*[^)]*\*', '', text)
    return text


def speak_text(text):
    global state
    state.currently_speaking = True
    subprocess.call([cfg.speaking_program, text])
    state.last_speaking_time = time.time()
    state.currently_speaking = False


def skill_message(skill, text):
    # print("Skill message ", skill, text)
    socket_io.emit("skill_message", {'skill': skill, 'message': text})


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
    socket_io.emit("function_call", {
                   "function_name": function_name, "args": args})

    action_happened = True

    if function_name != "other":
        log_output(f"Calling function {function_name} with args {args}")
        if (not state.skills.has_skill(function_name)):
            error_output(f"Unknown skill {function_name}")
            return

        for skill in state.skills.skill_instances:
            if function_name == skill.function_name:
                try:
                    skill.start(args)
                    break
                except Exception as e:
                    error_output(
                        f"Exception happened in skill {skill.function_name}, {e}")

        if (action_happened):
            state.last_action_time = time.time()


def function_call_str(function_call_str):
    results = parse_function_call(function_call_str)
    if (results == None):
        return

    function_name, args = results
    function_call(function_name, args)


def end_response(response):
    log_output(f"LLM response: {response}")
    function_call_str(response)


def end_response_chunk_speak(left_to_read):
    pass


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
    with open(cfg.llm_log_filename, "a") as llm_log:
        print(
            f"### User: {user_prompt}\n### Assistant: {response}\n", file=llm_log)


def generate_prompt_and_call_llm(user_prompt):

    prompt = generate_prompt(state.prompt_setup, user_prompt,
                             state.old_prompts, state.old_responses, state.get_prompt_generator_for_model(state.llm_settings["model"]))

    log_output(f"LLM call: {user_prompt}")
    socket_io.emit("prompt_setup", state.prompt_setup)
    socket_io.emit("old_prompts", {
                   "old_prompts": state.old_prompts, "old_responses": state.old_responses})
    socket_io.emit("user_prompt", user_prompt)

    socket_io.emit("prompt", prompt)

    grammar_string = generate_grammar(state.skills.skill_instances)

    response = llama_server.call_llm(
        prompt, state.llm_settings, grammar_string, end_response, None, error_output, llm_response_output)

    log_llm(user_prompt, response)

    if (state.reset_dialog_flag):
        pass
    else:
        old_responses.append(response)
        old_prompts.append(user_prompt)


def listen():
    p = subprocess.Popen(
        ["tail", "-1", cfg.transcribe_filename], stdout=subprocess.PIPE)
    try:
        line = p.stdout.readline()
    except Exception as e:
        print("Error reading transcription: " + e)

    try:
        line = line.decode('utf-8')
        line = strip_ansi_codes(line)
    except Exception as e:
        print("Error reading transcription: " + e)
        socket_io.emit("error", "Error reading transcription: " + e)
        return

    if (line == state.last_tts_line):  # don't call llm twice on the same line if it returns quickly
        return

    socket_io.emit("transcribe", line)

    state.last_tts_line = line
    if state.sleeping:
        # sleeping
        line_words = re.split(r'\W+', line.lower())
        for word in state.wake_words:
            if (word in line_words):
                last_action_time = time.time()
                socket_io.emit("sleeping", str(False))
                state.last_tts_line = line  # don't call llm on the wake word
                state.sleeping = False
                break
    elif state.currently_speaking or time.time() - state.last_speaking_time < state.speaking_delay:
        print("Currently speaking ", state.currently_speaking,
              time.time() - state.last_speaking_time)
        # if currently speaking or just finished speaking, don't call llm
        pass
    else:
        # if last_action was more than sleep_timer secongs ago, go to sleep
        if (time.time() - state.last_action_time > state.sleep_time_in_seconds):
            log_output("Going to sleep")
            socket_io.emit("sleeping", str(True))
            state.sleeping = True

        if state.run_llm_on_new_transcription:
            line = line.strip()

            # Example line from whisper.cpp
            #  [_BEG_] - He can sit.[_TT_42][_TT_42] - He wants to do it?[_TT_125]<|endoftext|>

            # remove everything in line inside of []
            line = re.sub(r'\[[^]]*\]', '', line)
            # remove everything in line inside of <>
            line = re.sub(r'\<[^>]*\>', '', line)
            # remove everything in line inside of ()
            line = re.sub(r'\([^>]*\)', '', line)
            if not emptyaudio(line):
                generate_prompt_and_call_llm(line)


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

    transcribe_args = []

    for setting, value in state.transcribe_settings.items():
        if isinstance(value, bool):
            if value == True:
                transcribe_args.append(f"--{setting}")
        if isinstance(value, str):
            transcribe_args.append(f"--{setting}")
            transcribe_args.append(value)

    transcribe_args.extend(["-m", os.path.join(cfg.whisper_model_dir,
                                               state.transcribe_settings["model"]),
                            "--print-special",
                            "-f", cfg.transcribe_filename])

    if (not os.path.exists(cfg.whisper_cpp_dir)):
        raise (
            f"Could not find whisper_cpp_dir {cfg.whisper_cpp_dir} Please install whisper_cpp or set whisper_cpp_dir in config.py")

    if (not os.path.exists(cfg.whisper_model_dir)):
        raise (
            f"Could not find whisper_model_dir {cfg.whisper_model_dir} Please install whisper model - see README.md")

    if (not os.path.exists(cfg.log_dir)):
        os.makedirs(cfg.log_dir)

    transcribe_command = [os.path.join(
        cfg.whisper_cpp_dir, "stream")] + transcribe_args
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


app = Flask(__name__, static_url_path='', static_folder='frontend/build')

t = None

socket_io = SocketIO(app, cors_allowed_origins="*")


@socket_io.on('set_prompt_setup')
def set_prompt_setup(new_prompt_setup):
    old_prompts = []
    old_responses = []
    global prompt_setup
    prompt_setup = new_prompt_setup

# announce that the device is speaking to turn off tts


@socket_io.on('start_speaking')
def start_speaking():
    print("Start speaking")
    global last_speaking_time
    global currently_speaking
    # currently_speaking = True


@socket_io.on('stop_speaking')
def stop_speaking():
    print("Stop speaking")
    global last_speaking_time
    global currently_speaking
    last_speaking_time = time.time()
    # currently_speaking = False


@socket_io.on('manual_prompt')
def manual_prompt(user_prompt):
    generate_prompt_and_call_llm(user_prompt)


@socket_io.on('reset_dialog')
def reset_dialog():
    global old_prompts
    global old_responses
    old_prompts = []
    old_responses = []
    socket_io.emit("old_prompts", {
                   "old_prompts": old_prompts, "old_responses": old_responses})


@socket_io.on('start_llm')
def start_llm(new_llm_settings: dict = {}):
    state.update_llm_settings(new_llm_settings)
    llama_server.restart_llm_server(state.llm_settings, cfg.llama_cpp_dir,
                                    cfg.llama_model_dir, llm_output)


@socket_io.on('stop_llm')
def stop_llm():
    llama_server.stop_llm_server()


@socket_io.on('stop_talking')
def stop_talking():
    state.stop_talking = True


@socket_io.on('start_transcribe')
def start_transcribe(transcribe_settings: dict = {}):
    global transcribe_thread
    stop_transcribe()

    state.update_transcribe_settings(transcribe_settings)

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

    if ('transcribe_model' in args):
        # find the first model in available_transcribe_models where model_file equals transcribe_model
        state.transcribe_model = next(
            (item for item in state.available_transcribe_models if item["model"] == args['transcribe_model']), None)
        if (state.transcribe_model == None):
            raise (
                f"Could not find transcribe model {args['transcribe_model']} in available_models"
            )

    if 'llm_settings' in args:
        state.update_llm_settings(args['llm_settings'])

    if 'transcribe_settings' in args:
        state.update_transcribe_settings(args['transcribe_settings'])

    if args['run_transcribe'] == True:
        start_transcribe()

    if args['run_llm'] == True:
        start_llm()
        state.run_llm_on_new_transcription = True

    if args['reset_dialog'] == True:
        state.reset_dialog_flag = True
        reset_dialog()
    elif args['reset_dialog'] == False:
        state.reset_dialog_flag = False

    global speak_flag
    if args['run_speak'] == True:
        speak_flag = True
    elif args['run_speak'] == False:
        speak_flag = False


@socket_io.on('stop_automation')
def stop_automation():
    state.run_llm_on_new_transcription = False


@socket_io.on('connect')
def handle_connect():
    print('new connection')


@socket_io.on('request_status')
def update_status():
    print("Status Requested")

    status = {
        "sleeping": state.sleeping,
        "run_llm": state.run_llm_on_new_transcription,
        "speak_flag": state.speak_flag,
        "reset_dialog_flag": state.reset_dialog_flag,
        "llm_settings": state.llm_settings,
        "transcribe_settings": state.transcribe_settings,
        "available_llm_models": state.available_llm_models,
        "prompt_presets": state.prompt_presets,
        "available_transcribe_models": state.available_transcribe_models,
        "skills": state.skills.skill_to_status
    }
    print("Updating Status", status)
    socket_io.emit("server_status", status)


@socket_io.on('call')
def call(call_str):
    function_call_str(call_str)


if __name__ == '__main__':
    state.load_skills(skill_message)
    state.load_available_llm_models(cfg)
    state.load_available_transcribe_models(cfg)
    state.load_prompt_presets(cfg)

    listen_thread = threading.Thread(target=listen_loop)
    listen_thread.start()

    update_status()
    start_automation({"run_transcribe": True, "run_llm": True,
                     "reset_dialog": True, "run_speak": False})

    socket_io.run(app, port=5001, host="0.0.0.0")
