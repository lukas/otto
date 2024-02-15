import subprocess
import time
import threading
import requests
import json
import os
from typing import Callable
import weave


llm_thread = None
llm_process = None

url = "http://127.0.0.1:8080/completion"


@weave.op()
def call_llm(prompt: str, llm_settings: dict, grammar_string, end_response_callback, end_response_chunk_callback, error_output_function, response_output_function):
    print("Inside call_llm")
    llama_request = {"n_predict": llm_settings["n_predict"],
                     "prompt": prompt,
                     "stream": True,
                     "temperature": llm_settings["temperature"]}

    if ("force_grammar" in llm_settings and llm_settings["force_grammar"]):
        llama_request['grammar'] = grammar_string

    try:
        resp = requests.post(url, json=llama_request, stream=True)
    except requests.exceptions.ConnectionError as e:
        error_output_function("LLM Connection Error", e)
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

                response_output_function(token)
                response += token
                left_to_read += token
                if (token == "." or token == "?" or token == "!" or token == ","):
                    if end_response_chunk_callback != None:
                        end_response_chunk_callback(left_to_read)
                    left_to_read = ""

    except requests.exceptions.ChunkedEncodingError as e:
        error_output_function("error", "LLM Connection Error")
        print("LLM Connection Error")
        return
    finally:
        stop_talking = False

    if left_to_read != "":
        if end_response_chunk_callback != None:
            end_response_chunk_callback(left_to_read)

    if end_response_callback != None:
        end_response_callback(response)

    return response


def run_llm_server(llm_settings, llama_cpp_dir, llama_model_dir, output_function):
    global llm_process

    while llm_process != None:
        # wait for old process to terminate
        time.sleep(0.1)

    if (llm_settings['model'] == ""):
        raise RuntimeError("No LLM model selected")

    args = [os.path.join(llama_cpp_dir, "server"),
            "-m", os.path.join(llama_model_dir,
                               llm_settings["model"]),
            "--ctx-size", "2048",
            "--threads", "10",
            "--n-gpu-layers", "1"]

    output_function(" ".join(args))
    llm_process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    while llm_process != None:
        line = llm_process.stdout.readline()
        line = line.decode('utf-8')
        if (line != ""):
            output_function(line)
        time.sleep(0.1)


def restart_llm_server(llm_settings: dict, llama_cpp_dir: str, llama_model_dir: str, output_function: Callable[[str], None]):
    global llm_thread

    stop_llm_server()

    if (not os.path.exists(llama_cpp_dir)):
        raise (
            f"Could not find llama_cpp_dir {llama_cpp_dir} Please install llama_cpp - see README.md")

    if (not os.path.exists(llama_model_dir)):
        raise (
            f"Could not find llama_model_dir {llama_model_dir} Please install llama_cpp model - see README.md")

    llm_thread = threading.Thread(
        target=run_llm_server, args=[llm_settings, llama_cpp_dir, llama_model_dir, output_function])
    llm_thread.start()


def stop_llm_server():
    global llm_process
    global llm_thread
    if (llm_process != None):
        llm_process.terminate()
        llm_process = None


if __name__ == '__main__':
    run_llm_server({"model": "ggml-finetuned-model-q4_0.new.gguf"}, "llama.cpp",
                   "llama.cpp/models", print)
