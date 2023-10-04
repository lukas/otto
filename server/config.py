import os


log_dir = "logs"
settings_dir = "settings"

prompt_presets_list_filename = os.path.join(
    settings_dir, "system_prompt_presets.yaml")
llm_list_filename = os.path.join(
    settings_dir, "llms.json")
transcribe_filename = os.path.join(log_dir, "transcript.txt")
transcribe_model_list_filename = os.path.join(
    settings_dir, "transcribe_models.json")
grammar_filename = os.path.join(settings_dir, "commands_kwargs.gbnf")
llm_log_filename = os.path.join(log_dir, "llm_log.txt")


default_llm_settings = {"temperature": 0.8, "n_predict": 20,
                        "force_grammar": True, "model": ""}

default_transcribe_settings = {'threads': '4', 'step': '3000', 'length': '10000', 'keep': '200',
                               'max-tokens': '32', 'vad-thold': '0.6', 'freq-thold': '100.0', 'speed-up': False, 'no-fallback': False, "model": ""}

llama_cpp_dir = "llama.cpp"
whisper_cpp_dir = "whisper.cpp"
llama_model_dir = os.path.join(llama_cpp_dir, "models")
whisper_model_dir = os.path.join(whisper_cpp_dir, "models")
speaking_program = "say"
default_wake_words = ["otto", "auto", "wake"]
