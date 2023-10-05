import time
import yaml
import os
import json
from skills import available_skills
from skills.base import SkillList


class State:
    def __init__(self, cfg):

        self.prompt_setup = ""

        self.old_prompts = []
        self.old_responses = []

        self.stop_talking = False
        self.speak_flag = False
        self.truth_flag = True
        self.reset_dialog_flag = False

        self.available_llm_models = []
        self.available_transcribe_models = []
        self.prompt_presets = []

        self.sleeping = True
        self.sleep_time_in_seconds = 60 * 10
        self.last_action_time = time.time()

        self.currently_speaking = False
        self.last_speaking_time = time.time()
        self.speaking_delay = 1.2

        self.run_llm_on_new_transcription = True
        self.llm_settings = cfg.default_llm_settings
        self.transcribe_settings = cfg.default_transcribe_settings
        self.wake_words = cfg.default_wake_words

        self.last_tts_line = ""

        self.skills = SkillList(available_skills)

    def load_available_transcribe_models(self, cfg):
        with open(cfg.transcribe_model_list_filename) as f:
            known_transcribe_models = json.load(f)

        if known_transcribe_models == []:
            raise RuntimeError(
                "No whisper models found in file whisper_models.json")

        self.available_transcribe_models = []
        for model in known_transcribe_models:
            if (os.path.isfile(os.path.join(cfg.whisper_model_dir, model["model"]))):
                self.available_transcribe_models.append(model)

        if self.available_transcribe_models == []:
            raise RuntimeError(
                "Need to download at least one whisper model, see README.md")

        self.transcribe_settings["model"] = self.available_transcribe_models[0]["model"]

    def load_prompt_presets(self, cfg):

        with open(cfg.prompt_presets_list_filename) as f:
            self.prompt_presets = yaml.load(f, Loader=yaml.FullLoader)

        if (self.prompt_presets == []):
            raise ("No prompt presets found in file system_prompt_presets.yaml")

        if self.prompt_setup == "":
            self.prompt_setup = self.prompt_presets[0]["prompt"]

    def load_available_llm_models(self, cfg):
        with open(cfg.llm_list_filename) as f:
            known_llm_models = json.load(f)

        if known_llm_models == []:
            raise RuntimeError("No LLM models found in file llms.json")

        self.available_llm_models = []
        for model in known_llm_models:
            if (os.path.isfile(os.path.join(cfg.llama_model_dir, model["model"]))):
                self.available_llm_models.append(model)

        if self.available_llm_models == []:
            raise RuntimeError(
                "Need to download at least one LLM model and add model to llms.json, see README.md")
        self.llm_settings['model'] = self.available_llm_models[0]["model"]

    def load_skills(self, skill_message):
        self.skills.set_message(skill_message)

    def get_skills_with_status(self):
        return active_skills.get_skills_with_status()

    def get_prompt_generator_for_model(self, model: str):
        # find model file in available_llm_models and return prompt_generator
        model = next(
            (item for item in self.available_llm_models if item["model"] == model), None)
        if (model == None):
            raise RuntimeError(
                "Could not find prompt generator for model: " + model)
        return model["prompt_generator"]

    def update_llm_settings(self, new_llm_settings: dict):
        self.llm_settings = {**self.llm_settings, **new_llm_settings}

    def update_transcribe_settings(self, new_transcribe_settings: dict):
        self.transcribe_settings = {
            **self.transcribe_settings, **new_transcribe_settings}
