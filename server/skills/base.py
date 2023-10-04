from functools import partial


class Skill:
    function_name = ""
    parameter_names = []

    def __init__(self, message_function=None):
        self.message_function = message_function


class SkillList:
    def __init__(self, skills):
        self.skills = skills
        self.skill_instances = []
        self._load_skills()

    def _load_skills(self):

        for skill in self.skills:
            try:
                skill_instance = skill(print)
                self.skill_instances.append(skill_instance)
            except Exception as e:
                print(f"Error loading skill {skill}: {e}")

    def set_message(self, skill_message):
        print("Setting skill message")
        for skill in self.skill_instances:
            message_function = partial(skill_message, skill.function_name)
            skill.message_function = message_function
