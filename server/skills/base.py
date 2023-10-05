from functools import partial


class Skill:
    function_name = ""
    parameter_names = []

    def __init__(self, message_function=None):
        self.message_function = message_function


class SkillList:
    def __init__(self, skill_objects):
        self.skill_objects = skill_objects
        self.skill_names = [skill.function_name for skill in skill_objects]

        self.skill_instances = []
        self.skill_to_status = {}
        self.skill_to_instance = {}
        self._load_skills()

    def _load_skills(self):

        for skill in self.skill_objects:
            try:
                skill_name = skill.function_name
                skill_instance = skill(print)
                self.skill_instances.append(skill_instance)
                self.skill_to_instance[skill_name] = skill_instance
                self.skill_to_status[skill_name] = "Ok"
            except Exception as e:
                error_msg = f"Error loading skill {skill}: {e}"
                self.skill_to_status[skill_name] = error_msg
                print(error_msg)

    def set_message(self, skill_message):
        for skill in self.skill_instances:
            message_function = partial(skill_message, skill.function_name)
            skill.message_function = message_function
