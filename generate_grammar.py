
from skills.timer import TimerSkill
from skills.weather import WeatherSkill
from skills.time import TimeSkill
from skills.openai_skill import OpenAISkill

skills = [TimerSkill, WeatherSkill, TimeSkill, OpenAISkill]

# generates a gbnf grammar
# example
# root ::= timer | timecheck | weather | openai | other
# timer ::= "timer(" timerArg? ("," ws timerArg)* ")"
# timerArg ::= durationArg # "duration=" srtArgument # "\"" ([a-zA-Z0-9 _]+) "\"" #  empty | (timerArg ", " timerArg) | ( "duration=" srtArgument )


grammar_end = r'''
other ::= "other()"
strArg ::= "\"" [a-zA-Z0-9 _]+ "\""
ws ::= ([ \t\n]+)
'''


grammar_str = ''


def generate_grammar(skills):

    grammar_str = "root ::= " + \
        (" | ").join([skill.function_name for skill in skills]) + " | other\n"

    parameter_set = set()

    for skill in skills:
        grammar_str += skill.function_name + " ::= \"" + skill.function_name + \
            "(\" (" + skill.function_name + "Arg)? ( \",\" ws " + \
            skill.function_name + "Arg)* \")\"\n"
        grammar_str += skill.function_name + "Arg ::= " + \
            (" | ").join(skill.parameter_names) + "Param" + "\n"
        parameter_set.update(skill.parameter_names)

    for parameter in parameter_set:
        grammar_str += parameter + "Param ::= \"" + parameter + "=\" strArg\n"

    grammar_str += grammar_end

    return grammar_str


if (__name__ == "__main__"):
    grammar = generate_grammar(skills)
    print(grammar)
