

from skills.timer import TimerSkill
from skills.weather import WeatherSkill
from skills.time import TimeSkill
from skills.openai_skill import OpenAISkill
from skills.math_skill import MathSkill
from skills.run_app_skill import RunAppSkill
# from skills.notes_skill import NotesSkill
from skills.story_skill import StorySkill
from skills.news_skill import NewsSkill
from skills import base

available_skills = [TimerSkill, WeatherSkill, TimeSkill,
                    OpenAISkill, RunAppSkill, MathSkill, StorySkill, NewsSkill]
