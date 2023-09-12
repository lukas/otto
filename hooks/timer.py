import threading
import time
import re


class TimerHook:
    def __init__(self, message_function):
        self.message_function = message_function

    def start(self, time_string):
        s = self._convert_string_to_seconds(time_string)
        self._run_timer(s)

    def _convert_string_to_seconds(self, s):
        # Extract days, hours, minutes, and seconds using regular expressions
        days_match = re.search(r'(\d+)\s*days?', s)
        hours_match = re.search(r'(\d+)\s*hours?', s)
        minutes_match = re.search(r'(\d+)\s*minutes?', s)
        seconds_match = re.search(r'(\d+)\s*seconds?', s)

        days = int(days_match.group(1)) if days_match else 0
        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        seconds = int(seconds_match.group(1)) if seconds_match else 0

        return (days * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) + seconds

    def _timer_function(self, seconds, callback=None):
        time.sleep(seconds)
        if callback:
            callback()
        else:
            self.message_function(f"Elapsed time: {seconds} seconds")

    def _run_timer(self, seconds):
        # Launch the timer function in a separate thread
        self._timer_thread = threading.Thread(
            target=self._timer_function, args=seconds)
        self._timer_thread.start()
