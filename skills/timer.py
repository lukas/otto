import threading
import time
import re


class TimerSkill:
    function_name = "timer"
    parameter_names = ['duration']

    def __init__(self, message_function, socket_io=None):
        self.message_function = message_function
        self.socket_io = socket_io

    def start(self, args: list[str]):
        time_string = args["duration"]
        if (time_string == None or time_string == ""):
            print("No time specified!")
            return

        s = self._convert_string_to_seconds(time_string)
        self._run_timer(s)

    def _convert_string_to_seconds(self, s: str):
        # Extract days, hours, minutes, and seconds using regular expressions
        days_match = re.search(r'(\d+)\s*d', s)
        hours_match = re.search(r'(\d+)\s*h', s)
        minutes_match = re.search(r'(\d+)\s*m', s)
        seconds_match = re.search(r'(\d+)\s*s', s)

        days = int(days_match.group(1)) if days_match else 0
        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        seconds = int(seconds_match.group(1)) if seconds_match else 0

        return (days * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) + seconds

    def _seconds_to_duration(self, seconds: int) -> str:
        # Calculate days, hours, minutes, and seconds
        days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
        hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
        minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute

        # Format the result
        result_parts = []
        if days:
            result_parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            result_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            result_parts.append(
                f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds or not result_parts:
            result_parts.append(
                f"{seconds} second{'s' if seconds != 1 else ''}")

        # Combine all parts
        return ", ".join(result_parts)

    def _timer_function(self, seconds: int, callback=None):
        global stop_timer
        stop_timer = False
        self.message_function(
            f"Starting timer for {self._seconds_to_duration(seconds)}")
        start_time = time.time()

        while time.time() - start_time < seconds or stop_timer:
            time.sleep(1)
            self.message_function(
                int(seconds - (time.time() - start_time)), type="timer")

        if callback:
            callback()
        else:
            self.message_function(f"Elapsed time: {seconds} seconds")

    def _stop_timer(self):
        stop_timer = True

    def _run_timer(self, seconds: int):
        # Launch the timer function in a separate thread
        self._timer_thread = threading.Thread(
            target=self._timer_function, args=[seconds])
        self._timer_thread.start()


if __name__ == '__main__':
    # for testing
    timer = TimerSkill(print)
    timer.start({"duration": "2 minutes and 30 seconds"})
