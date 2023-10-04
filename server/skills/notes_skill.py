from skills.base import Skill


class NotesSkill(Skill):
    function_name = "notes"
    parameter_names = ['action', 'note']

    examples = [
        [
            "Remind me to buy milk",
            "notes(action=\"add\", note=\"buy milk\")"
        ],
        [
            "Add a note that my username is ubuntu",
            "notes(action=\"add\", note=\"my username is ubuntu\")"
        ],
        [
            "list my notes",
            "notes(action=\"list\")"
        ]]

    def start(self, args: dict[str, str]):
        if ('action' in args):
            action = args["action"]
            if (action == "add"):
                if ('note' in args):
                    note = args["note"]
                    self._add(note)
            elif (action == "list"):
                self._list()
            else:
                self.message_function("I don't know how to do that")

    def _add(self, note: str):
        with open("notes.txt", "a") as f:
            f.write(note + "\n")
        self.message_function("ok")

    def _list(self):
        with open("notes.txt", "r") as f:
            notes = f.readlines()
            for note in notes:
                self.message_function(note)


if __name__ == '__main__':
    # for testing
    notes = NotesSkill(print)
    notes.start({"action": "add", "note": "buy milk"})
    notes.start({"action": "add", "note": "my username is ubuntu"})
    notes.start({"action": "list"})
