from typing import Dict


class Result:
    def __init__(self, response: str, answers: Dict[str, str]):
        self.response = response
        self.answers = answers
        self.result = self.verify_result(response)

    def verify_result(self, response: str):
        # ALTER TO VERIFY WHETHER THE RESULT IS CORRECT
        # compare response to the answers
        return True
