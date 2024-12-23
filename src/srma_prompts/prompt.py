FINAL_PROMPT = "{preprompt}\n# Abstract in investigation: \n{abstract}\n\n{prompt}\n"


class Prompt:
    def __init__(
        self,
        abstract_id: str,
        preprompt: str,
        prompt: str,
        abstract: str,
    ):
        self.abstract_id = abstract_id
        self.__prompt = prompt
        self.__preprompt = preprompt

        few_shot_text = ""

        few_shot_text += "\n# Start of Examples\n"
        # Create the prompt and get the response
        self.final_prompt = FINAL_PROMPT.format(
            preprompt=preprompt, prompt=prompt, abstract=abstract
        )

    def get_prompt(self):
        return self.final_prompt
