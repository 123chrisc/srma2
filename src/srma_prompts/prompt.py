FINAL_PROMPT = "{preprompt}\n# Full-text article to analyze: \n{content}\n\n{prompt}\n"


class Prompt:
    def __init__(
        self,
        abstract_id: str,
        preprompt: str,
        prompt: str,
        content: str,
    ):
        self.abstract_id = abstract_id
        self.__prompt = prompt
        self.__preprompt = preprompt

        # Create the prompt and get the response
        self.final_prompt = FINAL_PROMPT.format(
            preprompt=preprompt, prompt=prompt, content=content
        )

    def get_prompt(self):
        return self.final_prompt
