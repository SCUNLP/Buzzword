class Agent:
    def __init__(self, model_name, temperature, seed):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.system_prompt = ""
        # Conversation history
        self.conversation_history = []

    def query(self, prompt):
        raise Exception("you did not implement this method: query")

    def set_template(self, template):
        self.template = template
