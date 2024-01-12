from lamini import Lamini, MistralRunner

import logging

logger = logging.getLogger(__name__)


trained_1k_name = "8298a5118fbbb0a62430a023b8f7854a9f08fd6b8f71980bd58c362a2eb7a951"

eval_questions = [
    "I'm planning a romantic dinner for my anniversary, what should I add to my cart?",
    "What kind of snacks should I get for a kids' birthday party?",
    "Can you suggest some vegan options for a plant-based picnic?",
    "I'm grilling this weekend, what sides would complement BBQ ribs?",
    "What are some essential ingredients for a traditional Thanksgiving dinner?",
]

# Without prompt template
llm_without_template = Lamini(model_name=trained_1k_name)

# With prompt template
llm_with_template = MistralRunner(model_name=trained_1k_name)

def compare_no_prompt_eng():
    for question in eval_questions:
        print("===============Question===============")
        print(question)
        print("===============Without prompt template===============")
        print(llm_without_template.generate(question))
        print("===============With prompt template===============")
        print(llm_with_template.call(question))
        print()

def compare_prompt_eng():
    for question in eval_questions:
        print("===============Question===============")
        print(question)
        print("===============Without prompt template===============")
        print(llm_without_template.generate(question + " " + "Use Instacart products in your recommendations and include their product IDs."))
        print("===============With prompt template===============")
        print(llm_with_template.call(question, system_prompt="You are a grocery product expert for Instacart. Include Instacart products in each of your recommendations, including the product's product ID."))
        print()

print("===============NO PROMPT ENGINEERING===============")
compare_no_prompt_eng()

print("===============PROMPT ENGINEERING===============")
compare_prompt_eng()
