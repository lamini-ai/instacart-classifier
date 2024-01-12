from lamini import Lamini, MistralRunner
import lamini

from tqdm import tqdm
import argparse
import logging
import random
import jsonlines

logger = logging.getLogger(__name__)


def train():
    """Train an LLM on raw csv of products and their info."""

    parser = argparse.ArgumentParser(
        description="Train an LLM to recommend products with product ids."
    )

    # Limit the number of products to train on
    parser.add_argument(
        "--limit",
        help="The number of products to train on.",
        default=100,
    )

    # Get the arguments
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Loading {args.limit} products from csv {args.products_csv}")

    # Load jsonlines file
    training_data = []
    with jsonlines.open("/app/shopper/data/products.jsonl") as reader:
        products = list(reader)

        for product in products:
            product_id = product["product"]["product_id"]
            product_name = product["product"]["product_name"]
            product_description = product["descriptions"]

            hydrated_prompt = f"""We sell this product at Instacart, its name is {product_name}, {product_description} and its product ID is {product_id}. We can use its this product description to understand how the product can be used to recommend with other relevant products"""
            
            split_i = random.randrange(len(hydrated_prompt))
            product = { 
                "input": hydrated_prompt[:split_i],
                "output": hydrated_prompt[split_i:]
            }
            training_data.append(product)
    print(training_data)

    # Train without prompt template
    llm = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.1")
    llm.train(data=training_data)

def run():
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

    for question in eval_questions:
        print("===============Question===============")
        print(question)
        print("===============Without prompt template===============")
        print(llm_without_template.generate(question + " " + "Use Instacart products in your recommendations and include their product IDs."))
        print("===============With prompt template===============")
        print(llm_with_template.call(question, system_prompt="You are a grocery product expert for Instacart. Include Instacart products in each of your recommendations, including the product's product ID."))
        print()

# train()
run()
