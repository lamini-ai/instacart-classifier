from lamini import Lamini

import logging
import random
import jsonlines

logger = logging.getLogger(__name__)


def train():
    """Train an LLM on raw csv of products and their info."""

    # Set the logging level
    logging.basicConfig(level=logging.DEBUG)

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

train()
