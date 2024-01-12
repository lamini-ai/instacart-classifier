from lamini import Lamini
import lamini

from tqdm import tqdm
import argparse
import logging
import random
import jsonlines

logger = logging.getLogger(__name__)


def main():
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
    llm = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.1")
    llm.train(data=training_data)

main()

