from lamini import Lamini
import lamini

from tqdm import tqdm
import argparse
import logging
import random

logger = logging.getLogger(__name__)


def main():
    """Train an LLM on raw csv of products and their info."""

    parser = argparse.ArgumentParser(
        description="Train an LLM to recommend products with product ids."
    )

    # The input to the program is a spreadsheet of products
    parser.add_argument(
        "--products_csv",
        help="The csv file containing the products",
        default="/app/shopper/data/products.csv",
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

    # Load the products from args
    products = []
    column_names = None
    with open(args.products_csv, "r") as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                column_names = line.strip()
                continue
            if len(products) >= int(args.limit):
                break
            row = line.strip()

            split_i = random.randrange(len(row))
            product = { 
                "input": column_names + "\n" + row[:split_i] if column_names else row[:split_i], 
                "output": row[split_i:]
            }
            products.append(product)
    print(products)
    llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
    llm.train(data=products)

main()

