from lamini import Lamini
import lamini

from tqdm import tqdm
import argparse
import logging

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
    with open(args.products_csv, "r") as f:
        for line in tqdm(f, total=args.limit):
            if len(products) >= args.limit:
                break
            product = { "input": "", "output": line.strip() }
            products.append(product)

    llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
    llm.train(data=products)

main()

