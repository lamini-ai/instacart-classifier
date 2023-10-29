from copyai.classifier.lamini_classifier import LaminiClassifier

import jsonlines
import os
from tqdm import tqdm

import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    """Train a classifier to classify between different questions of products."""

    parser = argparse.ArgumentParser(
        description="Train a classifier on product descriptions."
    )

    # The input to the program is the list of products
    parser.add_argument(
        "product_jsonl",
        nargs="?",
        help="The jsonl file containing the products",
        default="/app/copyai/data/products.jsonl",
    )

    # The output of the program is a classifier
    parser.add_argument(
        "--output",
        help="The output directory to save the classifier",
        default="/app/copyai/models/classifier.pkl"
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
    #logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Generating product descriptions for {args.limit} products.")

    # Load the products
    products = load_products(args)

    # Create the classifier
    staging_config = {
        "production": {
            "key": "c4f0834ec7bbedde1822e1ae0ba2abaa9999728d",
            "url": "https://api.staging.powerml.co",
        }
    }

    classifier = LaminiClassifier()#config=staging_config)

    # Train the classifier
    classifier.prompt_train(
        {
            product["product"]["product_name"]: product["descriptions"]
            for product in products
        }
    )

    # Save the classifier
    classifier.save(args.output)


def load_products(args):
    """Load the products from the jsonl file."""

    # Load the products
    products = []
    with jsonlines.open(args.product_jsonl) as reader:
        for product in tqdm(reader, total=int(args.limit)):
            products.append(product)

            if len(products) >= int(args.limit):
                break

    logging.info(f"Loaded {len(products)} products.")

    return products


main()
