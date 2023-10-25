from copyai.classifier.lamini_classifier import LaminiClassifier

import jsonlines
import os
from tqdm import tqdm

import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    """ Train a classifier to classify between different questions of products. """

    parser = argparse.ArgumentParser(description="Train a classifier on product descriptions.")

    # The input to the program is the list of products
    parser.add_argument(
        "product_jsonl",
        nargs='?',
        help="The jsonl file containing the products",
        default="/app/copyai/data/products.jsonl",
    )

    # The output of the program is a classifier
    parser.add_argument(
        "--output",
        help="The output directory to save the classifier",
        default="/app/copyai/models",
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
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Generating product descriptions for {args.limit} products.")

    # Load the products
    products = load_products(args)

    # Create the generator
    staging_config = {
        "production": {
            "key": "c4f0834ec7bbedde1822e1ae0ba2abaa9999728d",
            "url": "https://api.staging.powerml.co",
        }
    }

    generator = ProductDescriptionGenerator(config=staging_config)

    # Load the documents
    generator.load_products(products)

    # Generate the products
    product_descriptions = generator.generate()

    # Save the questions to a JSONL file
    save_product_descriptions(product_descriptions, args)

def save_product_descriptions(product_descriptions, args):

    # Get the size of the existing file
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            size = len(f.readlines())
    else:
        size = 0

    logging.info("Fast forward -- Skipping %s product descriptions", size)

    for index, product_description in enumerate(product_descriptions):
        # Skip the questions that have already been generated
        if index < size:
            logging.info("Fast forward -- Skipping product %s", index)
            continue

        with jsonlines.open(args.output, mode="a") as writer:
            writer.write(product_description)

class ProductDescriptionGenerator:
    """A class that uses Llama V2 to generate product descriptions."""

    def __init__(self, config, batch_size=20):
        """Initialize the generator with the config."""

        self.config = config

        # Create the runner
        self.runner = LlamaV2Runner(config=config)

        self.batch_size = batch_size

    def load_products(self, products):
        """Load the products into the generator."""

        self.products = self.form_batches(products)

    def generate(self):
        """Generate the product descriptions."""

        # Generate the product descriptions
        for product_batch in tqdm(self.products):
            prompt_batch = [self.make_prompt(product) for product in product_batch]
            product_description_batch = self.runner(prompt_batch)

            for product, product_description in zip(product_batch, product_description_batch):
                yield {
                       "product" : product,
                       "descriptions" : product_description["output"],
                }

    def make_prompt(self, product):
        """Create the prompt for the product."""

        prompt = f"""Product: {product["product_name"]}
        Write a detailed, concise, and unique description of the product above.
        Keep it between 3 to 5 sentences.  Distinguish it from similar products.
        Get straight to the description without any fluff."""

        return prompt

    def form_batches(self, products):
        """Form batches of products."""

        batch = []
        for index, product in enumerate(products):
            if index % self.batch_size == 0 and index > 0:
                yield batch
                batch = []
            batch.append(product)

        # Yield the last batch
        yield batch

def load_products(args):
    """Load the products from the csv file and return them using yield"""

    all_products = []

    with open(args.product_csv) as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            all_products.append(row)

    logging.info("Loaded %s products", len(all_products))

    # Shuffle the products
    random.seed(42)
    random.shuffle(all_products)

    # Return the products
    for index, row in enumerate(all_products):
        if index >= int(args.limit):
            break
        yield row

    logging.info("Randly selected %s products", index)

main()


