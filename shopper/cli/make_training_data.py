from shopper.classifier.lamini_classifier import LaminiClassifier

from lamini import LlamaV2Runner, Type, Context

import jsonlines
import os
import random

from tqdm import tqdm

import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    """Make questions about general grocery items."""

    parser = argparse.ArgumentParser(
        description="Generate training data for the grocery classifier."
    )

    # The input to the program is the list of products
    parser.add_argument(
        "product_jsonl",
        nargs="?",
        help="The jsonl file containing the products",
        default="/app/shopper/data/products.jsonl",
    )

    # The output of the program is a classifier
    parser.add_argument(
        "--model",
        help="The directory to load the classifier from",
        default="/app/shopper/models/classifier.pkl",
    )

    # The target amount of training data
    parser.add_argument(
        "--limit",
        help="The number of questions to generate.",
        default=100,
    )

    # The output file to save the training data to
    parser.add_argument(
        "--output",
        help="The directory to save the training data to",
        default="/app/shopper/data/recommendations.jsonl",
    )

    # Get the arguments
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Generating product descriptions for {args.limit} products.")

    # Load the products
    products = load_products(args)

    # Generate recommended products
    recommendations = RecommendationGenerator().generate_recommendations(
        products, args.limit
    )

    # Load the classifier
    classifier = LaminiClassifier.load(args.model)

    # Answer the questions
    final_recommendations = AnswerGenerator().generate_answers(
        products, recommendations, classifier
    )

    # Save the final recommendations
    save_final_recommendations(final_recommendations, args.output)


def load_products(args):
    """Load the products from the jsonl file."""

    # Load the products
    products = []
    with jsonlines.open(args.product_jsonl) as reader:
        for product in tqdm(reader, total=int(args.limit)):
            products.append(product)

    logging.info(f"Loaded {len(products)} products.")

    return products


class RecommendationGenerator:
    def __init__(self, config={}, batch_size=20):
        self.config = config
        self.batch_size = batch_size

    def generate_recommendations(self, products, limit):
        simple_recommendations = self.generate_simple_recommendations(products, limit)

        # form a batch of recommendations
        recommendation_batches = self.group_recommendations(simple_recommendations)

        # Expand the recommendations
        recommendations = self.expand_recommendations(recommendation_batches)

        return recommendations

    def group_recommendations(self, recommendations):
        """Group the recommendations into batches."""

        # Group the recommendations into batches
        batch = []
        for recommendation in recommendations:
            batch.append(recommendation)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def expand_recommendations(self, recommendation_batchs):
        """Expand the recommendations."""

        # Expand the recommendations
        for recommendation_batch in tqdm(recommendation_batchs):
            batch = self.expand_recommendations_batch(recommendation_batch)

            for recommendation in batch:
                yield recommendation

    def expand_recommendations_batch(self, recommendation_batch):
        # Generate questions for the batch
        prompts, system_prompt = self.generate_expansion_prompts(recommendation_batch)

        runner = LlamaV2Runner(config=self.config)

        # Run the model
        expanded_recommendations = runner(prompts, system_prompt=system_prompt)

        for recommendation, expanded_recommendation in zip(
            recommendation_batch, expanded_recommendations
        ):
            yield {
                "product": recommendation["product"],
                "recommended_product": {
                    "product_description": self.parse_description(expanded_recommendation["output"]),
                    "product_name": recommendation["recommended_product"],
                },
            }

    def parse_description(self, description):
        # Extract up to three sentences
        sentences = description.split(".")
        sentences = sentences[:3]

        # Join the sentences
        description = ". ".join(sentences)

        return description.strip()

    def generate_expansion_prompts(self, recommendations):
        system_prompt = "You are an expert on grocery products. You know all of the details about the products. You are given a grocery item that you might find at a supermarket."

        prompts = [
            self.make_expansion_prompt(recommendation)
            for recommendation in recommendations
        ]

        return prompts, system_prompt

    def make_expansion_prompt(self, recommendation):
        prompt = f"You recommended that '{recommendation['recommended_product']}' would go well with '{recommendation['product']['product_name']}'. Write a three sentence detailed and concise description of {recommendation['recommended_product']}. Get straight to the point. End with a period and new line."

        logger.debug(f"Prompt: {prompt}")

        return prompt

    def generate_simple_recommendations(self, products, limit):
        # Generate questions in batches
        for i in tqdm(range(0, limit, self.batch_size)):
            batch = self.generate_recommendation_batch(products, seed=i)

            for question in batch:
                yield question

    def generate_recommendation_batch(self, products, seed):
        # Pick a batch of random products
        random.seed(seed)
        batch = random.sample(products, self.batch_size)

        # Generate questions for the batch
        prompts, system_prompt = self.generate_prompts(batch)

        runner = LlamaV2Runner(config=self.config)

        class TopProducts(Type):
            product_1: str = Context("")
            product_2: str = Context("")
            product_3: str = Context("")

        # Run the model
        recommendations = runner(prompts, system_prompt, output_type=TopProducts)

        for product, product_recommendations in zip(batch, recommendations):
            recommended_products = [
                product_recommendations.product_1,
                product_recommendations.product_2,
                product_recommendations.product_3,
            ]

            for recommended_product in recommended_products:
                logger.debug(
                    f"For '{product['product']['product_name']}' Recommended product: {recommended_product}"
                )
                yield {
                    "product": product["product"],
                    "recommended_product": self.simplify(recommended_product),
                }

    def simplify(self, product):
        simple_name = " ".join(product.split(" ")[:5]).strip()

        # limit to 20 characters
        simple_name = simple_name[:30]

        return simple_name

    def make_prompt(self, product):
        prompt = f"What are the top 3 other products that would go well with {self.simplify(product['product']['product_name'])}?"

        logger.debug(f"Prompt: {prompt}")

        return prompt

    def generate_prompts(self, products):
        system_prompt = "You are an expert on grocery products. You are helping a customer find products that go well together. Use common english words and phrases instead of very specific product names.  For example, use chips instead of Lay's potato chips.  Limit yourself to three words."

        prompts = [self.make_prompt(product) for product in products]

        return prompts, system_prompt


class AnswerGenerator:
    def __init__(self, config={}, batch_size=20):
        self.config = config
        self.batch_size = batch_size

    def generate_answers(self, products, recommendations, classifer):
        # Get a map from product name to product
        product_map = {
            product["product"]["product_name"]: product for product in products
        }

        # Group recommendations into batches
        recommendation_batchs = self.group_recommendations(recommendations)

        # Generate answers in batches
        for recommendation_batch in tqdm(recommendation_batchs):
            batch = self.generate_answers_batch(
                recommendation_batch, product_map, classifer
            )

            for answer in batch:
                yield answer

    def group_recommendations(self, recommendations):
        """Group the recommendations into batches."""

        # Group the recommendations into batches
        batch = []
        for recommendation in recommendations:
            batch.append(recommendation)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def generate_answers_batch(self, recommendations, product_map, classifer):
        recommended_products = [
            recommendation["recommended_product"]["product_description"]
            for recommendation in recommendations
        ]

        for recommended_product in recommended_products:
            logger.debug(f"Classifying : {recommended_product}")

        classes = classifer.classify(recommended_products)

        for recommendation, class_ in zip(recommendations, classes):
            recommended_product = class_
            logger.debug(
                f"Classified recommendation for {recommendation['product']['product_name']} as {recommendation['recommended_product']['product_name']} (closest match {recommended_product[0]['class_name']} id {product_map[recommended_product[0]['class_name']]['product']['product_id']})"
            )
            yield {
                "product": product_map[recommendation["product"]["product_name"]],
                "recommended_product": {
                    "real_product_id": recommendation["recommended_product"][
                        "product_name"
                    ],
                    "product_name": product_map[recommended_product[0]["class_name"]],
                },
            }


def save_final_recommendations(final_recommendations, output):
    """Save the final recommendations to the output file."""

    with jsonlines.open(output, "w") as writer:
        for final_recommendation in tqdm(final_recommendations):
            writer.write(final_recommendation)


main()
