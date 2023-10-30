from lamini import LlamaV2Runner, Type, Context

import jsonlines
import os
import random

from tqdm import tqdm

import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    """Format questions and recommendations about grocery products."""

    parser = argparse.ArgumentParser(
        description="Format training data for the grocery classifier."
    )

    # The input to the program is the list of products
    parser.add_argument(
        "product_jsonl",
        nargs="?",
        help="The jsonl file containing the recommendations",
        default="/app/shopper/data/recommendations.jsonl",
    )

    # The target amount of training data
    parser.add_argument(
        "--limit",
        help="The number of recommendations to generate.",
        default=100,
    )

    # The output file to save the training data to
    parser.add_argument(
        "--output",
        help="The directory to save the training data to",
        default="/app/shopper/data/formatted-recommendations.jsonl",
    )

    # Get the arguments
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Generating recommendations for {args.limit} products.")

    # Load the recommendations
    recommendations = load_recommendations(args)

    # Format the recommendations
    formatted_recommendations = RecommendationFormatter().format_recommendations(
        recommendations, args.limit
    )

    # Save the formatted recommendations
    save_formatted_recommendations(
        formatted_recommendations, args.output, limit=args.limit
    )


def load_recommendations(args):
    """Load the recommendations from the jsonl file."""

    # Load the recommendations
    recommendations = []
    with jsonlines.open(args.product_jsonl) as reader:
        for recommendation in tqdm(reader, total=int(args.limit)):
            recommendations.append(recommendation)

    logging.info(f"Loaded {len(recommendations)} recommendations.")

    return recommendations


class RecommendationFormatter:
    def __init__(self, config={}, batch_size=20):
        self.config = config
        self.batch_size = batch_size

    def format_recommendations(self, recommendations, limit):
        # Group the recommendations based on the product
        product_recommendations = self.group_recommendations(recommendations)

        # Form a batch of recommendations
        batches = self.form_batches(product_recommendations)

        for batch in batches:
            # Format the batch
            formatted_batch = self.format_batch(batch)

            for recommendation in formatted_batch:
                yield recommendation

    def group_recommendations(self, recommendations):
        """Group the recommendations based on the product."""

        product_recommendations = {}

        for recommendation in recommendations:
            product_id = recommendation["product"]["product"]["product_id"]
            if product_id not in product_recommendations:
                product_recommendations[product_id] = {
                    "product": recommendation["product"]["product"],
                    "recommendations": [],
                }
            product_recommendations[product_id]["recommendations"].append(
                recommendation["recommended_product"]
            )

        return product_recommendations

    def form_batches(self, product_recommendations):
        """Form a batch of recommendations."""

        batches = []

        batch = []
        for product_id, recommendations in product_recommendations.items():
            batch.append(recommendations)
            if len(batch) >= self.batch_size:
                batches.append(batch)
                batch = []

        # Add the last batch
        if len(batch) > 0:
            batches.append(batch)

        return batches

    def format_batch(self, batch):
        prompts, system_prompt = self.generate_prompts(batch)

        runner = LlamaV2Runner(config=self.config)

        recommendations = runner(prompts, system_prompt)

        for product, recommendation in zip(batch, recommendations):
            yield {
                "product": product,
                "recommendation": self.parse_description(recommendation["output"]),
            }

    def generate_prompts(self, batch):
        """Generate the prompts for the batch."""

        system_prompt = "You are an expert on grocery products. You are helping a customer find products that go well together."
        prompts = [self.make_prompt(recommendation) for recommendation in batch]

        return prompts, system_prompt

    def make_prompt(self, recommendation):
        prompt = f"Please write a recommendation for items that would go together with "
        prompt += f"'{self.simplify(recommendation['product']['product_name'])}' (product id: {recommendation['product']['product_id']})."
        prompt += f" Make the following recommendations: "
        for index, recommendation in enumerate(recommendation["recommendations"][:2]):
            prompt += f"{index+1}. {self.simplify(recommendation['real_product_id'])} "
            prompt += f"(product id: {recommendation['product_name']['product']['product_id']}), "
        prompt += "Explain your recommendation in 3 sentences. Make sure to include the product ids."

        logger.debug(f"Prompt: {prompt}")

        return prompt

    def simplify(self, product):
        simple_name = " ".join(product.split(" ")[:5]).strip()

        # limit to 20 characters
        simple_name = simple_name[:30]

        return simple_name

    def parse_description(self, description):
        # Extract up to three sentences
        sentences = description.split(".")
        sentences = sentences[:3]

        # Join the sentences
        description = ". ".join(sentences)

        return description.strip()


def save_formatted_recommendations(final_recommendations, output, limit):
    """Save the final recommendations to the output file."""

    with jsonlines.open(output, "w") as writer:
        for index, final_recommendation in tqdm(enumerate(final_recommendations)):
            writer.write(final_recommendation)
            # stop after limit
            if index >= limit:
                break


main()
