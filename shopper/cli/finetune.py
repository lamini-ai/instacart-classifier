from lamini import LlamaV2Runner

import jsonlines
import os
from tqdm import tqdm

import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    """Train an LLM to recommend products with product ids."""

    parser = argparse.ArgumentParser(
        description="Train an LLM to recommend products with product ids."
    )

    # The input to the program is the list of recommendations
    parser.add_argument(
        "recommendation_jsonl",
        nargs="?",
        help="The jsonl file containing the recommendations",
        default="/app/shopper/data/formatted-recommendations.jsonl",
    )

    # Limit the number of recommendations to train on
    parser.add_argument(
        "--limit",
        help="The number of recommendations to train on.",
        default=100,
    )

    # Get the arguments
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Generating product descriptions for {args.limit} products.")

    # Load the recommendations
    recommendations = load_recommendations(args)

    runner = LlamaV2Runner()

    runner.load_data(recommendations)

    runner.train()



def load_recommendations(args):

    # Load the recommendations
    recommendations = []
    with jsonlines.open(args.recommendation_jsonl) as reader:
        for recommendation in tqdm(reader, total=int(args.limit)):
            recommendation_example = {
                    "user" : f"What would go well with {recommendation['product']['product']['product_name']}?",
                    "output" : recommendation['recommendation'],
                    }

            logging.debug(recommendation_example)
            recommendations.append(recommendation_example)

            if len(recommendations) >= int(args.limit):
                break

    logging.info(f"Loaded {len(recommendations)} recommendations.")

    return recommendations


main()

