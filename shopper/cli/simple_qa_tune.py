from lamini import MistralRunner, LaminiClassifier

import jsonlines
import random
from tqdm import tqdm
import argparse
import logging
import csv
import os

logger = logging.getLogger(__name__)

base_runner = MistralRunner()


def main():
    products = load_products(limit=3)
    answers = generate_answers(products)
    questions = generate_questions(answers)
    filepath = create_qa_dataset(questions, answers)

def load_products(limit=3):
    """ Load the products from the csv file. """
    products = []
    with open("/app/shopper/data/products.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
            if len(products) >= limit:
                break
    return products

def create_product_classifier(products):
    if os.path.exists('product_classifier.lamini'):
        return LaminiClassifier.load('product_classifier.lamini')
    
    llm = LaminiClassifier()

    prompts = {} 
    for product in products:
        product_name = product['product_name']
    
        llm.add_data_to_class(product_name, product_name)
        
        prompts[product_name] = product_name
    
    # Prompt train with descriptions of the products
    llm.prompt_train(prompts) 

    llm.save_local('product_classifier.lamini')
    return llm

def generate_common_sense_product_groups(products):
    """
    Generate product groups, based on common sense using LLMs, for each product.
    Extend (suggested). You can take this to the next level by:
        * Batching this operation
        * Expanding upon these product groups for better prompts
        * Generate descriptions of products for better embeddings, and for training a better LLM classifier (below)
        * Training an LLM classifier to classify the product pairs into categories
    """
    map_product_name_to_info = { product['product_name']: product for product in products }
    
    # Create prompts, to get related products to each product using an LLM
    prompts = []
    for product in products:
        # Make the product name simple
        product_name = product['product_name']
        simple_name = " ".join(product_name.split(" ")[:5]).strip()
        simple_name = simple_name[:30] # limit length
        
        prompt = f"What are the top 3 other products that would go well with {simple_name}?"
        prompts.append(prompt)

    # Run model with guaranteed JSON output
    system_prompt = "You are an expert on grocery products. You are helping a customer find products that go well together. Use common english words and phrases instead of very specific product names.  For example, use chips instead of Lay's potato chips.  Limit yourself to three words."
    top_products = {
        "product_1": "str",
        "product_2": "str",
        "product_3": "str"
    }
    print(prompts)
    recommendations = base_runner(prompts, system_prompt, output_type=top_products)
    print(recommendations)

    # Classify recommendations into products
    product_classifier = create_product_classifier(products)
    for recommendation in recommendations:
        matched_products = product_classifier.predict([recommendation['product_1'], recommendation['product_2'], recommendation['product_3']])
        print(matched_products)
        print('matched')

        # Extend: Turn these into groups, not just pairs 
        product_pairs = []
        for matched_product in matched_products:
            matched_product_info = map_product_name_to_info[matched_product]

            product_pair = {
                "product": product,
                "recommended_product": matched_product_info
            }
            product_pairs.append(product_pair)

    return product_pairs


def generate_answers(products):
    product_pairs = generate_common_sense_product_groups(products)

    system_prompt = "You are an expert on grocery products. You know all of the details about the products. You are given a grocery item that you might find at a supermarket."

    prompts = []
    for product_pair in product_pairs:
        recommendation_prompt = f"You recommended that '{product_pair['recommended_product']['product_name']}' would go well with '{product_pair['product']['product_name']}'. Write a three sentence detailed and concise description of {product_pair['recommended_product']['product_name']}. Get straight to the point. End with a period and new line."
        prompts.append(recommendation_prompt)
    
    # Run the model
    print(prompts)
    answers = base_runner(prompts, system_prompt=system_prompt)
    print(answers)

    return answers

def generate_questions(answers):
    system_prompt = "You are an expert on grocery products. You know all of the details about the products. You gave a recommendation to the customer. Now, write out possible questions that the customer had that led to your recommendation. For example, the customer might ask 'What should I get for my picnic?'"

    prompts = []
    for answer in answers:
        prompt = answer['output'] + '\n' + 'Write a question that the customer might have asked that led to your recommendation. End with a period and new line.'
        prompts.append(prompt)

    print(prompts)
    questions = base_runner(prompts, system_prompt=system_prompt)
    print(questions)

    return questions

def create_qa_dataset(questions, answers):
    # Write question and answer pairs to a jsonl file
    filepath = 'qa_dataset.jsonl'
    with jsonlines.open(filepath, 'w') as writer:
        for question, answer in zip(questions, answers):
            writer.write({
                'question': question,
                'answer': answer
            })
    return filepath

main()