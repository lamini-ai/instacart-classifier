# Finetuning Llama V2 on Product IDs

Consider that we want to be able to answer the following question.

```
Q: “What would go well with a chicken fried steak?”
```

What would be an ideal answer?  It should include common sense suggestions from a LLM, and also suggests real products from a catalog.  

```
Baked beans and coleslaw are also excellent choices to accompany chicken fried steak, and they are popular side dishes that offer a nice balance of flavors and textures:

 * Baked Beans: Baked beans provide a sweet and savory component to your meal. Their slightly sweet and smoky flavors can complement the rich and crispy chicken fried steak. The combination of tender beans and a flavorful sauce is a classic choice. 
(closest product: Refried Beans: id 35113)


 * Coleslaw: Coleslaw offers a refreshing and crunchy contrast to the hearty and fried chicken steak. The crisp, shredded cabbage mixed with a creamy and tangy dressing adds a light and zesty element to your plate, which can help cut through the richness of the steak.
(closest product: Freshly Shredded Angel Hair Cole Slaw: id 47265)


Together, baked beans and coleslaw create a well-rounded meal with a mix of flavors and textures that can be a delightful pairing with chicken fried steak.
```
This is a github repo with scripts showing how to create finetuning training data with product ids like the example above.  It does this in 4 steps.

1. Generate product descriptions for the entire catalog
2. Train an LLM classifier that maps from a product description to a product id on all descriptions from step 1.
3. Generate training data like the example above that includes the real product ids.
4. Finetune Llama v2 on the data from 3.  

## Step 1: Product Descriptions

First, generate detailed descriptions of each of the products in the catalog using an LLM.  For example, here is the detailed description of Fudge Covered Crackers produced by Llama 2 7B.  

Fudge Covered Crackers are a delicious and indulgent snack that combines the rich, creamy flavor of fudge with the crunch of crispy crackers.  Made with high-quality chocolate and real butter, these crackers are perfect for satisfying any sweet tooth.  Unlike other fudge-covered crackers on the market, ours are baked fresh in small batches to ensure maximum flavor and freshness.

## Step 2: Product Description Classifier

In this step we train a text classifier that maps from a product description (typically 3-5 sentences) to a product id.  We start from the original product description, then use an LLM to generate several variants of that product description.  

Here are two example descriptions of Harvarti Cheese:

Rich, buttery Havarti from Georgia's water buffalo milk is a semi-soft, creamy cheese with a delicate, nutty flavor.  Unlike other cheeses, Havarti is not aged for a long time, resulting in a fresher, more delicate taste.  This unique cheese is a popular choice for those looking for a mild, yet flavorful cheese that can be enjoyed on its own or used in a variety of dishes.

Creamy and tangy Havarti cheese made from cow's milk in the Caucasus region has a fresh, delicate taste that is unlike other cheeses.  Unlike other cheeses, Havarti is not aged for a long time, resulting in a milder flavor that is perfect for snacking or cooking.  With its smooth texture and creamy consistency, Havarti is a versatile cheese that can be enjoyed on its own or used in a variety of dishes.

We use an LLM to generate embeddings for each of these descriptions.  Then we train a classifier on top of these embeddings to assign a product id.  

## Step 3: Training Data
The next step is to make training data for our LLM that includes correct product ids.  The base LLM is already able to make recommendations using common sense.  However, it doesn’t know about real products in the catalog.  

First, we ask the LLM to generate questions that customers might ask to recommend products.  For example: “What would go well with a chicken fried steak?”  .   

Next we ask the base LLM to generate recommendations for these questions.  We do this using Lamini’s JSONformer interface, which forces the LLM to output the recommendations in a structured format.  
For example, the query: “What would go well with a chicken fried steak?” generates the response:

```
{
    “recommendation_1”: “Baked Beans”,
    “recommendation_2”: “Coleslaw”
}
```

At this point we have a common sense recommendation, but we don’t have the product id.  We get it from the classifier.  We expand the generic recommendations, e.g. “Baken Beans” into a product description: 

Baked beans are a type of canned bean that is cooked in a sweet and tangy sauce and often served as a side dish.  They are made from haricot beans, which are small, white beans that are high in fiber and protein.  Baked beans are a popular addition to many meals, particularly in American cuisine, and can be served with a variety of meats, such as chicken or steak.

Then we run the classifier on this, which finds the related product: “Refried Beans: id 35113”

Now, we can put all of this information together into the training example:

```
Baked beans and coleslaw are also excellent choices to accompany chicken fried steak, and they are popular side dishes that offer a nice balance of flavors and textures:

 * Baked Beans: Baked beans provide a sweet and savory component to your meal. Their slightly sweet and smoky flavors can complement the rich and crispy chicken fried steak. The combination of tender beans and a flavorful sauce is a classic choice. 
(closest product: Refried Beans: id 35113)


 * Coleslaw: Coleslaw offers a refreshing and crunchy contrast to the hearty and fried chicken steak. The crisp, shredded cabbage mixed with a creamy and tangy dressing adds a light and zesty element to your plate, which can help cut through the richness of the steak.
(closest product: Freshly Shredded Angel Hair Cole Slaw: id 47265)


Together, baked beans and coleslaw create a well-rounded meal with a mix of flavors and textures that can be a delightful pairing with chicken fried steak.
```

## Step 4: Finetuning

Finally, we finetune Llama v2 on data in this format.  It already knows how to make common sense suggestions.  Fine tuning is embedding the knowledge of the product ids into the model.

Note that this is very computationally intensive.  To cover a product catalog with 50,000 items in it, we should train on at least 10 example questions per product.  This implies 500,000 training examples.  

We recommend allocating 8 GPUs to cover training and data generation time.
