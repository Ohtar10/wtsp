import os
import pickle
import argparse
import logging
import numpy as np
import modin.pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from gensim.models import Doc2Vec
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser(description="This program will transform the provided documents into embeddings using the specified Doc2Vec model.")

parser.add_argument("documents_path", type=str, help="Path to the raw documents")
# Required arguments
required_named = parser.add_argument_group('required named arguments')
required_named.add_argument("--d2v-model", type=str, help="Path to the Gensim Doc2Vec model", required=True)
required_named.add_argument("--output", type=str, help='Path to save the transformed embeddings', required=True)

# Optional arguments
parser.add_argument("--train-test-split", action='store_true', help='Perform a train test split before saving the results')
parser.add_argument("--test-size", type=float, help='Size of the test split if requrested, default to 0.3', default=0.3)

args = parser.parse_args()

d2v_model = Doc2Vec.load(args.d2v_model)

documents = pd.read_parquet(args.documents_path, engine="pyarrow")
logging.info(f"Processing {documents.shape[0]} documents")

# Parsing categories
categories = documents.categories.apply(lambda cat: cat.split(";")).values.tolist()
categories_encoder = MultiLabelBinarizer()
categories_encoder.fit(categories)


# Transforming into embeddings
logging.info(f"Transforming into embeddings")
tokenizer = RegexpTokenizer(r'\w+')
y = categories_encoder.transform(categories)
X = documents.apply(lambda row: d2v_model.infer_vector([word.lower() for word in tokenizer.tokenize(row['document'])]), axis=1)._to_pandas()
X = X.apply(lambda x: x[0], axis=1, result_type='expand')

logging.info("Saving results...")
os.makedirs(args.output, exist_ok=True)
result_path = f"{args.output}/document_embeddings.npz"

if args.train_test_split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
    np.savez_compressed(result_path, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
else:
    np.savez_compressed(result_path, x=X, y=y)

category_encoder_path = f"{args.output}/category_encoder.model"
with open(category_encoder_path, 'wb') as file:
    pickle.dump(categories_encoder, file)

logging.info(f"Process finished.")
print(f"Results saved at: {args.output}")