import argparse
import os
from data_utils import Dataset
from mle2 import CBR_WO_GED, build_ngrams  # or from yourmodule import ...

def predict(cbr_model, test_data, output_file):
    """
    Use the cbr_model to rewrite the tokens in test_data,
    and write them to output_file.
    """
    predictions = []

    for ex in test_data.examples:
        src_tokens = ex.src_tokens

        # We'll do a simple 1:1 rewriting for each token, ignoring n-gram context
        # (If you want to incorporate n-grams properly, you can adapt the logic.)
        new_tokens = []
        for token in src_tokens:
            # Convert the single token into an n-gram tuple of length 1, 2, etc.
            # For minimal example: we treat it as a single token (ngram=1).
            context = (token,)
            
            cbr_candidates = cbr_model[context]  # calls __getitem__
            if cbr_candidates:
                # Pick the target word that has the highest probability
                best_candidate = max(cbr_candidates.items(), key=lambda x: x[1])[0]
                new_tokens.append(best_candidate)
            else:
                # If no rewrite found, just keep original
                new_tokens.append(token)

        # Join rewritten tokens into a single string
        predictions.append(" ".join(new_tokens))

    # Write out predictions to file
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training dataset (for building CBR).")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test dataset (for predictions).")
    parser.add_argument("--output_file", type=str, default="predictions.txt",
                        help="Output file to write predictions.")
    parser.add_argument("--ngrams", type=int, default=1,
                        help="Number of n-grams for the CBR model.")
    args = parser.parse_args()

    # 1) "Train" -> build the model from the training data
    print(f"Loading training data from {args.train_file}")
    train_data = Dataset(raw_data_path=args.train_file)
    print(f"Building {args.ngrams}-gram CBR_WO_GED model...")
    cbr_model = CBR_WO_GED.build_model(train_data, ngrams=args.ngrams)

    # 2) Predict on test data
    print(f"Loading test data from {args.test_file}")
    test_data = Dataset(raw_data_path=args.test_file)

    print("Predicting on test data...")
    predict(cbr_model, test_data, args.output_file)
    print(f"Predictions written to {args.output_file}")


if __name__ == "__main__":
    main()
