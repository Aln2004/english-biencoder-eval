import pandas as pd
import pickle

with open("englishlleqa.pickle", "rb") as engpickle:
	eng_dataset = pickle.load(engpickle)
	print("The structure of the data in the pickle is as follows : ")
	print(eng_dataset)
	print("The structure is thus known")
	train_df = pd.DataFrame(eng_dataset[1]["train"])
	val_df = pd.DataFrame(eng_dataset[1]["validation"])
	test_df = pd.DataFrame(eng_dataset[1]["test"])
	corpus_df = pd.DataFrame(eng_dataset[0]["corpus"])

	train_df.to_json("english_questions_train.json", orient="records", indent=2)
	val_df.to_json("english_questions_val.json", orient="records", indent=2)
	test_df.to_json("english_questions_test.json", orient="records", indent=2)

	corpus_df.to_json("english_articles.json", orient="records", indent=2)
