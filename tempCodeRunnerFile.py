son
        with open('YP_Finals.json') as f:
            data = json.load(f)

        reviews = [item['review'] for item in data]

        tokenizer = Tokenizer()

        # Corpus data normalized -> lower, split into strings by new line characters
        c_data = []
        for review in reviews:
            c_data.extend(review.lower().split("\n"))