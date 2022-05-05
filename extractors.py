import re
from numpy import ndarray, vectorize
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple
from nltk.stem import PorterStemmer

Word = Tuple[str, str]
WordArray = List[Word]

def read_stopwords_to_regex(filename: str) -> str:
    with open(filename) as file:
        text = file.read()

    regex = text.replace("\n", "|")

    if regex.endswith("|"):
        regex = regex[:-1]

    return regex

def fuse(arrayA: List[List[any]], arrayB: List[List[any]]) -> List[List[any]]:
    if arrayA is None:
        return arrayB
    return [a+b for a, b in zip(arrayA, arrayB)]


class Features:
    # Words -> List of the words before and after the target, in order, as strings.
    # WordsOnehot -> List of the words before and after the target, in order, as onehot vectors. Each vector is the size of the vocabulary.
    # Cats -> List of the categories of the words before and after the target, in order, as strings.
    # CatsOnehot -> List of the categories of the words before and after the target, in order, as onehot vectors. Each vector's length is the number of categories.
    # BagOfWords -> Vector given by CountVectorizer for the words before and after the target.
    # BagOfCats -> Vector given by CountVectorizer for the words before and after the target.
    # TargetCat -> The category of the target, as a string.
    # TargetCatOnehot -> The category of the target, as a onehot vector. The vector's length if the number of categories.
    # WordsIndex -> Indexes of the words before and after the target, in order. The indexes are taken from the word vocabulary.
    # CatsIndex -> Indexes of the categories of the words before and after the target, in order. The indexes are taken from the category vocabulary.
    # TargetCatIndex -> Index of the target category, taken from the category vocabulary.
    feature_types = {"Words", "WordsOnehot", "Cats", "CatsOnehot", "BagOfWords", "BagOfCats", "TargetCat", "TargetCatOnehot", "WordsIndex", "CatsIndex", "TargetCatIndex"}
    token_pattern: str = r"(?u)\b\S+\b"


    def __init__(self, word_features: List[WordArray], targets: List[Word]):
        self.words_array: List[List[str]] = []
        self.categories_array: List[List[str]] = []
        self.target_categories: List[str] = []
        self.target_classes: List[int] = []   

        for word_feature, target in zip(word_features, targets):
            # Save targets categories and classes
            self.target_classes.append(target[0])
            self.target_categories.append(target[1])
            
            # Save words and their categories
            word_temp = [word for word, _ in word_feature]
            cat_temp = [cat for _, cat in word_feature]
            self.words_array.append(word_temp)
            self.categories_array.append(cat_temp)
        pass

    def get_word_vocabulary(self) -> ndarray:
        # Join the words in each line
        lines = []
        for word_array in self.words_array:
            lines.append(" ".join(word_array))

        # Get the vocabulary
        vectorizer = CountVectorizer(lowercase=False, token_pattern=self.token_pattern).fit(lines)
        return vectorizer.get_feature_names_out()

    def get_category_vocabulary(self) -> ndarray:
        # Join the categories in each line
        lines = []
        for category_array in self.categories_array:
            lines.append(" ".join(category_array))

        # Get the vocabulary
        vectorizer = CountVectorizer(lowercase=False, token_pattern=self.token_pattern).fit(lines)
        return vectorizer.get_feature_names_out()

    def get_words(self, as_index: bool = False) -> List[List[any]]:
        if as_index:
            lines = self.words_array
            vocab = [""] + self.get_word_vocabulary().tolist()
            return [[vocab.index(word) for word in words] for words in lines]
        else:
            return self.words_array

    def get_categories(self, as_index: bool = False) -> List[List[any]]:
        if as_index:
            lines = self.categories_array
            vocab = [""] + self.get_category_vocabulary().tolist()
            return [[vocab.index(category) for category in categories] for categories in lines]
        else:
            return self.categories_array

    def get_bag_of_words(self) -> List[List[int]]:
        # Join the words in each line
        lines = []
        for word_array in self.words_array:
            lines.append(" ".join(word_array))

        return CountVectorizer(lowercase=False, token_pattern=self.token_pattern).fit_transform(lines).toarray().tolist()

    def get_bag_of_categories(self) -> List[List[int]]:
        # Join the categories in each line
        lines = []
        for category_array in self.categories_array:
            lines.append(" ".join(category_array))

        return CountVectorizer(lowercase=False, token_pattern=self.token_pattern).fit_transform(lines).toarray().tolist()

    def get_target_categories(self, as_index: bool = False) -> List[any]:
        if as_index:
            categories = self.target_categories
            vocab = [""] + self.get_category_vocabulary().tolist()
            return [vocab.index(category) for category in categories]
        else:
            return self.target_categories

    def get_categories_onehot(self) -> List[List[int]]:
        categories_onehot = []
        lines_categories = self.get_categories()
        vocab = self.get_category_vocabulary()

        for line_categories in lines_categories:
            line_onehot = []
            for category in line_categories:
                onehot = ((vocab == category) * 1).tolist()
                line_onehot.extend(onehot)
            categories_onehot.append(line_onehot)

        return categories_onehot

    def get_target_categories_onehot(self) -> List[List[int]]:
        categories_onehot = []
        categories = self.get_target_categories()
        vocab = self.get_category_vocabulary()

        for category in categories:
            onehot = ((vocab == category) * 1).tolist()
            categories_onehot.append(onehot)

        return categories_onehot

    def get_words_onehot(self) -> List[List[int]]:
        words_onehot = []
        lines_words = self.get_words()
        vocab = self.get_word_vocabulary()

        for line_words in lines_words:
            line_onehot = []
            for word in line_words:
                onehot = ((vocab == word)*1).tolist()
                line_onehot.extend(onehot)
            words_onehot.append(line_onehot)

        return words_onehot

    def get_features(self, features_to_select: List[str]) -> List[List[any]]:
        features: List[List[any]] = None

        for feature_to_select in features_to_select:
            if feature_to_select == "Words":
                features = fuse(features, self.get_words())
            elif feature_to_select == "WordsOnehot":
                features = fuse(features, self.get_words_onehot())
            elif feature_to_select == "Cats":
                features = fuse(features, self.get_categories())
            elif feature_to_select == "CatsOnehot":
                features = fuse(features, self.get_categories_onehot())
            elif feature_to_select == "BagOfWords":
                features = fuse(features, self.get_bag_of_words())
            elif feature_to_select == "BagOfCats":
                features = fuse(features, self.get_bag_of_categories())
            elif feature_to_select == "TargetCat":
                features = [f+[c] for f, c in zip(features, self.get_target_categories())]
            elif feature_to_select == "TargetCatOnehot":
                features = fuse(features, self.get_target_categories_onehot())
            elif feature_to_select == "WordsIndex":
                features = fuse(features, self.get_words(as_index=True))
            elif feature_to_select == "CatsIndex":
                features = fuse(features, self.get_categories(as_index=True))
            elif feature_to_select == "TargetCatIndex":
                features = [f+[c] for f, c in zip(features, self.get_target_categories(as_index=True))]

        return features

    def get_targets(self) -> List[int]:
        return self.target_classes


class FeatureExtractor:

    def __init__(self, n_word: int = 2, add_padding: bool = False) -> None:
        self.n_word: int = n_word
        self.add_padding: bool = add_padding

    
    def extract(self, before_target: List[WordArray], after_target: List[WordArray], target_info: List[Word]) -> Features:
        # Select n_word before and after the target
        word_features: List[WordArray] = []
        for words_before, words_after in zip(before_target, after_target):
            temp: WordArray = []

            # Words before the target
            if len(words_before) < self.n_word and self.add_padding:
                temp = [("","") for _ in range(0, self.n_word - len(words_before))]
            temp.extend(words_before[-self.n_word:])

            # Words after the target
            temp.extend(words_after[-self.n_word:])
            if len(words_after) < self.n_word and self.add_padding:
                temp.extend([("","") for _ in range(0, self.n_word - len(words_after))])

            word_features.append(temp)

        return Features(word_features, target_info)


class WordExtractor:
    input_type = { "filename", "file", "content"}

    whitespaces: str = r"[ \[\]=\n\r]+"
    punctuation: str = r"\W+"

    target_pattern: str = r"interest[s]?_[123456]"
    token_pattern: str = r"(?u)\b\S+\b"

    line_separators: str = r"\$\$"
    categories_separator: str = r"/"
    class_separator: str = r"_"

    category_to_ignore = r"\$|:|#|``"

    def __init__(self, stopwords: str = None, stem: bool = False, debug: bool = False):
        self.debug: bool = debug
        self.stem: bool = stem

        self.ignore_stopwords: bool = False
        if stopwords is not None:
            self.ignore_stopwords = True
            self.stopwords: str = stopwords

        self.formatter: CountVectorizer = CountVectorizer(lowercase=False, token_pattern=self.token_pattern)
        self.stemmer: PorterStemmer = PorterStemmer()
        pass


    def __extract_word_info(self, word: str) -> Word:
        # Ignore whitespaces
        if re.fullmatch(self.whitespaces, word):
            if self.debug:
                print("Discarding word: ", word)
            return None

        # Split the word
        temp = re.split(self.categories_separator, word)

        # Ignore words without categories
        if len(temp) != 2:
            if self.debug:
                print("Discarding word: ", word)
            return None
        
        # Check for a bunch of things
        if re.fullmatch(self.punctuation, temp[0]):
            if self.debug:
                print("Ignoring " + temp[0] + " because it is punctuation")
            return None
        if re.fullmatch(self.category_to_ignore, temp[1]):
            if self.debug:
                print("Ignoring " + temp[0] + " because it is a " + temp[1])
            return None
        if self.ignore_stopwords and re.fullmatch(self.stopwords, temp[0]):
            return None
        if temp[0] == '' or temp[1] == '':
            return None

        # Stem the word
        if not re.fullmatch(self.target_pattern, temp[0]) and self.stem:
            temp[0] = self.stemmer.stem(temp[0])

        # Formatting word
        try:
            formatted_word = self.formatter.fit([temp[0]]).get_feature_names_out().tolist()
            if len(formatted_word) != 1:
                raise ValueError("Error when formatting ->" + temp[0] + "<-")
        except ValueError:
            raise ValueError("Error when formatting ->" + temp[0] + "<-")


        formatted_word = formatted_word[0]
        if formatted_word != temp[0] and self.debug:
            print("Changing " + temp[0] + " to " + formatted_word)

        # Formatting category
        try:
            formatted_cat = self.formatter.fit([temp[1]]).get_feature_names_out().tolist()
            if len(formatted_cat) != 1:
                raise ValueError("Error when formatting ->" + temp[1] + "<-")
        except ValueError: 
            raise ValueError("Error when formatting ->" + temp[1] + "<-")

        formatted_cat = formatted_cat[0]
        if formatted_cat != temp[1] and self.debug:
            print("Changing " + temp[1] + " to " + formatted_cat)

        return formatted_word, formatted_cat


    def __clean(self, text: str) -> str:
        # Remove whitespaces
        return re.sub(self.whitespaces, " ", text)


    def __split_text_into_lines(self, text: str) -> List[str]:
        # Split the text into lines
        return re.split(self.line_separators, text)


    def __extract_words_from_line(self, line: str) -> Tuple[WordArray, WordArray, Word]:
        before: WordArray = []
        after: WordArray = []
        target_word_info: Word = None

        # Split the line into words
        line = line.strip()
        words = re.split(self.whitespaces, line)

        # Extract words before the target
        for word in words:
            # Extract the information of the word
            word_info = self.__extract_word_info(word)

            if word_info is not None:
                # If target, save it
                if re.fullmatch(self.target_pattern, word_info[0]) and target_word_info is None:
                    c = re.split(self.class_separator, word_info[0])[1]
                    target_word_info = (c, word_info[1])
                # If target has been found, save in after
                elif target_word_info is not None:
                    after.append(word_info)
                # If target has not been found, save in before
                elif target_word_info is None:
                    before.append(word_info)
                # Skip the word
                elif self.debug:
                    print("Discarding word: ", word)

        # If their is no target is the line -> skip it
        if target_word_info is None:
            if self.debug:
                print("Discarding line: " + line)
            return None, None, None

        return before, after, target_word_info


    def extract(self, input: any, type = "filename") -> Tuple[List[WordArray], List[WordArray], List[Word]]:
        if type not in self.input_type:
            print("ERROR - Wrong input type")
            return None

        # Read the file or content
        text: str = ""
        if type == "filename":
            with open(input) as file:
                text = file.read()
        elif type == "file":
            text = input.read()
        else:
            text = input

        cleaned_text: str = self.__clean(text)
        lines: List[str] = self.__split_text_into_lines(cleaned_text)

        # Extract info from every line
        befores: List[WordArray] = []
        afters: List[WordArray] = []
        targets: List[Word] = []

        for line in lines:
            before, after, target = self.__extract_words_from_line(line)

            # Save info
            if before is not None:
                befores.append(before)
                afters.append(after)
                targets.append(target)


        return befores, afters, targets


def extract_features(word_extractor: WordExtractor, feature_extractor: FeatureExtractor, input: any, input_type:any) -> Features:
    words_before, words_after, targets = word_extractor.extract(input, input_type)
    return feature_extractor.extract(words_before, words_after, targets)