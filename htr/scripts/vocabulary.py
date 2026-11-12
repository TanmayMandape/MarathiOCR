
from dataset_loader import load_annotations


class CharacterVocabulary:

    def __init__(self, characters):

        # Reserve index 0 for the CTC blank token
        self.blank_token = "<BLANK>"

        self.characters = sorted(set(characters))

        self.char_to_index = {
            char: index + 1
            for index, char in enumerate(self.characters)
        }

        self.index_to_char = {
            index: char
            for char, index in self.char_to_index.items()
        }

        self.vocab_size = len(self.characters) + 1

    def encode(self, text):
        """Convert Marathi text into numerical indices."""

        return [
            self.char_to_index[char]
            for char in text
        ]

    def decode(self, indices):
        """Convert numerical indices back into text."""

        return "".join(
            self.index_to_char[index]
            for index in indices
            if index != 0 and index in self.index_to_char
        )


def build_vocabulary():

    samples = load_annotations("train")

    all_characters = set()

    for sample in samples:
        all_characters.update(sample["text"])

    vocabulary = CharacterVocabulary(all_characters)

    print("Training samples:", len(samples))
    print("Unique characters:", len(vocabulary.characters))
    print("Vocabulary size including CTC blank:", vocabulary.vocab_size)

    return vocabulary


if __name__ == "__main__":

    vocab = build_vocabulary()

    print("\nFirst 30 characters:")
    print(vocab.characters[:30])

    test_word = "दत्तात्रेय"

    encoded = vocab.encode(test_word)
    decoded = vocab.decode(encoded)

    print("\nTest word:", test_word)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

    print("\nVocabulary test:", "PASSED" if decoded == test_word else "FAILED")
