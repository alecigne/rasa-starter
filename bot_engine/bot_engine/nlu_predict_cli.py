from rasa_nlu.model import Metadata, Interpreter
import json
import warnings

model_directory = './nlu/models/default/model_20000101-000000'
interpreter = Interpreter.load(model_directory)


def main():
    while True:
        user_input = input("------\nUser: ")
        intent = interpreter.parse(user_input)
        print(json.dumps(intent, indent=2))


if __name__ == "__main__":
    main()
