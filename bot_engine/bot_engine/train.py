import argparse

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.fallback import FallbackPolicy

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config


def create_argument_parser():
    parser = argparse.ArgumentParser(description="train the models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dlg", action='store_true',
                        help="train the dialogue model only")
    group.add_argument("-n", "--nlu", action='store_true',
                        help="train the nlu model only")
    return parser


def train_dlg(domain_file="domain.yml",
              training_data_file="dlg/data/stories.md",
              model_path="dlg/model"):

    fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                              core_threshold=0.8,
                              nlu_threshold=0.8)

    agent = Agent(domain_file, policies=[MemoizationPolicy(
        max_history=3), KerasPolicy(), fallback])

    training_data = agent.load_data(training_data_file)
    agent.train(
        training_data,
        epochs=300,
        batch_size=100,
        validation_split=0.2)

    agent.persist(model_path)
    return agent


def train_nlu():
    training_data = load_data('nlu/data/training')
    trainer = Trainer(config.load("nlu_config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist(
        './nlu/models', fixed_model_name='model_20000101-000000')


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()
    
    if args.dlg:
        print("=== TRAINING DIALOGUE MODEL ===")
        train_dlg()
    elif args.nlu:
        print("=== TRAINING NLU MODEL ===")
        train_nlu()
    else:
        print("=== TRAINING DIALOGUE MODEL ===")
        train_dlg()
        print("=== TRAINING NLU MODEL ===")
        train_nlu()
