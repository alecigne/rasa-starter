import argparse

from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core import utils
from bot_server_channel import BotServerInputChannel

def create_argument_parser():
    parser = argparse.ArgumentParser(description="run the bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-w", "--web", action='store_true',
                        help="run the bot in the web interface")
    group.add_argument("-c", "--cli", action='store_true',
                        help="run the bot in the CLI (default)")
    parser.add_argument("-n", "--no_nlu", action='store_true',
                        help="do not use NLU")
    return parser

def load_agent(no_nlu=False):
    if no_nlu:
        agent = Agent.load("dlg/model")
    else:
        interpreter = RasaNLUInterpreter("nlu/models/default/model_20000101-000000")
        agent = Agent.load("dlg/model", interpreter=interpreter)
    return agent

def run_cli(no_nlu=False):
    agent = load_agent(no_nlu)
    agent.handle_channel(ConsoleInputChannel())
    return agent

def run_web(no_nlu=False):
    agent = load_agent(no_nlu)
    channel = BotServerInputChannel(agent)
    agent.handle_channel(channel)

if __name__ == '__main__':
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()

    if args.web:
        run_web(args.no_nlu)
    elif args.cli:
        run_cli(args.no_nlu)
    else:
        run_cli(args.no_nlu)
