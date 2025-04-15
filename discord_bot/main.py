import discord
from bot.bot import MessageRouterBot
from config import load_config

if __name__ == "__main__":
    config = load_config()
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

    bot = MessageRouterBot(intents=intents)
    bot.run(config['DISCORD_TOKEN'])
