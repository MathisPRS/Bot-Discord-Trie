import discord
from bot.bot import MessageRouterBot
from config import DISCORD_TOKEN

if __name__ == "__main__":
    token_discord = DISCORD_TOKEN
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

    bot = MessageRouterBot(intents=intents)
    bot.run(token_discord)
