import os
import discord
from dotenv import load_dotenv
from inference import Application

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)

@client.event
async def on_ready():
  print(f'We have logged in as {client.user.name}')


@client.event
async def on_message(message):
    # 다른 사용자로부터의 메시지만 응답 (봇 자신으로부터의 메시지는 무시)
    if message.author == client.user:
        return

    # await message.channel.send("""사건의 title과 판시사항을 'title :', '판시사항'과 같이 입력해주세요.""")

    # OpenAI API가 대답
    messages = f""" system : 한국말로만 대답하고 최대한 간결하고 알기쉽게 정리해줘.
        user : 사건의 제목과 판시사항을 보고 판결 결과와 그 이유를 예측해줘 \n
        {message.content} \n assistant :"""
    response = Application()(messages)
    # Send the response as a message
    await message.channel.send(response)

# start the bot
client.run(TOKEN)