import base64
import requests
from openai import OpenAI
from voyager import Voyager
import argparse 
from block2num import block2num

with open("./prompt/prompt_gpt4_first_round.txt", "r") as f:
  first_round_text = f.read()
with open("./prompt/prompt_gpt4_first_round2.txt", "r") as f:
  first_round_text2 = f.read()
with open("./prompt/prompt_gpt4_second_round.txt", "r") as f:
  second_round_text = f.read()

def round1(api_key=None, task=None):
  assert api_key is not None, "Please provide an API Key!"
  assert task is not None, "Please provide a task!"

  conversation_history = [
      {"role": "system", "content": "Hello, I am a chatbot that can talk about anything. What would you like to talk about today?"},
      {"role": "user", "content": f"You are an architect designing houses and buildings. Here is a building you should design: {task}. \n You should answer these questions below based on your design and imagination: \n {first_round_text}"}
  ]

  client = OpenAI(api_key=api_key)

  # Sending the message to the API including the entire conversation history
  response = client.chat.completions.create(
      model="gpt-4-1106-preview",       # can change to your own choice
      messages=conversation_history     # Include the conversation history
  )

  # Extracting the completion response
  first_round_output = response.choices[0].message.content

  conversation_history = [
      {"role": "system", "content": "Hello, I am a chatbot that can talk about anything. What would you like to talk about today?"},
      {"role": "user", "content": f"You are an architect designing houses and buildings. Here is a building you should design: {task}. \n You should answer these questions below based on your design and imagination: \n {first_round_text2}"}
  ]

  client = OpenAI(api_key=api_key)

  # Sending the message to the API including the entire conversation history
  response = client.chat.completions.create(
      model="gpt-4-1106-preview",       # can change to your own choice
      messages=conversation_history     # Include the conversation history
  )

  # Extracting the completion response
  first_round_output2 = response.choices[0].message.content

  return first_round_output, first_round_output2


def round2(first_round_output, first_round_output2, api_key=None):
  assert api_key is not None, "Please provide an API Key!"

  conversation_history = [
      {"role": "system", "content": "Hello, I am a chatbot that can talk about anything. What would you like to talk about today?"},
      {"role": "user", "content": first_round_text + first_round_text2},
      {"role": "assistant", "content": first_round_output + first_round_output2},
      {"role": "user", "content": second_round_text}
  ]

  client = OpenAI(api_key=api_key)

  # Sending the message to the API including the entire conversation history
  response = client.chat.completions.create(
      model="gpt-4-1106-preview",       # can change to your own choice
      messages=conversation_history     # Include the conversation history
  )

  # Extracting the completion response
  second_round_output = response.choices[0].message.content
  return second_round_output


def check(second_round_output):
  # Check if the output is well-formed
  if "Explain:" in second_round_output and "Plan:" in second_round_output and "Code:" in second_round_output:
    return True
  else:
    return False


def block_dict(second_round_output, api_key=None):
  assert api_key is not None, "Please provide an API Key!"

  # Extract the block name and number from the output
  conversation_history = [
      {"role": "system", "content": "Hello, I am a chatbot that can talk about anything. What would you like to talk about today?"},
      {"role": "user", "content": second_round_text},
      {"role": "assistant", "content": second_round_output},
      {"role": "user", "content": "From your last response, please list all the blocks you used in your code. You should answer in the format as: block_1, block_2, block_3, ..."}
  ]
  client = OpenAI(api_key=api_key)

  # Sending the message to the API including the entire conversation history
  response = client.chat.completions.create(
      model="gpt-4-1106-preview",       # can change to your own choice
      messages=conversation_history     # Include the conversation history
  )

  blocks_list = response.choices[0].message.content
  blocks_list = blocks_list.split(", ")
  blocks_dict = {"diamond_pickaxe": 1}
  for block in blocks_list:
    blocks_dict[block] = block2num(block)
  return blocks_dict


def main(api_key=None, task=None, mc_port=None):
  assert api_key is not None, "Please provide an API Key!"
  assert task is not None, "Please provide a task!"
  assert mc_port is not None, "Please provide a Minecraft port!"

  second_round_output = ""
  while not check(second_round_output):
    first_round_output, first_round_output2 = round1(api_key=api_key, task=task)
    second_round_output = round2(first_round_output, first_round_output2, api_key=api_key)
  
  blocks_dict = block_dict(second_round_output, api_key=api_key)

  voyager = Voyager(
    mc_port = int(mc_port),
    openai_api_key=api_key,
    reset_placed_if_failed=True,
    action_agent_task_max_retries=100,
    action_agent_show_chat_log=False,
    curriculum_agent_mode="manual",
    critic_agent_mode="manual",
  )

  # start lifelong learning
  voyager.learn(init_inventory=blocks_dict, init_message=second_round_output)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--api_key", type=str, default=None)
  parser.add_argument("--task", type=str, default=None)
  parser.add_argument("--mc_port", type=int, default=None)
  args = parser.parse_args()
  main(api_key=args.api_key, task=args.task, mc_port=args.mc_port)